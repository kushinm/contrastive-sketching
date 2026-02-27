#!/usr/bin/env python3
"""
Environment compatibility checker for Contrastive CLIPasso.

Run this BEFORE installation to diagnose issues:
    python check_env.py

Run AFTER installation to verify everything works:
    python check_env.py --full
"""

import sys
import os
import subprocess
import shutil
import re
from pathlib import Path

# ─── ANSI colors ──────────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{'─'*60}\n  {msg}\n{'─'*60}{RESET}")


# ─── Version parsing helper ──────────────────────────────────────────

def parse_version(v: str):
    """Parse '1.2.3' into tuple (1, 2, 3) for comparison."""
    parts = re.findall(r'\d+', v.split('+')[0])  # strip +cu124 etc
    return tuple(int(x) for x in parts[:3])


# ─── Checks ──────────────────────────────────────────────────────────

def check_python():
    header("Python")
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    
    if v.major != 3:
        fail(f"Python {ver_str} — need Python 3.x")
        return False
    
    if v.minor < 9:
        fail(f"Python {ver_str} — need >= 3.9 for PyTorch 2.5+ with CUDA 12.x")
        warn("Python 3.8 was dropped by PyTorch 2.5. Use Python 3.10 or 3.11.")
        return False
    elif v.minor > 12:
        warn(f"Python {ver_str} — not yet tested. 3.10-3.12 recommended.")
        return True
    else:
        ok(f"Python {ver_str}")
        return True


def check_cuda():
    header("CUDA")
    
    # Check nvidia-smi (driver)
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        warn("nvidia-smi not found — no GPU or driver not installed")
        warn("You can still run on CPU (slower) or with CPU-only diffvg + GPU PyTorch")
        return None
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                ok(f"GPU: {parts[1].strip() if len(parts) > 1 else 'unknown'}")
                ok(f"Driver: {parts[0].strip()}")
    except Exception:
        pass
    
    # Check CUDA toolkit version
    nvcc = shutil.which("nvcc")
    cuda_version = None
    if nvcc:
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                cuda_version = match.group(1)
                cuda_major = int(cuda_version.split('.')[0])
                cuda_minor = int(cuda_version.split('.')[1])
                ok(f"CUDA toolkit: {cuda_version}")
                
                if cuda_major >= 12:
                    warn(f"CUDA {cuda_version} — diffvg CUDA build may fail (written for CUDA 10/11)")
                    warn("Recommendation: build diffvg with DIFFVG_CUDA=0 (CPU rendering)")
                    warn("This is fine — SVG rendering is fast on CPU; CLIP uses GPU via PyTorch")
        except Exception:
            pass
    else:
        warn("nvcc not found — CUDA toolkit not on PATH")
        warn("PyTorch can still use GPU via bundled CUDA runtime")
        warn("diffvg should be built CPU-only (DIFFVG_CUDA=0)")
    
    # Check CUDA_HOME / CUDA_PATH
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        ok(f"CUDA_HOME: {cuda_home}")
    else:
        # Check common locations
        for path in ["/usr/local/cuda", "/usr/local/cuda-12.6", "/usr/local/cuda-12"]:
            if os.path.isdir(path):
                warn(f"CUDA_HOME not set, but found {path}")
                warn(f"Consider: export CUDA_HOME={path}")
                break
    
    return cuda_version


def check_cmake():
    header("CMake (needed for diffvg)")
    cmake = shutil.which("cmake")
    if not cmake:
        fail("cmake not found — required to build diffvg")
        warn("Install: sudo apt install cmake  OR  pip install cmake")
        return False
    
    try:
        result = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        match = re.search(r'version (\d+\.\d+\.\d+)', result.stdout)
        if match:
            ver = parse_version(match.group(1))
            if ver >= (3, 12):
                ok(f"cmake {match.group(1)}")
                return True
            else:
                fail(f"cmake {match.group(1)} — need >= 3.12")
                return False
    except Exception:
        fail("Could not determine cmake version")
        return False


def check_pytorch():
    header("PyTorch")
    try:
        import torch
    except ImportError:
        fail("PyTorch not installed")
        warn("Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return False
    
    ver = parse_version(torch.__version__)
    ok(f"torch {torch.__version__}")
    
    # Check version is recent enough for CUDA 12
    if ver < (2, 0):
        fail(f"PyTorch {torch.__version__} is too old for CUDA 12.x — need >= 2.4")
        return False
    elif ver < (2, 4):
        warn(f"PyTorch {torch.__version__} — cu124 builds available from 2.4+")
        warn("Upgrade recommended: pip install torch>=2.4 --index-url https://download.pytorch.org/whl/cu124")
    
    # Check CUDA support in PyTorch
    if torch.cuda.is_available():
        ok(f"CUDA available in PyTorch: {torch.version.cuda}")
        ok(f"GPU: {torch.cuda.get_device_name(0)}")
        ok(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Warn about CUDA version mismatch
        pt_cuda = torch.version.cuda
        if pt_cuda:
            pt_major = int(pt_cuda.split('.')[0])
            if pt_major < 12:
                warn(f"PyTorch built with CUDA {pt_cuda} but system has CUDA 12.x")
                warn("This usually works (driver forward compat) but cu124 build is better")
    else:
        warn("CUDA not available in PyTorch — will run on CPU")
        warn("For GPU support: pip install torch --index-url https://download.pytorch.org/whl/cu124")
    
    # Check torchvision
    try:
        import torchvision
        ok(f"torchvision {torchvision.__version__}")
    except ImportError:
        fail("torchvision not installed (required)")
        return False
    
    return True


def check_numpy():
    header("NumPy")
    try:
        import numpy as np
    except ImportError:
        fail("numpy not installed")
        return False
    
    ver = parse_version(np.__version__)
    ok(f"numpy {np.__version__}")
    
    if ver >= (2, 0):
        fail("numpy >= 2.0 has ABI breaking changes — PyTorch and scikit-image may crash")
        warn("Fix: pip install 'numpy>=1.23,<2.0'")
        return False
    elif ver < (1, 22):
        warn(f"numpy {np.__version__} is old — some deps need >= 1.22")
    
    return True


def check_pillow():
    header("Pillow")
    try:
        import PIL
        from PIL import Image
    except ImportError:
        fail("Pillow not installed")
        return False
    
    ver = parse_version(PIL.__version__)
    ok(f"Pillow {PIL.__version__}")
    
    # Check for BICUBIC deprecation
    if ver >= (10, 0):
        # PIL.Image.BICUBIC still works as alias but is deprecated
        if not hasattr(Image, 'Resampling'):
            warn("Pillow 10+ detected but missing Resampling — unexpected version")
        else:
            ok("Resampling.BICUBIC available (Pillow 10+ compatible)")
    
    return True


def check_clip():
    header("CLIP")
    try:
        import clip
        ok("CLIP package found")
        
        # Check it can list models
        models = clip.available_models()
        ok(f"Available models: {', '.join(models[:4])}...")
        return True
    except ImportError:
        fail("CLIP not installed")
        warn("Install: pip install git+https://github.com/openai/CLIP.git")
        return False
    except Exception as e:
        warn(f"CLIP installed but error: {e}")
        return False


def check_diffvg():
    header("diffvg / pydiffvg")
    try:
        import pydiffvg
        ok("pydiffvg importable")
        
        # Check if it can actually render
        import torch
        try:
            pydiffvg.set_use_gpu(False)  # test CPU path
            ok("pydiffvg CPU rendering available")
        except Exception as e:
            warn(f"pydiffvg imported but rendering failed: {e}")
        
        # Check GPU support
        if torch.cuda.is_available():
            try:
                pydiffvg.set_use_gpu(True)
                ok("pydiffvg GPU rendering available")
            except Exception:
                warn("pydiffvg GPU rendering not available — CPU rendering will be used")
                warn("This is fine: SVG rendering is fast on CPU")
        
        return True
    except ImportError:
        fail("pydiffvg not installed — see install_diffvg.sh")
        return False
    except Exception as e:
        fail(f"pydiffvg import error: {e}")
        return False


def check_other_deps():
    header("Other Dependencies")
    all_ok = True
    
    deps = {
        "scipy": ("1.6", None),
        "skimage": ("0.18", None),      # import name for scikit-image
        "matplotlib": ("3.4", None),
        "imageio": ("2.9", None),
        "ftfy": (None, None),
        "regex": (None, None),
        "tqdm": (None, None),
        "svgwrite": (None, None),
        "svgpathtools": (None, None),
    }
    
    for name, (min_ver, max_ver) in deps.items():
        try:
            mod = __import__(name)
            ver = getattr(mod, '__version__', 'unknown')
            ok(f"{name} {ver}")
        except ImportError:
            fail(f"{name} not installed")
            all_ok = False
    
    return all_ok


def check_u2net():
    header("U2Net (optional — for background masking)")
    weights_path = Path("U2Net_/saved_models/u2net.pth")
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / 1e6
        ok(f"U2Net weights found ({size_mb:.0f} MB)")
        return True
    else:
        warn("U2Net weights not found at U2Net_/saved_models/u2net.pth")
        warn("Only needed if using --mask_object flag")
        warn("Download: gdown 'https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ' -O U2Net_/saved_models/")
        return True  # optional


# ─── Compatibility matrix summary ────────────────────────────────────

def print_recommended_install(cuda_version):
    header("Recommended Installation")
    
    print(f"""
  Based on your system (CUDA {cuda_version or 'N/A'}), here's the recommended stack:

  {BOLD}# 1. Create environment with Python 3.10{RESET}
  conda create -n clipasso python=3.10 -y
  conda activate clipasso

  {BOLD}# 2. Install PyTorch with CUDA 12.4 support{RESET}
  #    (cu124 builds work with CUDA 12.6 driver — forward compatible)
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

  {BOLD}# 3. Install CLIP{RESET}
  pip install git+https://github.com/openai/CLIP.git

  {BOLD}# 4. Install diffvg (CPU-only build — safe for CUDA 12){RESET}
  #    See install_diffvg.sh for details
  bash install_diffvg.sh

  {BOLD}# 5. Install remaining deps{RESET}
  pip install -r requirements.txt

  {BOLD}# 6. (Optional) U2Net weights for background masking{RESET}
  pip install gdown
  gdown 'https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ' -O U2Net_/saved_models/
""")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    full_check = "--full" in sys.argv
    
    print(f"\n{BOLD}Contrastive CLIPasso — Environment Check{RESET}")
    print(f"{'='*60}")
    
    results = {}
    
    results['python'] = check_python()
    cuda_version = check_cuda()
    results['cmake'] = check_cmake()
    
    if full_check:
        results['pytorch'] = check_pytorch()
        results['numpy'] = check_numpy()
        results['pillow'] = check_pillow()
        results['clip'] = check_clip()
        results['diffvg'] = check_diffvg()
        results['other'] = check_other_deps()
        results['u2net'] = check_u2net()
    else:
        print(f"\n  {YELLOW}Run with --full to check all installed packages{RESET}")
    
    # Summary
    header("Summary")
    all_ok = all(v for v in results.values() if v is not None)
    
    if all_ok and full_check:
        print(f"\n  {GREEN}{BOLD}All checks passed! Ready to run.{RESET}")
    elif all_ok:
        print(f"\n  {GREEN}Pre-install checks passed.{RESET}")
        print_recommended_install(cuda_version)
    else:
        failed = [k for k, v in results.items() if v is False]
        print(f"\n  {RED}Issues found: {', '.join(failed)}{RESET}")
        print_recommended_install(cuda_version)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())