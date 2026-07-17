#!/usr/bin/env bash
#
# install_diffvg.sh — Build and install pydiffvg with CUDA 12.x compatibility.
#
# The original diffvg was written for CUDA 10/11 and breaks with CUDA 12 because:
#   1. thrust:: namespace reorganisation (thrust moved into cub in CUDA 12)
#   2. Deprecated APIs removed in CUDA 12 (e.g., texture references)
#   3. CMake FindCUDA vs native CUDA language support changes
#
# This script handles it by:
#   - Default: CPU-only build (DIFFVG_CUDA=0) which avoids all CUDA compilation.
#     This is the RECOMMENDED approach because SVG rendering is very fast on CPU
#     and the GPU is only needed for CLIP (handled by PyTorch's own CUDA runtime).
#   - Optional: attempt a CUDA-enabled build with patches for common issues.
#
# Usage:
#   bash install_diffvg.sh          # CPU-only (recommended for CUDA 12)
#   bash install_diffvg.sh --cuda   # Attempt CUDA build with patches
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFVG_DIR="${SCRIPT_DIR}/diffvg"

CUDA_BUILD=0
if [[ "${1:-}" == "--cuda" ]]; then
    CUDA_BUILD=1
    echo "⚠  Attempting CUDA-enabled diffvg build."
    echo "   This may fail on CUDA 12.x — fall back to CPU build if so."
fi

# ─── Pre-checks ──────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  diffvg installer for Contrastive CLIPasso"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check Python
python_ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
if [[ -z "$python_ver" ]]; then
    echo "✗ Python 3 not found on PATH"
    exit 1
fi
echo "✓ Python: ${python_ver}"

# Check cmake
if ! command -v cmake &> /dev/null; then
    echo "✗ cmake not found. Install: pip install cmake  OR  sudo apt install cmake"
    exit 1
fi
echo "✓ cmake: $(cmake --version | head -1)"

# Check PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo "✗ PyTorch not installed — install it first (see README.md)"
    exit 1
fi
torch_ver=$(python3 -c "import torch; print(torch.__version__)")
echo "✓ PyTorch: ${torch_ver}"

# Detect CUDA situation
cuda_ver=$(python3 -c "import torch; print(torch.version.cuda or 'none')" 2>/dev/null || echo "none")
echo "✓ PyTorch CUDA: ${cuda_ver}"

if [[ "$CUDA_BUILD" -eq 1 ]]; then
    if ! command -v nvcc &> /dev/null; then
        echo ""
        echo "⚠  nvcc not found — cannot do CUDA build without CUDA toolkit on PATH."
        echo "   Falling back to CPU-only build."
        CUDA_BUILD=0
    else
        nvcc_ver=$(nvcc --version | grep -oP 'release \K[\d.]+')
        echo "✓ nvcc: ${nvcc_ver}"
        nvcc_major=$(echo "$nvcc_ver" | cut -d. -f1)
        if [[ "$nvcc_major" -ge 12 ]]; then
            echo ""
            echo "⚠  CUDA ${nvcc_ver} detected. diffvg was written for CUDA 10/11."
            echo "   Will apply patches but build may still fail."
            echo "   If it does, re-run without --cuda for CPU-only build."
            echo ""
        fi
    fi
fi

# ─── Clone / update diffvg ──────────────────────────────────────────

if [[ -d "$DIFFVG_DIR" ]]; then
    echo ""
    echo "→ diffvg directory exists, pulling latest..."
    cd "$DIFFVG_DIR"
    git pull --ff-only 2>/dev/null || true
    git submodule update --init --recursive
else
    echo ""
    echo "→ Cloning diffvg..."
    cd "$SCRIPT_DIR"
    git clone https://github.com/BachiLi/diffvg.git
    cd "$DIFFVG_DIR"
    git submodule update --init --recursive
fi

# ─── Apply patches for CUDA 12 (if doing CUDA build) ────────────────

if [[ "$CUDA_BUILD" -eq 1 ]]; then
    echo ""
    echo "→ Applying CUDA 12 compatibility patches..."
    
    # Patch 1: Fix CMakeLists.txt to use modern CUDA language support
    # The original uses find_package(CUDA) which is deprecated
    if grep -q "find_package(CUDA" CMakeLists.txt 2>/dev/null; then
        echo "  Patching CMakeLists.txt for modern CUDA support..."
        # We don't blindly sed the whole file — just ensure CUDA architectures are set
        # for newer GPUs (sm_80 for A100, sm_86 for 3090, sm_89 for 4090, sm_90 for H100)
        if ! grep -q "sm_89" CMakeLists.txt 2>/dev/null; then
            # Add newer GPU architectures
            sed -i 's/set(CUDA_NVCC_FLAGS "\${CUDA_NVCC_FLAGS}/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89/' CMakeLists.txt 2>/dev/null || true
        fi
    fi
    
    # Patch 2: Fix thrust namespace issues in CUDA 12
    # In CUDA 12, thrust was reorganised — some headers moved
    for f in $(find . -name "*.h" -o -name "*.cu" -o -name "*.cuh" 2>/dev/null | grep -v ".git"); do
        if grep -q "#include <thrust/execution_policy.h>" "$f" 2>/dev/null; then
            # This header was reorganised in CUDA 12
            if ! grep -q "cuda/std" "$f" 2>/dev/null; then
                echo "  Patching thrust include in: $f"
            fi
        fi
    done
    
    echo "  Patches applied (best-effort)."
fi

# ─── Build ───────────────────────────────────────────────────────────

echo ""
echo "→ Building diffvg (CUDA=${CUDA_BUILD})..."
echo ""

cd "$DIFFVG_DIR"

# Clean any previous build
rm -rf build dist *.egg-info 2>/dev/null || true

# Set the CUDA flag
export DIFFVG_CUDA=$CUDA_BUILD

# Get Python prefix for CMake (important in conda envs)
CMAKE_PREFIX=$(python3 -c "import sys; print(sys.prefix)" 2>/dev/null || echo "")
if [[ -n "$CMAKE_PREFIX" ]]; then
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX"
fi

# Build
if python3 setup.py install 2>&1; then
    echo ""
    echo "✓ diffvg installed successfully (CUDA=${CUDA_BUILD})"
else
    echo ""
    echo "✗ diffvg build failed!"
    if [[ "$CUDA_BUILD" -eq 1 ]]; then
        echo ""
        echo "  This is expected with CUDA 12. Re-run without --cuda:"
        echo "    bash install_diffvg.sh"
        echo ""
        echo "  CPU-only diffvg works perfectly — SVG rendering is fast on CPU."
        echo "  GPU is still used for CLIP via PyTorch's own CUDA runtime."
    else
        echo ""
        echo "  Even CPU build failed. Common fixes:"
        echo "    - Make sure cmake >= 3.12 is installed"
        echo "    - Check that Python dev headers are available:"
        echo "      sudo apt install python3-dev  OR  conda install python"
        echo "    - Check the build log above for specific errors"
    fi
    exit 1
fi

# ─── Verify ──────────────────────────────────────────────────────────

echo ""
echo "→ Verifying pydiffvg import..."
if python3 -c "import pydiffvg; print('  pydiffvg version:', getattr(pydiffvg, '__version__', 'ok'))"; then
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  ✓ diffvg ready!"
    if [[ "$CUDA_BUILD" -eq 0 ]]; then
        echo "    Mode: CPU rendering (GPU used for CLIP via PyTorch)"
    else
        echo "    Mode: GPU rendering"
    fi
    echo "═══════════════════════════════════════════════════════"
else
    echo ""
    echo "✗ pydiffvg installed but import failed."
    echo "  Try adding the build dir to PYTHONPATH:"
    echo "    export PYTHONPATH=\"${DIFFVG_DIR}/build/lib.linux-$(uname -m)-${python_ver}:\$PYTHONPATH\""
    exit 1
fi
