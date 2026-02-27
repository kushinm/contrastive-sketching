from setuptools import setup, find_packages

setup(
    name="contrastive-clipasso",
    version="0.1.0",
    description="CLIP-guided sketch generation with contrastive mode",
    packages=find_packages(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "contrastive-clipasso=contrastive_clipasso.sketch:main",
        ],
    },
)
