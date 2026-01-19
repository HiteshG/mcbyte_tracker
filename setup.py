#!/usr/bin/env python
"""
Hockey McByte Tracker - Setup Script
=====================================
Production-ready multi-object tracking for hockey analysis.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="hockey-mcbyte-tracker",
    version="1.0.0",
    author="Hockey Analytics Team",
    description="Production-ready McByte multi-object tracking optimized for hockey",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/hockey-mcbyte-tracker",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.0,<2.0.0",
        "opencv-python>=4.6.0",
        "scipy>=1.9.0",
        "lap>=0.4.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "ultralytics>=8.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
        "omegaconf>=2.2.0",
        "hydra-core>=1.2.0",
    ],
    extras_require={
        "masks": [
            # SAM and Cutie for mask propagation
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hockey-track=core.pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="object tracking, hockey, sports analytics, computer vision, deep learning",
)
