"""
Hockey McByte Tracker - Setup
=============================
Production-ready occlusion-robust tracking for hockey.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hockey_mcbyte_tracker",
    version="2.0.0",
    author="Hockey Tracking System",
    description="Occlusion-robust multi-object tracking for hockey with UKF, SAM, and CUTIE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hockey-tracker/hockey-mcbyte-tracker",
    packages=find_packages(),
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
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "ultralytics>=8.0.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "full": [
            "lap>=0.4.0",
            "filterpy>=1.4.5",
            "matplotlib>=3.4.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hockey-track=hockey_mcbyte_tracker.run:main",
        ],
    },
)
