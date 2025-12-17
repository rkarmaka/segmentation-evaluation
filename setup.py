from setuptools import setup, find_packages
import os

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="segmentation-evaluation",
    version="0.1.0",
    author="Ranit Karmakar",
    description="Segmentation Evaluation Metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rkarmaka/segmentation-evaluation",
    license="MIT",
    packages=find_packages(include=["metrics*"]),
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-image",
        "scipy",
        "rich"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
)
