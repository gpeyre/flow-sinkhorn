"""
Setup script for Flow Sinkhorn package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flowsinkhorn",
    version="0.1.0",
    author="Gabriel Peyré",
    author_email="gabriel.peyre@ens.fr",
    description="Flow-based Sinkhorn algorithm for W1 optimal transport on graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gpeyre/flow-sinkhorn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
    ],
    extras_require={
        "sparse": ["sparse>=0.13.0"],
        "exact": ["cvxpy>=1.1.0"],
        "torch": ["torch>=1.9.0"],
        "gpu": ["torch>=1.9.0"],  # Alias for torch
        "examples": [
            "matplotlib>=3.1.0",
            "networkx>=2.4",
            "scikit-learn>=0.22.0",
            "jupyter>=1.0.0",
            "pandas>=1.0.0",
        ],
        "all": [
            "sparse>=0.13.0",
            "cvxpy>=1.1.0",
            "torch>=1.9.0",
            "matplotlib>=3.1.0",
            "networkx>=2.4",
            "scikit-learn>=0.22.0",
            "jupyter>=1.0.0",
            "pandas>=1.0.0",
        ],
    },
    keywords="optimal transport, wasserstein distance, sinkhorn, graphs, gpu, pytorch",
)
