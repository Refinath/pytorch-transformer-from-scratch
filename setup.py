from setuptools import setup, find_packages

setup(
    name="pytorch-transformer-from-scratch",
    version="0.1.0",
    description="Transformer model implemented from scratch using PyTorch",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "tqdm>=4.66",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)