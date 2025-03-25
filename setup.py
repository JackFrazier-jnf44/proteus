from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="proteus",
    version="0.1.0",
    author="Jack Frazier",
    author_school_email="jnf44@cornell.edu",
    author_personal_email="jack.frazier03@gmail.com",
    description="A comprehensive framework for protein structure prediction and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JackFrazier-jnf44/proteus",
    package_dir={"": "src", "helpful": "helpful"},
    packages=find_packages(where="src") + find_packages(where="helpful"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "gpu": read_requirements("requirements-gpu.txt"),
        "dev": read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "proteus=src.main:main",
        ],
    },
)