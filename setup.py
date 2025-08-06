"""
Setup script for PrismRAG
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="prismrag",
    version="0.1.0",
    author="PrismRAG Team",
    author_email="prismrag@example.com",
    description="PrismRAG: Improving RAG Factuality through Distractor Resilience and Strategic Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/prismrag",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/prismrag/issues",
        "Documentation": "https://github.com/yourusername/prismrag/blob/main/docs/",
        "Source Code": "https://github.com/yourusername/prismrag",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "vllm": [
            "vllm>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prismrag-demo=demo:main",
            "prismrag-generate-data=experiments.generate_training_data:main",
            "prismrag-train=experiments.train_prismrag:main",
            "prismrag-evaluate=experiments.evaluate_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    keywords=[
        "artificial intelligence",
        "natural language processing",
        "retrieval augmented generation",
        "RAG",
        "factuality",
        "chain of thought",
        "machine learning",
        "deep learning",
    ],
    zip_safe=False,
)