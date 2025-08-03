"""
VIVOX: Living Voice and Ethical Conscience System
Setup configuration for pip installation
"""

from setuptools import setup, find_packages
import os

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read dev requirements
dev_requirements = []
if os.path.exists("requirements-dev.txt"):
    with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
        dev_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vivox-ai-private-research",
    version="1.0.0-research-preview",
    author="LUKHAS AI Team",
    author_email="research@lukhas-ai.com",
    description="[PRIVATE RESEARCH] Living Voice and Ethical Conscience System for AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lukhas-ai/vivox",
    project_urls={
        "Bug Tracker": "https://github.com/lukhas-ai/vivox/issues",
        "Documentation": "https://vivox.readthedocs.io",
        "Source Code": "https://github.com/lukhas-ai/vivox",
        "Research Paper": "https://arxiv.org/abs/vivox2024",
    },
    packages=find_packages(include=["vivox", "vivox.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Private :: Do Not Upload",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.3.0"],
        "google": ["google-generativeai>=0.3.0"],
        "all": ["openai>=1.0.0", "anthropic>=0.3.0", "google-generativeai>=0.3.0"],
    },
    entry_points={
        "console_scripts": [
            "vivox=vivox.cli:main",
            "vivox-benchmark=vivox.tools.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vivox": [
            "moral_alignment/precedent_seeds.py",
            "consciousness/state_variety_enhancement.py",
            "moral_alignment/decision_strictness_enhancement.py",
        ],
    },
    keywords=[
        "ai", "ethics", "consciousness", "moral-alignment",
        "artificial-intelligence", "ethical-ai", "ai-safety",
        "consciousness-simulation", "memory-systems"
    ],
)