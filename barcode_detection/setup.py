from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="barcode-detector",
    version="1.0.0",
    author="Production Barcode Detector",
    description="A production-level barcode detector with custom implementation and library fallbacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/barcode-detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.1",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "python-dotenv>=1.0.0",
        "pyzbar>=0.1.9",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "line-profiler>=4.0.0",
            "memory-profiler>=0.61.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "barcode-detector=barcode_detector.cli:main",
        ],
    },
)