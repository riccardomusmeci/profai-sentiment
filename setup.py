from setuptools import setup, find_packages

setup(
    name="profai",
    version="0.1.0",
    description="A library for building and deploying NLP AI models with a focus on professional applications.",
    author="Riccardo Musmeci",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "pyyaml>=6.0",
        "datasets",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.10",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)