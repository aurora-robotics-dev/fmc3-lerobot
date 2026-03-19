from setuptools import setup, find_packages

setup(
    name="fourier_gr2_robot",
    version="0.1.0",
    description="A high-level Python SDK wrapper for Fourier GR-2 Humanoid Robot",
    author="Fourier User",
    packages=find_packages(),
    install_requires=[
        "fourier-aurora-client",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
