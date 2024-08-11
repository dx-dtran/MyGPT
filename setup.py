from setuptools import setup, find_packages

setup(
    name="MyGPT",
    version="1.0",
    author="Daniel Tran",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0"
    ]
)
