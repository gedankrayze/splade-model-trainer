from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="splade-model-trainer",
    version="0.1.0",
    author="Gedank Rayze",
    author_email="info@gedankrayze.com",
    description="A toolkit for training, evaluating, and deploying SPLADE models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gedankrayze/splade-model-trainer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
