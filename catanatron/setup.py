import setuptools  # type: ignore
import os

readme_path = os.path.abspath(os.path.join(__file__, "..", "..", "README.md"))

with open(readme_path, "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="catanatron",
    version="3.2.1",
    author="Bryan Collazo",
    author_email="bcollazo2010@gmail.com",
    description="Fast Settlers of Catan Python Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bcollazo/catanatron",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["networkx"],
)
