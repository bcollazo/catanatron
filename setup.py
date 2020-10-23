import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="catanatron",  # Replace with your own username
    version="0.0.1",
    author="Bryan Collazo",
    author_email="bcollazo2010@gmail.com",
    description="Settlers of Catan simluation environment in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bcollazo/catanatron",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
