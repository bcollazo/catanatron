import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="catanatron",
    version="1.0.1",
    author="Bryan Collazo",
    author_email="bcollazo2010@gmail.com",
    description="Fast Settlers of Catan Python Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bcollazo/catanatron",
    packages=setuptools.find_packages(
        exclude=[
            "catanatron_server",
        ]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx",
    ],
)
