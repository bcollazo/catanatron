import setuptools
import os

readme_path = os.path.abspath(os.path.join(__file__, "..", "README.md"))

with open(readme_path, "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="catanatron_gym",
<<<<<<< HEAD
    version="3.0.1",
=======
    version="2.0.0",
>>>>>>> master
    author="Bryan Collazo",
    author_email="bcollazo2010@gmail.com",
    description="Open AI Gym to play 1v1 Catan against a random bot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bcollazo/catanatron",
<<<<<<< HEAD
    packages=setuptools.find_packages(),
=======
    packages=setuptools.find_packages(exclude=["catanatron_server"]),
>>>>>>> master
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["catanatron", "gym"],
)
