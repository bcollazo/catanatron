import setuptools
import os

readme_path = os.path.abspath(os.path.join(__file__, "..", "..", "README.md"))

with open(readme_path, "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="catanatron",
<<<<<<< HEAD:catanatron_core/setup.py
    version="3.1.2",
=======
    version="3.0.0",
>>>>>>> master:setup.py
    author="Bryan Collazo",
    author_email="bcollazo2010@gmail.com",
    description="Fast Settlers of Catan Python Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bcollazo/catanatron",
<<<<<<< HEAD:catanatron_core/setup.py
    packages=setuptools.find_packages(),
=======
    packages=setuptools.find_packages(exclude=["catanatron_server"]),
>>>>>>> master:setup.py
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["networkx"],
)
