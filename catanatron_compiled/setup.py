from glob import glob
import setuptools
import os
from pybind11.setup_helpers import (
    Pybind11Extension,
    build_ext,
    ParallelCompile,
    naive_recompile,
)

ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile).install()

ext_modules = [
    Pybind11Extension(
        "catanatron_compiled",
        sorted(glob("catanatron_compiled/src/*.cc")),
    ),
]

setuptools.setup(
    name="catanatron_compiled",
    version="1.0.0",
    author="Anthony Erb Lugo",
    author_email="aerblugo@gmail.com",
    description="Fast compiled implementation of a subset of Catanatron's features",
    url="https://github.com/bcollazo/catanatron",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    setup_requires=["pybind11"],
)
