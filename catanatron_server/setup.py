import setuptools

setuptools.setup(
    name="catanatron_server",
    version="1.0.0",
    author="Bryan Collazo",
    author_email="bcollazo2010@gmail.com",
    description="Server to watch catanatron games",
    url="https://github.com/bcollazo/catanatron",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "catanatron",
        "flask",
        "flask_cors",
        "flask_sqlalchemy",
        "sqlalchemy",
        "psycopg2-binary",
    ],
)
