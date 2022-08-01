from setuptools import find_packages
from setuptools import setup

with open("README.md", "r", encoding="utf8") as f:
    long_description: str = f.read()

setup(
    name="pyate",
    version=
    "0.5.5",  # Start with a small number and increase it with every change you make
    license=
    "MIT",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    # Give a short description about your library
    description="PYthon Automated Term Extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kevin Lu",
    author_email="kevinlu1248@gmail.com",
    url="https://github.com/kevinlu1248/pyate",
    download_url="https://github.com/kevinlu1248/pyate/archive/0.4.tar.gz",
    keywords=[
        "nlp",
        "python3",
        "spacy",
        "term_extraction",
    ],  # Keywords that define your package best
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"": ["resources/*.zip"]},
    install_requires=[
        "pandas>=1.0.3",
        "numpy>=1.18.4",
        "spacy>=2.2.4",
        "pyahocorasick>=1.4.0",
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
