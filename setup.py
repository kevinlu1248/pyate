from setuptools import setup, find_packages, find_namespace_packages

with open("README.md", "r", encoding="utf8") as f:
    long_description = f.read()

setup(
    name="pyate",
    # packages=["pyate"],
    version="0.3.8",  # Start with a small number and increase it with every change you make
    license="MIT",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="PYthon Automated Term Extraction",  # Give a short description about your library
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kevin Lu",
    author_email="kevinlu1248@gmail.com",
    url="https://github.com/kevinlu1248/pyate",  # Provide either the link to your github or to your website
    download_url="https://github.com/kevinlu1248/pyate/archive/0.3.tar.gz",  # I explain this later on
    keywords=[
        "nlp",
        "python3",
        "spacy",
        "term_extraction",
    ],  # Keywords that define your package best
    # include_package_data=True,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        '': ['*.csv']
    },
    install_requires=[
        "pandas==1.0.3",
        "numpy==1.18.4",
        "spacy==2.2.4",
        "pyahocorasick==1.4.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3",
    ],
)
