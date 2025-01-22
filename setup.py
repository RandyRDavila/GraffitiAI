from setuptools import setup, find_packages

setup(
    name="graffitiai",
    version="0.1.0",
    author="Randy Davila",
    author_email="randyrdavila@gmail.com",
    description="A Python package for automated mathematical conjecturing.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RandyRDavila/GraffitiAI",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    package_data={"graffitiai": ["data/*.csv"]},
    install_requires=[
        "pandas",
        "numpy",
        "reportlab"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
