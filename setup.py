#!/usr/bin/env python

from setuptools import setup, find_packages
import os.path

setup(
    name="tap-outbrain",
    version="0.2.0",
    description="Singer.io tap for extracting data from the Outbrain Amplify API",
    author="Your Name",
    url="https://github.com/your-username/tap-outbrain",
    classifiers=["Programming Language :: Python :: 3.10"],
    install_requires=[
        "singer-python>=5.12.1",
        "requests>=2.28.0",
        "backoff>=2.1.2",
        "python-dateutil>=2.8.2"
    ],
    python_requires=">=3.8",
)