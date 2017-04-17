try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    "description": "CLIQS: Crosslinguistic Investigations in Quantitative Syntax",
    "author": "Richard Futrell",
    "url": "http://web.mit.edu/futrell/www",
    "download_url": "",
    "install_requires": "nose networkx requests pyrsistent".split(),
    "packages": "cliqs ".split(),
    "scripts": "".split(),
    "name": "cliqs",
}

setup(**config)
