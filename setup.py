from setuptools import setup, find_packages
import re

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open("torchsat/__init__.py", encoding="utf-8") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

requirements = [x.strip() for x in open("requirements.txt").readlines()]

setup(
    name="torchsat",
    version=version,
    author="sshuair",
    author_email="sshuair@gmail.com",
    url="https://github.com/sshuair/torchsat",
    description="TorchSat is an open-source PyTorch framework for satellite imagery analysis.",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="MIT",
    install_requires=requirements,
    keywords='pytorch deep learning satellite remote sensing',
)
