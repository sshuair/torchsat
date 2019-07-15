from setuptools import setup, find_packages

readme = open('README.md').read()

requirements = [x.strip() for x in open('requirements.txt').readlines()]

setup(
    name='torchsat',
    version="0.1",
    author='sshuair',
    author_email='sshuair@gmail.com',
    url='https://github.com/sshuair/torchsat',
    description='TorchSat is an open-source PyTorch framework for satellite imagery analysis.',
    long_description=readme,
    packages=find_packages(),
    license='MIT',

    install_requirements=requirements,
)