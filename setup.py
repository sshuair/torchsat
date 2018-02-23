#coding: --utf-8--
from setuptools import setup, find_packages
import os

# Check if OpenCV is installed and raise an error if it is not
# but don't do this if the ReadTheDocs systems tries to install
# the library, as that is configured to mock cv2 anyways
READ_THE_DOCS = os.environ.get('READTHEDOCS') == 'True'
if not READ_THE_DOCS:
    try:
        import cv2 # pylint: disable=locally-disabled, unused-import, line-too-long
    except ImportError as e:
        raise Exception("Could not find package 'cv2' (OpenCV). It cannot be automatically installed, so you will have to manually install it.")

# version
with open('torchvision_x/__init__.py') as f:
    for line in f:
        if line.find('__version__') >= 0:
            version = line.split('=')[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            break

# install requirements
inst_reqs = [
    'numpy','scipy','Pillow','torch','scikit-image'
]

# readme
with open('README.md') as f:
    readme = f.read()

setup(
    name='torchvision-enhance',
    version = version,
    description = u'Enhance torchvision for multi-channel images, 16-bit image, segmentation...',
    long_description = readme,
    keywords=['pytorch', 'vision', 'augmentation', 'deep learning'],
    author='sshuair',
    author_email='sshuair@gmail.com',
    url='https://github.com/sshuair/torchvision-enhance',
    packages = find_packages(exclude=('test')),
    install_requires=inst_reqs,

)
