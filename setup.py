#coding: --utf-8--
from setuptools import setup
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
    description = u'PyTorch data transforms and dataloader for classification and segmentation.',
    long_description = readme,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
        'Topic :: Scientific/Engineering :: GIS'],
    keywords='pytorch vision augmentation deep learning',
    author='sshuair',
    author_email='sshuair@gmail.com',
    url='https://github.com/sshuair/torchvison-enhance',
    packages = ['torchvision_x'],
    install_requires=inst_reqs,

)
