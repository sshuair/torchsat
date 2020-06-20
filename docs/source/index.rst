.. TorchSat documentation master file, created by
   sphinx-quickstart on Sat Sep 14 10:56:23 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TorchSat's documentation!
====================================

TorchSat is an open-source deep learning framework for satellite imagery analysis based on PyTorch_.

  This project is still **work in progress**. If you want to know more about it, please refer to the Roadmap_ .

**Hightlight**

- Support multi-channels(> 3 channels, e.g. 8 channels) images and TIFF file as input.
- Convenient data augmentation method for classification, sementic segmentation and object detection.
- Lots of models for satellite vision tasks, such as ResNet, DenseNet, UNet, PSPNet, SSD, FasterRCNN ...
- Lots of common satellite datasets loader.
- Training script for common satellite vision tasks.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.md
   core-conception.md
   tutorials/image-classification.md
   tutorials/semantic-segmentation.md
   tutorials/object-detection.md
   tutorials/change-detection.md
   tutorials/data-augumentation.md
   tools.md
   api/api.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _Roadmap: https://github.com/sshuair/torchsat/wiki/Roadmap
.. _PyTorch: https://pytorch.org/
