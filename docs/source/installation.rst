Installtation
=============
Deepndencies
------------

TorchSat is based on `PyTorch`_, so you should install the PyTorch
first. And if you want to use the GPU version, you should install CUDA.

Python package dependencies(seee also requirements.txt): pytorch,
torchvision, numpy, pillow, tifffile, six, scipy, opencv

   Note: TorchSat only suport `Python 3`_. We recommend version after
   Python 3.5(including python 3.5), but wo have not tested any version
   below Python 3.5

Install
-------

Install from PyPI or Anconda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  PyPI: ``pip3 install torchsat``
-  Anconda: `` ``

Install from source
~~~~~~~~~~~~~~~~~~~

-  Install the latest version

.. code:: bash

    git clone https://github.com/sshuair/torchsat.git
    cd torchsat
    python3 setup.py install

-  Install the stable version

   1. Visit the `release`_ page and download the version you want.
   2. Decompress the zip or tar file.
   3. Enter the torchsat directory and run this command
      ``python3 setup.py install``.

For data preparation
--------------------

[wip]


Docker
------
You can pull the docker image from Docker Hub if you want use TorchSat in docker.

1. pull image 
    - cpu: `docker pull sshuair/torchsat:cpu-latest` 
    - gpu: `docker pull sshuair/torchsat:gpu-latest` 

2. run container 
    - cpu: `docker run -ti --name <NAME> docker pull sshuair/torchsat:cpu-latest bash`
    - gpu: `docker run -ti --gpu 0,1 --name <NAME> docker pull sshuair/torchsat:gpu-latest bash`


This way you can easily use the TorchSat in docker container.



.. _PyTorch: https://pytorch.org/
.. _Python 3: https://www.python.org/
.. _release: https://github.com/sshuair/torchsat/releases