# Installation
## Deepndencies

TorchSat is based on [PyTorch](https://pytorch.org/), so you should install the PyTorch
first. And if you want to use the GPU version, you should install CUDA.

Python package dependencies(seee also requirements.txt): pytorch,
torchvision, numpy, pillow, tifffile, six, scipy, opencv

>Note: TorchSat only suport [Python 3](https://www.python.org/). We recommend version after
   Python 3.5(including python 3.5), but wo have not tested any version
   below Python 3.5

## Install
-------

### Install from PyPI

-  PyPI: ``pip3 install torchsat``

### Install from source

-  Install the latest version

```

    git clone https://github.com/sshuair/torchsat.git
    cd torchsat
    python3 setup.py install
```

-  Install the stable version

   1. Visit the [release](https://github.com/sshuair/torchsat/releases) page and download the version you want.
   2. Decompress the zip or tar file.
   3. Enter the torchsat directory and run this command
      ``python3 setup.py install``.


### Docker
You can pull the docker image from [Docker Hub](https://hub.docker.com/r/sshuair/torchsat) if you want use TorchSat in docker.

1. pull image 
    - cpu: `docker pull sshuair/torchsat:cpu-latest` 
    - gpu: `docker pull sshuair/torchsat:gpu-latest` 

2. run container 
    - cpu: `docker run -ti --name <NAME> sshuair/torchsat:cpu-latest bash`
    - gpu: `docker run -ti --gpu 0,1 --name <NAME> sshuair/torchsat:gpu-latest bash`


This way you can easily use the TorchSat in docker container.

