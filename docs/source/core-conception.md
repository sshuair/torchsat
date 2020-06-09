# Core Conceptions

Because TorchSat is based on PyTorch, 
you'd better have some deep learning and PyTorch knowledge to use and modify this project.

Here are some core conceptions you should know.

1. All input image data, Whether it is PNG, JPEG or GeoTIFF, will be converted to [NumPy](https://numpy.org/) ndarray, 
and the ndarray dimension is `[height, width]` (single channel image) or `[height, width, channels]` (multi-channel image).

2. After the data is read into [NumPy](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index) ndarray, there are only three data types, `np.uint8`, `np.uint16`, `np.float`. 
These three data types are chosen because, in satellite imagery, 
    - The most common image stored in JPEG or PNG (such as the Google Map Satellite Image) is mostly `8-bit (np.uint8)` data;
    - And the `16-bit (np.uint16)` is mostly the original data type of remote sensing satellite imagery, which is also very common; 
    - The third type `float (np.float)` is considered, because sometimes, we will use remote sensing index (such as NDVI_) as features to participate in training. For this data we suppose all input data values range from 0 to 1.
