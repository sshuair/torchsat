# vision-multi

PyTorch transforms and DataLoader for classification and semantic segmentation. It not only support three channels image, but also support multi-channel(>=4) images, especially 16-bit multi-channel **tiff file**



in progress...

## TODO:
1. transform for multi-channel image
    - [ ] resize
    - [ ] center_crop
    - [x] random_crop
    - [x] flip
    - [ ] horizontal_flip
    - [ ] vertical_flip
    - [x] rotate
    - [x] shift
    - [x] normaize
    - [x] noise
    - [ ] pad
2. dataloader for multi-channel image, such as jpg, tiff, png....
    - [x] single-label classification
    - [x] multi-label classification
    - [ ] object localization
    - [x] semantic segmentaion
4. use example
    - [x] classification
    - [ ] object localization
    - [ ] semantic segmentaion