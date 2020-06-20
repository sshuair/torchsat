# Change Detection
This is tutorial for satellite change detection.

## Prepare the training data
The satellite change detection dataset should be organized by the following struct.

```
.
├── train
│   ├── pre
│   │   ├── train_1.png
│   │   ├── train_2.png
│   │   ├── ...
│   ├── post
│   │   ├── train_1.png
│   │   ├── train_2.png
│   │   ├── ...
│   └── label
│       ├── train_1.png
│       ├── train_2.png
│       ├── ...
└── val
    ├── pre
    │   ├── val_10.png
    │   ├── val_11.png
    │   ├── ...
    ├── post
    │   ├── val_10.png
    │   ├── val_11.png
    │   ├── ...
    └── label
        ├── val_10.png
        ├── val_11.png
        ├── ...
```
You can use the `make_mask_cls` generate the pre and post image.

## train a model
We provide the following models:
- 