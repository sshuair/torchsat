# Useful Tools
Collections of usefle tools for satellite deep learning processing.

## processing tools

### make_mask_cls
This tool can split the large satellite image(e.g.,10000 x 10000 pixel) to small chips(e.g., 512 x 512 pixel). Both the satellite image and label image can use this tool.
```
Usage: ts make_mask_cls [OPTIONS]

  this tool is to patch the large satellite image to small image.

Options:
  --filepath TEXT      the target satellite image to split. Note: the file
                       should have crs
  --width INTEGER      the width of the patched image, default 512
  --height INTEGER     the height of the patched image, default 512
  --drop_last BOOLEAN  set to True to drop the last column and row, if the
                       image size is not divisible by the height and width.
                       default True.
  --outpath TEXT       the output file path
  --colormap BOOLEAN   weather write the colormap to the output patched image.
                       only work for single channel image. default False.
  --extensions TEXT    the train image extension, only work for dirctory file
                       path.
  --help               Show this message and exit.
```

example: 

``` bash
ts make_mask_cls --filepath ./tests/classification \
                 --width 512 \
                 --height 512 \
                 --drop_last True \
                 --extensions tif \
                 --extensions tiff \
                 --outpath ./test/out
```

### make_mask_seg
This tool is to patch the large satellite image to small image and label for segmentation. The input should be projected satellite image and vector file. e.g., geojson shpfile.

```
Usage: ts make_mask_seg [OPTIONS]

  this tool is to patch the large satellite image to small image and label
  for segmentation.

Options:
  --image_file TEXT    the target satellite image to split. Note: the file
                       should have crs
  --label_file TEXT    the corresponding label file of the satellite image.
                       vector or raster file. Note the crs should be same as
                       satellite image.
  --field TEXT         field to burn
  --width INTEGER      the width of the patched image
  --height INTEGER     the height of the patched image
  --drop_last BOOLEAN  set to True to drop the last column and row, if the
                       image size is not divisible by the height and width.
  --outpath TEXT       the output file path
  --help               Show this message and exit.
```

### calcuate_mean_std
This tool is for calcuating each channel mean and std value of the datasets.
```
Usage: ts calcuate_mean_std [OPTIONS]

  calcuate the datasets mean and std value

Options:
  --root PATH         root dir of image datasets  [required]
  --percent FLOAT     percent of images to calcuate
  --channels INTEGER  datasets image channels
  --maxvalue FLOAT    max value of all images default: {255}
  --extension TEXT    file suffix to calcuate, default ('jpg', 'jpeg', 'png',
                      'tif', 'tiff')
  --help              Show this message and exit.
```

example:
``` bash
ts calcuate_mean_std --root /tests/classification/val/
100%|█████████████████████████████████████████████████████████████████| 163/163 [00:02<00:00, 70.53it/s]
scaled  mean:[0.36473823 0.40924644 0.41250621]
scaled  std: [0.09052812 0.07698209 0.0671676 ]
orginal mean: [ 93.00824798 104.35784201 105.18908467]
orginal std: [23.08467009 19.6304323  17.12773828]
```

## train scripts
The script can be used for training the model. You can modify the script according to your own project. It is independent of torchsat, you can get it from the `scripts` directory from [torchsat](https://github.com/sshuair/torchsat/tree/develop/scripts) repo.

### scripts/train_cls.py

```
$ python test.py --help
usage: test.py [-h] [--train-path TRAIN_PATH] [--val-path VAL_PATH]
               [--model MODEL] [--pretrained PRETRAINED] [--resume PATH]
               [--num-classes NUM_CLASSES] [--in-channels IN_CHANNELS]
               [--device DEVICE] [-b BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
               [--print-freq PRINT_FREQ] [--ckp-dir CKP_DIR]

TorchSat Classification Training

optional arguments:
  -h, --help            show this help message and exit
  --train-path TRAIN_PATH
                        train dataset path
  --val-path VAL_PATH   validate dataset path
  --model MODEL         the classification model
  --pretrained PRETRAINED
                        use the ImageNet pretrained model or not
  --resume PATH         path to latest checkpoint (default: none)
  --num-classes NUM_CLASSES
                        num of classes
  --in-channels IN_CHANNELS
                        input image channels
  --device DEVICE
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size
  --epochs EPOCHS       train epochs
  --lr LR               initial learning rate
  --print-freq PRINT_FREQ
                        print frequency
  --ckp-dir CKP_DIR     path to save checkpoint
```
