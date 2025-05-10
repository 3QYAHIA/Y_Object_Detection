# Dataset Instructions

This directory contains scripts and loaders for the Pascal VOC dataset.

## Pascal VOC Dataset

The Pascal VOC dataset will be automatically downloaded when you run training with the `--download` flag:

```
python train.py --download --epochs 10 --backbone resnet50
```

Or you can manually download it by running:

```
python -c "from data.voc_dataset import download_voc_dataset; download_voc_dataset('./data', '2012')"
```

## Dataset Structure

After downloading, the dataset will be structured as follows:

```
data/
├── VOCdevkit/
│   └── VOC2012/
│       ├── Annotations/
│       ├── ImageSets/
│       ├── JPEGImages/
│       └── ...
└── voc_dataset.py
```

## Note

The dataset files are excluded from version control due to their large size (approximately 2GB). When cloning this repository, you'll need to download the dataset using one of the methods above. 