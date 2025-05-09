# COCO Dataset

This directory will contain the COCO dataset after running `download_coco.py`.

## Structure
- `train2017/`: Training images
- `val2017/`: Validation images
- `annotations/`: Annotation JSON files
- `subset/`: 10-class subset of COCO
- `tiny_subset/`: 5-class subset of COCO

## Download
To download the dataset:

```bash
python data/download_coco.py
```

Note: The dataset is large (>10GB for the full dataset). The script also provides options to download smaller subsets for easier experimentation. 