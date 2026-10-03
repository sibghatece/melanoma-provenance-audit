# Data layout

No images are distributed with this repository. Download the datasets from
their providers and arrange them in one folder (your `MELANOMA_ROOT`) as
below. If you keep this repository's `data/` folder as `MELANOMA_ROOT`, put
the files here.

```
MELANOMA_ROOT/
  HAM10000_images_part_1/                 HAM10000 JPEGs, part 1
  HAM10000_images_part_2/                 HAM10000 JPEGs, part 2
  HAM10000_metadata.tab                   (or HAM10000_metadata.csv)
  ISIC_Image_Dataset/                     ISIC 2020 training JPEGs (33,126 files)
  ISIC_2020_Training_GroundTruth_v2.csv
  ISIC_2020_Training_Duplicates.csv
  melanoma-skin-cancer/                   Fanconi mirror (or set KAGGLE_ROOT)
  Javed_melanoma_cancer_dataset/          Javid mirror
  skin_cancer9_calssesisic/               Skin Cancer ISIC mirror
  skin-cancer-isic-2019-2020-malignant-or-benign/   ISIC 2019 and 2020 mirror
  skin-cancer-mnist10000-ham-augmented-dataset/     Balanced HAM10000 mirror
```

The mirror folder names are the ones used in `code/mirror_audit.py`
(`MIRRORS`); change them there if you store the mirrors elsewhere. Unzip each
mirror as downloaded; its own train/test folders are detected automatically.

## Sources

| Dataset | Source |
|---|---|
| HAM10000 (images and metadata) | Harvard Dataverse, https://doi.org/10.7910/DVN/DBW86T |
| ISIC 2020 training images, ground truth v2, duplicates list | ISIC Challenge, https://challenge.isic-archive.com/data/#2020 |
| Fanconi mirror (3,297 images) | https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign |
| Javid mirror (10,605 images) | https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images |
| Skin Cancer ISIC mirror (2,357 images) | https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic |
| ISIC 2019 and 2020 mirror (11,400 images) | https://www.kaggle.com/datasets/sallyibrahim/skin-cancer-isic-2019-2020-malignant-or-benign |
| Balanced HAM10000 mirror (39,507 images) | https://www.kaggle.com/datasets/utkarshps/skin-cancer-mnist10000-ham-augmented-dataset |

HAM10000 and ISIC 2020 are released under CC BY-NC 4.0. The Kaggle mirrors
are subject to the terms set by their uploaders. Mirrors can change after
they are downloaded; the image counts above are those of our copies.
