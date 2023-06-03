# 2d-instance-microvascular-segmentation
The goal of this repository is to segment instances of microvascular structures, including capillaries, arterioles, and venules. It'll create a model trained on 2D PAS-stained histology images from healthy human kidney tissue slides.

dataset acquired:
https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data
You need download the dataset and put it in 'archive' folder

folder structure:
--archive
  --hubmap-hacking-the-human-vasculature
    --train
    --gt (This folder is created by code to store visual tags)
    --test
    polygons.jsonl
    sample_submission.csv
    tile_meta.csv
    wsi_meta.csv
reference.py

…to be added
    