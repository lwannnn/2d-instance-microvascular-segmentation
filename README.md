# 2d-instance-microvascular-segmentation
The goal of this repository is to segment instances of microvascular structures, including capillaries, arterioles, and venules. It'll create a model trained on 2D PAS-stained histology images from healthy human kidney tissue slides.

dataset acquired:  
https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data   
You need download the dataset and put it in 'archive' folder  

folder structure:  
--archive  
&ensp;--hubmap-hacking-the-human-vasculature  
&ensp;&ensp;--train  
&ensp;&ensp;--gt (This folder is created by code to store visual tags)  
&ensp;&ensp;--test  
&ensp;&ensp;polygons.jsonl  
&ensp;&ensp;ample_submission.csv  
&ensp;&ensp;tile_meta.csv  
&ensp;&ensp;wsi_meta.csv  
reference.py  
  
…to be added  
    