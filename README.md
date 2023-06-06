# 2d-instance-microvascular-segmentation
The goal of this repository is to segment instances of microvascular structures, including capillaries, arterioles, and venules. It'll create a model trained on 2D PAS-stained histology images from healthy human kidney tissue slides.

dataset acquired:  
https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data   
You need download the dataset and put it in 'archive' folder  

###Folder structure  
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
oid_mask_encoding.py  
--Weight(This folder is created manually^_^ to store training results and logs)
  
…to be added    
    
My environment:  
python-version: 3.10  
cuda-version:11.6 
pytorch-version:1.13.1  


###TODO List
1.Test[√]  
2.Visible[√]  
3.convert mask to expected format[√]   
4.It seems still some problem on training(no-grad?)[√]  
5.It seems just a semantic segmentation?[√]   
6.Unexpected EOFError??  
7.Calculate Confidence(How???)