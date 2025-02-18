#TODO This version only can use for batch size==1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from modelSelector import train_model
from Dataset.hubMapDataset import HubMapDataset as Dataset
import torch
import Transforms as transforms
from Loss.diceloss import *
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import label
from oid_mask_encoding import encode_binary_mask
image_dir = './archive/hubmap-hacking-the-human-vasculature/gt/'

def test_to_image(pred_mask,id):#for batch_size=1
    filename = id.split("/")[-1]
    id = filename.split(".")[0]
    image_file = os.path.join(image_dir+id+".tif")
    image = Image.open(image_file)
    pred_mask = pred_mask.contiguous().detach().cpu().numpy().squeeze()#要做一个深拷贝
    # print(np.unique(pred_mask,return_counts=True))
    mask_image = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(mask_image)

    for i in range(pred_mask.shape[0]):
        for j in range(pred_mask.shape[1]):
            if pred_mask[i, j] >0.6:
                draw.point((j,i), fill=(255, 255, 255,128))#TODO:Confirm it is (j,i)(like canvas?)
    masked_image = Image.alpha_composite(image.convert('RGBA'), mask_image)
    output_path = f"./Results/{id}.tif"
    masked_image.save(output_path)

#This part partly conference: https://www.kaggle.com/code/averma111/pytorch-hubmap-cnn
def submission(outputdict):
    submission = pd.DataFrame(outputdict.items(), columns=["id", "prediction_string"])
    submission["height"] = 512
    submission["width"] = 512
    submission = submission[["id", "height", "width", "prediction_string"]]
    submission.to_csv("submission.csv", index=False)
    print("Submission Completed!!!")


def encode_output(outputs, idx):
    id = idx#image name
    outputs = (outputs>0.5) * 1#二值化
    array = outputs.cpu().numpy()
    labeled_array, num_features = label(array)# 使用scipy的label函数获取连通域
    result = torch.zeros(num_features, 512, 512)
    for i in range(1, num_features + 1):
        result[i - 1] = torch.tensor(labeled_array == i)
    blood_vessel = result ==1
    all_encode = {}
    list_encode = []
    blood_vessel = blood_vessel.cpu().numpy()
    for i in range(blood_vessel.shape[0]): #i = vessel_num
        sliceImage = blood_vessel[i, :, :]
        binarized = sliceImage > 0
        coded_len = encode_binary_mask(binarized)
        list_encode.append(coded_len)
    all_encode[id] = list_encode
    return all_encode

def prediction(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model_name = ''.join(list(filter(str.isalpha,args.model)))
    model_to_load = "./Weight/"+args.model+'.pt'
    model = train_model(model_name)
    checkpoint = torch.load(model_to_load)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    outputdict = {}
    criterion = DiceLoss()
    test_set = Dataset(split='Testing', dataset_path=args.images, transform=None)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)
    loss_valid=[]
    for batch_index, (input, mask,img_id) in enumerate(tqdm(test_loader)):
        ref_img_id = str(img_id).split("/")[-1].split(".")[0]
        input = input.to(device)
        mask = mask.to(device)
        y_pred = model(input)
        outputs = y_pred.contiguous().squeeze(0)#深拷贝,shape:(1,512,512)
        encoded = encode_output(outputs,ref_img_id)
        for key in encoded:# "key" is image name(except ".tif")
            outputdict[key] = " ".join([f"0 1.0 {x.decode('utf-8')}" for x in encoded[key]])# 因为一张图可能会有好多mask

        ACC = 1 - criterion(y_pred, mask)  # DICE
        loss_valid.append(ACC.item())

        test_to_image(y_pred,ref_img_id)

    print("Final Best validation mean ACC: {:4f}".format(np.mean(loss_valid)))
    submission(outputdict)

if __name__ == "__main__":
    #set arguments
    parser = argparse.ArgumentParser(description="Validating model for Segmentation instances of microvascular structures")
    parser.add_argument("--model",type=str,default="test202306052310",help="modelname")
    parser.add_argument("--images", type=str, default="./archive/hubmap-hacking-the-human-vasculature", help="root folder with images")
    parser.add_argument('--cuda', default=True, type=bool,help='Use GPU calculating')
    args = parser.parse_args()
    prediction(args)