import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from modelSelector import train_model
from Dataset.hubMapDataset import HubMapDataset as Dataset
import torch
import Transforms as transforms
from Loss.diceloss import *
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from PIL import Image, ImageDraw
image_dir = './archive/hubmap-hacking-the-human-vasculature/gt/'

def test_to_image(pred_mask,id):
    filename = id.split("/")[-1]
    id = filename.split(".")[0]
    image_file = os.path.join(image_dir+id+".tif")
    image = Image.open(image_file)
    pred_mask = pred_mask.detach().cpu().numpy().squeeze()
    mask_image = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(mask_image)
    for i in range(pred_mask.shape[0]):
        for j in range(pred_mask.shape[1]):
            if pred_mask[i, j] >0.5:
                draw.point((j, i), fill=(255, 255, 255))#TODO:Confirm it is (j,i)(like canvas?)
    masked_image = Image.alpha_composite(image.convert('RGBA'), mask_image)
    output_path = f"./Results/{id}.tif"
    masked_image.save(output_path)


def Prediction(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model_name = ''.join(list(filter(str.isalpha,args.model)))
    model_to_load = "./Weight/"+args.model+'.pt'
    model = train_model(model_name)
    checkpoint = torch.load(model_to_load)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    criterion = DiceLoss()
    test_set = Dataset(split='Testing', dataset_path=args.images, transform=None)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)
    loss_valid=[]
    for batch_index, (input, mask) in enumerate(tqdm(test_loader)):
        input = input.to(device)
        mask = mask.to(device)
        y_pred = model(input)
        ACC = 1 - criterion(y_pred, mask)  # DICE
        loss_valid.append(ACC.item())
        ref_img_id = str(test_loader.dataset.test_data[batch_index])
        test_to_image(y_pred,ref_img_id)
    print("Final Best validation mean ACC: {:4f}".format(np.mean(loss_valid)))

if __name__ == "__main__":
    #set arguments
    parser = argparse.ArgumentParser(description="Validating model for Segmentation instances of microvascular structures")
    parser.add_argument("--model",type=str,default="test202306042116",help="modelname")
    parser.add_argument("--images", type=str, default="./archive/hubmap-hacking-the-human-vasculature", help="root folder with images")
    parser.add_argument('--cuda', default=True, type=bool,help='Use GPU calculating')
    args = parser.parse_args()
    Prediction(args)