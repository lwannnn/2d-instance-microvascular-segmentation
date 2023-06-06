"""
This code is used to upload dataset.
Current version only add masked images!!!(about 1633 images)
"""
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import numpy as np
from PIL import Image,ImageDraw
import torch
import json

class HubMapDataset(Dataset):
    def __init__(self,dataset_path='../archive/train',split="Testing",transform=None):
        image_dir = dataset_path + '/train/'
        jsonl_file = dataset_path + '/polygons.jsonl'
        self.split = split
        self.transform = transform
        image_container,mask_container=[],[]
        with open(jsonl_file, 'r') as f:#另外一种思路是仅仅保留index 在get item里面读取,但考虑到数据量可能并不是很大，先这样
            for line in f:
                data = json.loads(line)
                image_id = data['id']
                image_file = os.path.join(image_dir, image_id + '.tif')
                image_container.append(str(image_file)) #type:'string'
                annotations = data['annotations']
                mask_container.append(annotations) #type:'list'

        train_data, test_data, train_mask, test_mask = train_test_split(image_container, mask_container, test_size=0.005,random_state=1,shuffle=True)

        if self.split == 'Training':
            self.train_data = train_data
            self.train_mask = train_mask
        elif self.split == 'Testing':
            self.test_data = test_data
            self.test_mask = test_mask

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Testing':
            return len(self.test_data)

    def __getitem__(self, index):
        img = ""
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask_image = Image.new('1', (512, 512))
        draw = ImageDraw.Draw(mask_image)
        annotations = None
        if self.split == 'Training':
            img = self.train_data[index]
            annotations = self.train_mask[index]
        elif self.split == "Testing":
            img = self.test_data[index]
            annotations = self.test_mask[index]

        image = Image.open(img)
        #This part is partially referenced in the following links: https://www.kaggle.com/code/thomasrochefort/hubmap-simple-pytorch-dataloader
        for annotation in annotations:
            annotation_type = annotation['type']
            if annotation_type == 'blood_vessel':
                polygons=[]
                coordinates = annotation['coordinates']
                for polygon_coords in coordinates:
                    for [x, y] in polygon_coords:
                        polygons.append((x, y))
                draw.polygon(polygons,fill=1)

        mask = np.array(mask_image)
        # print("Tests:"+str(np.unique(mask,return_counts=True)))
                # for polygon_coords in coordinates:
                    # rr, cc = np.array([i[1] for i in polygon_coords]), np.asarray([i[0] for i in polygon_coords])
                    # mask[rr, cc] = 1

        # Convert PIL Image and mask to PyTorch tensor
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)  # Shape: [C, H, W]
        mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0) #add channel

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        if self.split == 'Training':
            return image,mask
        elif self.split == "Testing":
            return image,mask,img


if __name__=='__main__':
    traindata = HubMapDataset('../archive/hubmap-hacking-the-human-vasculature')
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=1, shuffle=True, num_workers=1)
    for batch_idx, (inputs, targets,img_id) in enumerate(trainloader):
        print(targets.unique(return_counts=True))
        print(str(img_id).split("/")[-1].split(".")[0])
