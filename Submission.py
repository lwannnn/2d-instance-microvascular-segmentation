"""
This code is only used for Upload Notebook: A simple test version
"""
### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from PIL import Image
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import torch.nn as nn
from collections import OrderedDict
from scipy.ndimage import label
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        # nn.ConvTranspose2d用来进行转置卷积的，它主要做了这几件事：首先，对输入的feature
        #map进行padding操作，得到新的feature
        #map；然后，随机初始化一定尺寸的卷积核；最后，用随机初始化的一定尺寸的卷积核在新的feature
        #map上进行卷积操作。
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)#将两个张量(tensor)拼接在一起
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(#有序字典
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""
    # check input mask --
    if mask.dtype != bool:
        raise ValueError(
           "encode_binary_mask expects a binary mask, received dtype == %s" %
           mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
           "encode_binary_mask expects a 2d mask, received shape == %s" %
           mask.shape)
    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str
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


def submission_test():
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    model_name = './Weight/test202306052310.pt'#pt name
    model = UNet()
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    outputdict = {}
    with torch.no_grad():
        for dirname, _, filenames in os.walk('./archive/hubmap-hacking-the-human-vasculature/test'):#/kaggle/input
            for filename in filenames:
                print(os.path.join(dirname, filename))
                image_name=os.path.join(dirname, filename)
                image = Image.open(image_name)
                image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)#因为没有batch了，加一维
                input = image.to(device)
                y_pred = model(input)
                outputs = y_pred.contiguous().squeeze(0)  # 深拷贝,shape:(1,512,512)
                ref_img_id = filename.split(".")[0]
                encoded = encode_output(outputs, ref_img_id)
                for key in encoded:  # "key" is image name(except ".tif")
                    outputdict[key] = " ".join([f"0 1.0 {x.decode('utf-8')}" for x in encoded[key]])  # 因为一张图可能会有好多mask

        submission(outputdict)

submission_test()

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session