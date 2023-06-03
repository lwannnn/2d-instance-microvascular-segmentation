"""
This code is used to visualize the mask in json format onto the dataset
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw

def mask_to_image(dataset_path):
    image_dir = dataset_path+'/train/'
    jsonl_file = dataset_path+'/polygons.jsonl'
    gt_path = os.path.join(dataset_path, 'gt')#create a folder named 'gt' to store visual tags

    if not os.path.exists(gt_path):
        try:
            os.mkdir(gt_path)
        except OSError as e:
            print(f"Error': {e}")

    type_colors = {
            'blood_vessel': (0,255,0),    # Green
            'glomerulus': (0,0,255),      # Blue
            'unsure': (255,0,0)          # Red
        }

    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            image_id = data['id']
            annotations = data['annotations']# there are some annotations in it.e.g., there are some vessles,glomerulus in one image.
            image_file = os.path.join(image_dir, image_id + '.tif')
            image = Image.open(image_file)

            mask_image = Image.new('RGB', image.size)
            draw = ImageDraw.Draw(mask_image)

            for annotation in annotations:
                annotation_type = annotation['type']
                coordinates = annotation['coordinates']
                polygons = []
                for polygon_coords in coordinates:
                    for [x, y] in polygon_coords:
                        polygons.append((x, y))
                draw.polygon(polygons, fill=type_colors[annotation_type])
                """
                ——xy        – 由 [(x1, y1), (x2, y2),…]  等2元组或 [x1，y1，x1，y1，…] 等数值组成的序列
                ——fill      –用于填充的颜色 
                 ——outline   –轮廓使用的颜色
                """
            result_image = Image.blend(image, mask_image, alpha=0.5)
            result_image.save(gt_path+'/'+image_id+'.tif')

if __name__=='__main__':
    dataset_path = './archive/hubmap-hacking-the-human-vasculature'
    mask_to_image(dataset_path)