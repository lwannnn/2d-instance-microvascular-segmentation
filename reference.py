"""
This code is used to visualize the mask in json format onto the dataset
"""
import os
import json
from PIL import Image, ImageDraw
import numpy as np

def mask_to_image(dataset_path):
    image_dir = dataset_path+'/train/'
    jsonl_file = dataset_path+'/polygons.jsonl'
    gt_path = os.path.join(dataset_path, 'task')#create a folder named 'gt' to store visual tags

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
            mask_1 = Image.new('1',image.size)
            mask_line =Image.new('1',image.size)
            draw = ImageDraw.Draw(mask_image)
            draw_1 = ImageDraw.Draw(mask_1)
            draw_line = ImageDraw.Draw(mask_line)

            for annotation in annotations:
                annotation_type = annotation['type']
                coordinates = annotation['coordinates']
                polygons = []
                for polygon_coords in coordinates:
                    for [x,y] in polygon_coords:
                        polygons.append((x, y))
                draw.polygon(polygons, fill=type_colors[annotation_type])

                if annotation_type == 'blood_vessel':
                    polygons = []
                    coordinates = annotation['coordinates']
                    for polygon_coords in coordinates:
                        for [x, y] in polygon_coords:
                            polygons.append((x, y))
                    draw_1.polygon(polygons, fill=1)

                if annotation_type == 'blood_vessel':
                    polygons = []
                    coordinates = annotation['coordinates']
                    for polygon_coords in coordinates:
                        for i in polygon_coords:
                            draw_line.point((i[0],i[1]),fill=1)#看着好像不需要旋转的？
                """
                ——xy        – 由 [(x1, y1), (x2, y2),…]  等2元组或 [x1，y1，x1，y1，…] 等数值组成的序列
                ——fill      –用于填充的颜色 
                 ——outline   –轮廓使用的颜色
                """
            result_image = Image.blend(image, mask_image, alpha=0.5)
            result_image.save(gt_path+'/'+image_id+'.tif')
            mask_1.save(gt_path+'/'+image_id+'_mask.tif')
            mask_line.save(gt_path+'/'+image_id+'_line.tif')

if __name__=='__main__':
    dataset_path = './archive/hubmap-hacking-the-human-vasculature'
    mask_to_image(dataset_path)