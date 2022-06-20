import os, base64
import json
import requests
from PIL import Image
from io import BytesIO

# dataset_dir = 'D:\\dataset\\coco'
dataset_dir = os.path.dirname(os.path.abspath(__file__))
train_image_dir = os.path.join(dataset_dir, 'train2017')
val_image_dir = os.path.join(dataset_dir, 'val2017')

train_json = os.path.join(dataset_dir, 'annotations', 'instances_train2017.json')
val_json = os.path.join(dataset_dir, 'annotations', 'instances_val2017.json')


def complete_coco_images(image_dir, json_file_name):

    main = json.load(open(json_file_name))
    stats_nums = 0
    for image in main['images']:
        image_filename = os.path.join(image_dir, image['file_name'])
        if not os.path.exists(image_filename):
            print(image_filename)
            stats_nums = stats_nums +1
            print(stats_nums)
            response = requests.get(image['coco_url'])
            ls_f = base64.b64encode(BytesIO(response.content).read()).decode('utf-8')
            imgdata = base64.b64decode(ls_f)
            file = open(image_filename, 'wb')
            file.write(imgdata)
            file.close()


complete_coco_images(train_image_dir, train_json)
complete_coco_images(val_image_dir, val_json)
