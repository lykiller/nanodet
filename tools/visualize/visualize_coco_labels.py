import cv2
import os
import numpy as np
from pycocotools.coco import COCO

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

img_path = '/home/user/work/dataset/df_bsd/imgs'
annFile = '/home/user/work/dataset/df_bsd/annotations/bsd_soda_coco.json'
save_path = '/home/user/work/dataset/df_bsd/visualize_soda_df_bsd'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# id_to_category = [ 'person', 'car', 'heavy-vehicle', 'bicycle', 'tricycle']


def draw_rectangle(coordinates, image, image_name):
    for coordinate in coordinates:
        left, top, right, bottom, label = map(int, coordinate)
        color = colors[label % len(colors)]
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(image, str(label), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    cv2.imshow(image_name, image)
    cv2.waitKey(0)

    # cv2.imwrite(save_path + '/' + image_name, image)

import random
coco = COCO(annFile)

# catIds = coco.getCatIds(catNms=['Crack','Manhole', 'Net', 'Pothole','Patch-Crack', "Patch-Net", "Patch-Pothole", "other"])
# catIds = coco.getCatIds()
# imgIds = coco.getImgIds(catIds=catIds)
imgIds = coco.getImgIds()
random.shuffle(imgIds)

for imgId in imgIds:

    img = coco.loadImgs(imgId)[0]
    image_name = img['file_name']
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
    anns = coco.loadAnns(annIds)

    # coco.showAnns(anns)

    coordinates = []
    img_raw = cv2.imread(os.path.join(img_path, image_name))
    for j in range(len(anns)):
        coordinate = anns[j]['bbox']
        coordinate[2] += coordinate[0]
        coordinate[3] += coordinate[1]
        coordinate.append(anns[j]['category_id'])
        coordinates.append(coordinate)

    draw_rectangle(coordinates, img_raw, image_name)
