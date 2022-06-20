
import numpy as np
import os
import glob
import os.path as osp
import shutil
import xml.etree.ElementTree as ET
import cv2
import random
import math
import json

# 'person', 'car', 'trunk', 'bicycle', 'bus', 'train', 'motorcycle'
# 'person', 'motor_vehicle', 'non_motor_vehicle'
coco_classes_to_my_class = {
    'person': 'person',
    'car': 'car',
    'truck': 'heavy-vehicle',
    'bus': 'heavy-vehicle',
    'bicycle': 'bicycle',
    'motorbike': 'bicycle',
    'tricycle': 'tricycle',
    'train': 'heavy-vehicle',
    'motorcycle': 'bicycle',

}


def make_dir(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)


def get(root, name):
    return root.findall(name)


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_caregories(xml_list):
    category_list = []
    for xmlPath in xml_list:
        tree = ET.parse(xmlPath)
        root = tree.getroot()
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in set(category_list):
                category_list.append(category)
    return category_list


def convert(img_dir, xml_list, label2id):
    # finish cateKey first

    cate_value = [{"id": label2id[x], "name": x, "supercategory": ''} for x in label2id]
    json_dict = {"images": [], "annotations": [], "categories": cate_value}
    print(len(xml_list))
    bbox_id = 0

    for image_id, xml_f in enumerate(xml_list):
        tree = ET.parse(xml_f)
        root = tree.getroot()

        img_name = os.path.basename(xml_f)[:-4] + ".jpg"
        print(image_id, img_name)
        src_img = cv2.imread(osp.join(img_dir, img_name))
        if src_img is not None:
            # shutil.copy(osp.join(img_dir, img_name), goal_img_dir)
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)

            if width == 0 or height == 0:
                img = cv2.imread(osp.join(img_dir, img_name))
                width, height, _ = img.shape

            image = {'file_name': img_name, 'height': height, 'width': width, 'id': image_id}
            json_dict['images'].append(image)


            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                if category in coco_classes_to_my_class.keys():
                # if True:
                    # print(category)
                    category = coco_classes_to_my_class[category]
                    # print(category)
                    category_id = label2id[category]

                    bndbox = get_and_check(obj, 'bndbox', 1)
                    xmin = math.floor(float(get_and_check(bndbox, 'xmin', 1).text))
                    ymin = math.floor(float(get_and_check(bndbox, 'ymin', 1).text))
                    xmax = math.floor(float(get_and_check(bndbox, 'xmax', 1).text))
                    ymax = math.floor(float(get_and_check(bndbox, 'ymax', 1).text))

                    # cv2.rectangle(src_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(src_img, category, (xmin, ymin - 1), font, 0.5, (0, 255, 0), thickness=1)
                    if xmax < xmin or ymax < ymin:
                        print(img_name)
                    assert xmax >= xmin
                    assert ymax >= ymin
                    o_width = abs(xmax - xmin)
                    o_height = abs(ymax - ymin)
                    ann = {'area': o_width * o_height,
                           'iscrowd': 0,
                           'image_id': image_id,
                           'bbox': [xmin, ymin, o_width, o_height],
                           'category_id': category_id,
                           'id': bbox_id,
                           'segmentation': []}
                    json_dict['annotations'].append(ann)
                    bbox_id += 1
                else:
                    continue
            print(bbox_id)
        else:
            continue
    return json_dict


def xml2json(img_dir, xml_dir, target_dir):

    make_dir(target_dir)
    img_dir_n = osp.join(target_dir, 'imgs')
    # annos_dir = osp.join(target_dir, 'annos')
    # make_dir(img_dir_n)
    # make_dir(annos_dir)

    xml_list = glob.glob(osp.join(xml_dir, '*.xml'))
    label2id = {}
    # cate_list = get_caregories(xml_list)
    category_list = ['person', 'car', 'heavy-vehicle', 'bicycle', 'tricycle']
    for i, cate in enumerate(category_list):
        label2id[cate] = i

    random.shuffle(xml_list)

    train_anno = convert(img_dir, xml_list, label2id)

    with open(osp.join(target_dir, 'coco_tiny_val.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(train_anno))


if __name__ == '__main__':
    train_img_dir = "/home/user/work/dataset/coco/images/val2017"
    train_xml_dir = "/home/user/work/dataset/coco/xmls_tiny/val2017"
    target_dir = "/home/user/work/dataset/df_bsd/annotations"

    xml2json(train_img_dir, train_xml_dir, target_dir)




