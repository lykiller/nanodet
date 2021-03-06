# -*- coding: utf-8 -*-
# coco_classes_extract.py
# Author: FineBit
import argparse
import os
import re
import shutil


def copy_txt_files(src_path, dst_path, c: list):
    for root, dirs, files in os.walk(src_path):
        total = len(files)
        print("Find {} txt files in {}".format(total, src_path))
        copy_num = 0
        for i, file in enumerate(files):
            classes_exist = False
            src_f_path = os.path.join(root, file)
            dst_f_path = os.path.join(dst_path, file)
            with open(src_f_path) as f:
                for line in f.readlines():
                    items = re.split(r"[ ]+", line)
                    if items[0] in c:
                        copy_num += 1
                        classes_exist = True
                        break
            if classes_exist:
                shutil.copy(src_f_path, dst_f_path)
            print("Check files: {}/{},  Copy txt files: {}".format(i + 1, total, copy_num), end='\r')


def delete_other_class(dst_l_path, c: list, renumber: bool):
    for root, dirs, files in os.walk(dst_l_path):
        total = len(files)
        print("\nFind {} txt files in {}".format(total, dst_l_path))
        for i, file in enumerate(files):
            f_path = os.path.join(dst_l_path, file)
            with open(f_path, 'r') as f:
                lines = f.readlines()
            with open(f_path, 'w') as f_w:
                for line in lines:
                    items = re.split(r"[ ]+", line)
                    if items[0] in c:
                        if renumber:
                            items[0] = str(c.index(items[0]))
                            line = " ".join(items)
                        f_w.write(line)
            print("Delete other class: {}/{}".format(i + 1, total), end='\r')


def copy_images(img_dir, dst_img_dir, dst_l_path):
    for root, dirs, files in os.walk(dst_l_path):
        total = len(files)
        print("\nFind {} txt files in {}".format(total, dst_l_path))
        for i, file in enumerate(files):
            image_file = os.path.splitext(file)[0] + ".jpg"  # txt???????????????jpg
            src = os.path.join(img_dir, image_file)
            dst = os.path.join(dst_img_dir, image_file)
            shutil.copy(src, dst)
            print("Copy images: {}/{}".format(i+1, total), end='\r')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("source dir, destination dir, classes list")
    parser.add_argument('-s', "--src", dest="src_root", default="./coco_yolo/",
                        help="source root dir of coco_yolo labels and images")
    parser.add_argument('-d', "--dst", dest="dst_root", default="./coco_yolo_extract/",
                        help="destination root dir")
    parser.add_argument('-c', "--classes", dest="dst_classes", default=['0'], nargs='+',
                        help="input the classes you wanna extract, for example: --classes 0 1 2")
    parser.add_argument('-r', "--renumber", action="store_true", help="renumber the class number")
    arg = parser.parse_args()
    print('111')
    for item in ["train", "val"]:
        labels_path = os.path.join(arg.src_root, "labels/"+item+"2017")  # ???????????????
        print(labels_path)
        images_dir = os.path.join(arg.src_root, "images/"+item+"2017")  # "images/train2017"  # ???????????????
        dest_labels_path = os.path.join(arg.dst_root, "labels/"+item+"2017")  # ??????????????????
        if not os.path.exists(dest_labels_path):
            os.makedirs(dest_labels_path)
        dest_images_dir = os.path.join(arg.dst_root, "images/"+item+"2017")  # ??????????????????
        if not os.path.exists(dest_images_dir):
            os.makedirs(dest_images_dir)
        dest_classes = arg.dst_classes  # ????????????????????????
        print(dest_classes)

        # ???????????????????????????class???txt???????????????????????????
        copy_txt_files(labels_path, dest_labels_path, dest_classes)

        # ??????????????????????????????txt??????????????????class
        delete_other_class(dest_labels_path, dest_classes, arg.renumber)

        # ???????????????????????????class???images?????????????????????
        copy_images(images_dir, dest_images_dir, dest_labels_path)