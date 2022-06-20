# -*- coding: utf-8 -*-
import os
import shutil

src_data_dir = "D:/dataset/data/"
dst_dir = "D:/dataset/dongfeng_bsd/"
dst_train_img_dir = os.path.join(dst_dir, 'train', 'img')
dst_train_xml_dir = os.path.join(dst_dir, 'train', 'xml')
dst_val_img_dir = os.path.join(dst_dir, 'val', 'img')
dst_val_xml_dir = os.path.join(dst_dir, 'val', 'xml')


for i, image_name in enumerate(os.listdir(src_data_dir + '/img/')):
    ext = os.path.splitext(image_name)
    xml_name = ext[0] + '.xml'
    image_name_abs = os.path.join(src_data_dir, 'img', image_name)
    xml_name_abs = os.path.join(src_data_dir, 'xml', xml_name)
    if i % 5 == 0:
        output_img_path = os.path.join(dst_val_img_dir, image_name)
        output_xml_path = os.path.join(dst_val_xml_dir, xml_name)
    else:
        output_img_path = os.path.join(dst_train_img_dir, image_name)
        output_xml_path = os.path.join(dst_train_xml_dir, xml_name)
    if os.path.exists(xml_name_abs):
        shutil.copy(image_name_abs, output_img_path)
        shutil.copy(xml_name_abs, output_xml_path)
