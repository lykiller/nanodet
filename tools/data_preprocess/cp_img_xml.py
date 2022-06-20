import os
from shutil import copy
dst_img_dir = 'E:\\dataset\\df_bsd\\imgs'

dst_xml_dir = 'E:\\dataset\\df_bsd\\xmls'

src_dir = 'E:\\BaiduNetdiskDownload'


for root, dirs, files in os.walk(src_dir, topdown=False):
    for file_name in files:
        file_name_abs_path = os.path.join(root, file_name)
        if file_name.endswith(".jpg" or ".png"):
            dst_abs_path = os.path.join(dst_img_dir, file_name)
            copy(file_name_abs_path, dst_abs_path)
        if file_name.endswith(".xml"):
            dst_abs_path = os.path.join(dst_xml_dir, file_name)
            copy(file_name_abs_path, dst_abs_path)

