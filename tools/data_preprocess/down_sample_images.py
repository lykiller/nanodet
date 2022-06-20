import os
import shutil
image_dir = '/home/user/work/dataset/df_bsd/imgs'
xml_dir = '/home/user/work/dataset/df_bsd/xmls'

dst_xml_dir = '/home/user/work/dataset/df_bsd/tiny_xml'

for i, xml_name in enumerate(os.listdir(xml_dir)):
    src_path = os.path.join(xml_dir, xml_name)
    dst_path = os.path.join(dst_xml_dir, xml_name)
    if i % 5 == 0:
        shutil.copy(src_path, dst_path)
