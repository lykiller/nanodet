import os
import xml.etree.ElementTree as ET

_dir = 'E:\\dataset\\df_bsd\\xmls'

category_dict = {}

map

for xml in os.listdir(_dir):

    tree = ET.parse(os.path.join(_dir, xml))
    root = tree.getroot()
    for _object in root.findall("object"):
        category = _object.find("name").text
        if category_dict.__contains__(category):
            category_dict[category] = category_dict[category] + 1
        else:
            category_dict[category] = 1

print(category_dict)