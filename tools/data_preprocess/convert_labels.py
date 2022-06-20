import os
import xml.etree.ElementTree as ET
from nanodet.data.dataset.xml_dataset import get_file_list

src_class_name = ['traffic_cone', 'truck', 'car', 'front_rear', 'bus',
                          'person', 'bicycle', 'wheel', 'traffic_sign', 'humanlike',
                          'tunnel', 'vehicle_like', 'head', 'motorbike', 'up_body',
                          'closed_barrier_gate', 'pavement_signs', 'open_barrier_gate','zebra_crossing']

dst_class_name = ['human_like', 'vehicle_like', 'motor_like']

map_src_to_dst = {
    'truck': 'vehicle_like',
    'car': 'vehicle_like',
    'front_rear': 'vehicle_like',
    'bus': 'vehicle_like',
    'wheel': 'vehicle_like',
    'vehicle_like': 'vehicle_like',

    'head': 'human_like',
    'person': 'human_like',
    'human_like': 'human_like',
    'up_body': 'human_like',

    'bicycle': 'motor_like',
    'motorbike': 'motor_like'
}


def generate_new_xml(ann_path, new_ann_path):
    ann_file_names = get_file_list(ann_path)
    for xml_name in ann_file_names:
        tree = ET.parse(os.path.join(ann_path, xml_name))
        root = tree.getroot()
        for _object in root.findall("object"):
            category = _object.find("name").text
            if category in map_src_to_dst.keys():
                category = map_src_to_dst[category]
                _object.find("name").text = category
            else:
                root.remove(_object)
        tree.write(os.path.join(new_ann_path, xml_name))


if __name__ == "__main__":
    root_path = 'D:\\dataset\\dongfeng_bsd\\'
    ann_path = root_path + "train\\xml"
    new_ann_path = root_path + "train\\xml2"
    generate_new_xml(ann_path, new_ann_path)
