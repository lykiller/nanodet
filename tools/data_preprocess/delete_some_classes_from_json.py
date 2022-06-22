import json
import os

json_dir = 'D:/dataset/jm2021_adas_coco/annotations'

train_json_name = 'instances_train.json'
val_json_name = 'instances_val.json'

class_names_list = ['car', 'bus', 'truck', 'person', 'bicycle',
                    'wheel', 'front_rear', 'traffic_sign', 'humanlike', 'tunnel',
                    'zebra_crossing', 'vehicle_like', 'tricycle', 'vehicle_side']


def generate_new_json(src_json, dst_json, json_dir=json_dir, class_names_list=class_names_list):
    src_json = os.path.join(json_dir, src_json)
    dst_json = os.path.join(json_dir, dst_json)

    label2id = {}

    for i, name in enumerate(class_names_list):
        label2id[name] = i

    cate_value = [{"id": label2id[x], "name": x, "supercategory": ''} for x in label2id]

    main_json = open(src_json)
    main_json = json.load(main_json)
    old_categories = main_json["categories"]
    print(old_categories)
    old_id2label = {}
    for cate in old_categories:
        old_id2label[cate['id']] = cate['name']
    print(old_id2label)
    json_dict = {"images": main_json['images'], "annotations": [], "categories": cate_value}
    i = 0
    for annos in main_json['annotations']:
        old_cate_id = annos['category_id']
        cate_name = old_id2label[old_cate_id]

        if label2id.__contains__(cate_name):
            annos['category_id'] = label2id[cate_name]
            annos['id'] = i
            json_dict['annotations'].append(annos)
            i = i + 1
            print(annos)

    with open(dst_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_dict))


generate_new_json(train_json_name, 'train14.json')
generate_new_json(val_json_name, 'val14.json')

