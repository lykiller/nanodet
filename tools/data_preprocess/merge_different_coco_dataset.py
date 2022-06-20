import json
import os

coco_classes_to_my_class = {

    'Pedestrian': 'person',
    'Car': 'car',
    'Truck': 'heavy-vehicle',
    'Tram': 'heavy-vehicle',
    'Cyclist': 'bicycle',
    'Tricycle': 'tricycle'
}

category_to_id = {
    'person': 0,
    'car': 1,
    'heavy-vehicle': 2,
    'bicycle': 3,
    'tricycle': 4
}

old_id_to_category = {
    1: 'Pedestrian',
    2: 'Cyclist',
    3: 'Car',
    4: 'Truck',
    5: 'Tram',
    6: 'Tricycle'
}

old_id_to_new_id ={
    1:0,
    2:3,
    3:1,
    4:2,
    5:2,
    6:4
}

path = '/home/user/work/dataset/df_bsd/annotations'
main_json = os.path.join(path, 'bsd_soda.json')
entry = os.path.join(path, 'coco_tiny.json')
dst_json = 'bsd_soda_coco.json'
main_json = open(main_json)
main = json.load(main_json)

old_main_image_len = len(main['images'])
old_main_len = len(main['annotations'])

file = open(entry)
file = json.load(file)
print(file['categories'])

for i in file['images']:
    i['id'] = i['id'] + old_main_image_len
    main['images'].append(i)

for i in file['annotations']:
    i['image_id'] = i['image_id'] + old_main_image_len
    i['id'] = i['id'] + old_main_len
    main['annotations'].append(i)

'''
for i in main['annotations']:
    old_id = i['category_id']
    i['category_id'] = old_id_to_new_id[old_id]

main['categories'] = []
for name in category_to_id:
    _dict = {
        'name':name,
        'id':category_to_id[name],
        'supercategory':''
    }
'''

with open(os.path.join(path, dst_json), 'w') as outfile:
    json.dump(main, outfile)

