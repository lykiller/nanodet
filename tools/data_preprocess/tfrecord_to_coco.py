import numpy as np
import os, random
import sys
import tensorflow as tf
import cv2
import time

# from distutils.version import StrictVersion
from matplotlib import pyplot as plt
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import json

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "10"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)



# cls_remap = {'car':'car', 'bus':'bus', 'truck':'truck', 'person':'person', 'bicycle':'bicycle', 'motorbike':'bicycle',
#     'wheel':'wheel', 'front_rear':'front_rear', 'traffic_sign':'traffic_sign','traffic_lights':'traffic_lights',
#     'tunnel':'tunnel','pavement_signs':'pavement_signs','open_barrier_gate':'barrier_gate', 'closed_barrier_gate':'barrier_gate',
#     'head':'head','up_body':'up_body','zebra_crossing':'zebra_crossing','humanlike':'humanlike','humanlik':'humanlike','traffic_cone':'traffic_cone',
#     'traffic_signd':'traffic_sign','traffic-sign':'traffic_sign','foont_rear':'front_rear','traff_lights':'traffic_lights','traffic cone':'traffic_cone'}

# tf_queue = [
#             r'/home/share/data/jm202009/jm_val_remap_shuffle0.tfrecord',
#             # r'/home/share/data/jm202008-humanlike2/jm_val_remap0.tfrecord',
#             # r'/home/share/data/jm-tunnel/jm_pedestrian_val1.tfrecord',
#             # r'/home/share/data/jm-tunnel/jm_humanlike_val1.tfrecord',
#             ]
# tf_queue = [r'/home/liyuan/data/jm2021_adas_data/jm_train_remap_shuffle%d.tfrecord' % i for i in range(21)]
# PATH_TO_LABELS_NEW = r'/home/liyuan/data/jm2021_adas_data/all_labels.pbtxt'
# Output_dir = r'/home/liyuan/data/jm2021_adas_coco/images'
# Output_file='/home/liyuan/data/jm2021_adas_coco/annotations'
MaxEvalNum = 0 #2000

tf_queue = [r'D:\\dataset\\jm2021_adas_data\\jm_train_remap_shuffle%d.tfrecord' % i for i in range(21)]
PATH_TO_LABELS_NEW = r'D:\\dataset\\jm2021_adas_data\\jm2021_adas_data/all_labels.pbtxt'
Output_dir = r'D:\\dataset\\jm2021_adas_data\\jm2021_adas_coco/images'
Output_file = 'D:\\dataset\\jm2021_adas_data\\jm2021_adas_coco/annotations'

colors_list = [( 0, 0, 0 ), ( 0, 0, 255 ), ( 0, 255, 0 ),
 ( 255, 0, 0 ), ( 0, 255, 255 ), ( 255, 0, 255 ), ( 255, 255, 0 ),
 ( 128, 0, 0 ), ( 0, 128, 0 ), ( 0, 0, 128 )] + [(255, 128, 128),]*20

from object_detection.utils import label_map_util
category_index_new = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS_NEW, use_display_name=True)
new_label_map_dict = label_map_util.get_label_map_dict(PATH_TO_LABELS_NEW)

GLOBAL_ANN_ID = 0

def get_ann_id():
  """Return unique annotation id across images."""
  global GLOBAL_ANN_ID
  GLOBAL_ANN_ID += 1
  return GLOBAL_ANN_ID

if not os.path.exists(Output_dir):
    os.makedirs(Output_dir)

detection_graph = tf.Graph()


def read_and_decode(filename_queue, batch_size=32):
    """
    read and decode
    """

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    def _decode_image(content, channels):
        return tf.cond(
            tf.image.is_jpeg(content),
            lambda: tf.image.decode_jpeg(content, channels),
            lambda: tf.image.decode_png(content, channels))

    features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/class/text': tf.VarLenFeature(dtype=tf.string),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
    }

    parsed_features = tf.parse_single_example(serialized_example, features)

    image = _decode_image(parsed_features['image/encoded'], channels=3)
    width = tf.cast(parsed_features['image/width'], tf.int32)
    height = tf.cast(parsed_features['image/height'], tf.int32)
    cls = parsed_features['image/object/class/label'].values
    cls_name = parsed_features['image/object/class/text'].values
    xmin = parsed_features['image/object/bbox/xmin'].values
    xmax = parsed_features['image/object/bbox/xmax'].values
    ymin = parsed_features['image/object/bbox/ymin'].values
    ymax = parsed_features['image/object/bbox/ymax'].values
    filename = parsed_features['image/filename']
    source_id = parsed_features['image/source_id']
    # image_shape = tf.stack([height, width, 3])
    # image = tf.reshape(image, image_shape)
    # image.set_shape([height, width, 3])

    # images = tf.train.shuffle_batch([image], batch_size=batch_size, num_threads=1,
    #                                                capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
    return image, filename, source_id, width, height, cls, cls_name, xmin, xmax, ymin, ymax

unknown_name_set = set()

def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


with detection_graph.as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        tf.global_variables_initializer().run(session=sess)
        tf.local_variables_initializer().run(session=sess)

        filename_queue = tf.train.string_input_producer(tf_queue, shuffle=False)  # , capacity=1024, num_epochs=1)
        images, filename, source_id, width, height, cls, cls_name, xmin, xmax, ymin, ymax = read_and_decode(
            filename_queue)
        sample_num = []

        for tf_record in tf_queue:
            c = 0
            for record in tf.python_io.tf_record_iterator(tf_record):
                c += 1
            sample_num.append(c)
        print('total pic number: %d' % sum(sample_num))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        ann_json_dict = {
            'images': [],
            'type': 'instances',
            'annotations': [],
            'categories': []
        }
        for class_name, class_id in new_label_map_dict.items():
            cls_ = {'supercategory': 'none', 'id': class_id, 'name': class_name}
            ann_json_dict['categories'].append(cls_)

        try:
            idx = 0
            img_set = set()
            cls_list = {}
            total_num = min(sum(sample_num), MaxEvalNum) if MaxEvalNum > 0 else sum(sample_num)
            import time

            time_start = time.time()
            while not coord.should_stop() and idx < total_num:
                # Run training steps or whatever
                if idx % 500 == 0:
                    time_end = time.time()
                    print('[%d / %d] time: %0.2fs' % (idx, total_num, time_end - time_start))
                    time_start = time.time()
                imgs, imgname, imgid, w_, h_, c_, c_str, x1, x2, y1, y2 = sess.run(
                    [images, filename, source_id, width, height, cls, cls_name, xmin, xmax, ymin, ymax])
                if imgname in img_set:
                    print(str(imgname, 'utf-8') + 'already exist!')
                    idx = idx + 1
                    continue
                img_set.add(imgname)

                for c_one in c_str:
                    cls_list[c_one] = 0 if c_one not in cls_list else cls_list[c_one] + 1

                x1 = np.maximum(x1, 0)
                y1 = np.maximum(y1, 0)
                x2 = np.minimum(x2, 1)
                y2 = np.minimum(y2, 1)

                file_path = os.path.join(Output_dir, str(imgname, 'utf-8'))
                # image_pil = Image.fromarray(np.uint8(imgs))
                # image_pil.save(file_path)

                imgh, imgw = imgs.shape[0:2]

                image_cpy = imgs.copy()
                # for y1, x1, y2, x2, s, c in zip(y1, x1, y2, x2, c_str, c_):
                #     # y1, x1, y2, x2, s, c = int(y1*imgh), int(x1*imgw), int(y2*imgh), int(x2*imgw), s, int(c)
                #     # cv2.rectangle(image_cpy, (x1, y1), (x2, y2), colors_list[c], 2)
                #     draw_bounding_box_on_image_array(image_cpy, y1, x1, y2, x2, colors_list[int(c)], 2, [str(s, 'utf-8')], True)

                # plt.imshow(image_cpy)
                cv2.imwrite(file_path, image_cpy[..., ::-1])

                # if b'tricycle' in c_str:
                #     cv2.imwrite(file_path, image_cpy[..., ::-1])
                #     # pass#print(file_path)
                # elif b'vehicle_like' in c_str:
                #     pass#print(file_path)

                if ann_json_dict:
                    image = {
                        'file_name': imgname.decode(),
                        'height': int(imgh),
                        'width': int(imgw),
                        'id': idx,
                    }
                    ann_json_dict['images'].append(image)

                    for idx_, (x1_, y1_, x2_, y2_, cls_name_) in enumerate(
                            zip(x1, y1, x2, y2, c_str)):
                        abs_xmin = int(x1_ * imgw + 0.5)
                        abs_ymin = int(y1_ * imgh + 0.5)
                        abs_xmax = int(x2_ * imgw + 0.5)
                        abs_ymax = int(y2_ * imgh + 0.5)
                        abs_width = abs_xmax - abs_xmin
                        abs_height = abs_ymax - abs_ymin
                        name_decode = cls_name_.decode()
                        if name_decode in new_label_map_dict.keys():
                            ann = {
                                'area': abs_width * abs_height,
                                'iscrowd': 0,
                                'image_id': idx,
                                'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
                                'category_id': new_label_map_dict[name_decode],
                                'id': get_ann_id(),
                                'ignore': 0,
                                'segmentation': [],
                            }
                            ann_json_dict['annotations'].append(ann)

                idx = idx + 1
                # plt.show()

        except tf.errors.OutOfRangeError:
            print
            'Done training -- epoch limit reached'
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # coord.request_stop()
        coord.join(threads)
        print(tf_queue)
        print('total pic number: %d' % len(img_set))
        print(cls_list)

        json_file_path = os.path.join(
            os.path.dirname(Output_file),
            'json_' + os.path.basename(Output_file) + '.json')
        print(json_file_path)
        with tf.io.gfile.GFile(json_file_path, 'w') as f:
            json.dump(ann_json_dict, f)