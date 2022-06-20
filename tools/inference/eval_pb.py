import numpy as np
import os
import sys
import tensorflow as tf
import cv2

# from distutils.version import StrictVersion
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

# if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
#   raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from object_detection.utils import label_map_util
from object_detection.utils import object_detection_evaluation
from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

# from utils import visualization_utils as vis_util
preds_nanme = []
quantize=True
ralative=True
theta = 66.8 / 180 * np.pi
center_scale = True
MaxEvalNum = 0#1000

# w = 576
# h = 320
# class_num = 15
# layer_num = 3
# ModelPath = r'/home/wangyr/code/automl/efficientdet/tmp/frozen_model_v2a/jm-v2a_sapd.pb'
# # ModelPath = r'/home/wangyr/code/automl/efficientdet/tmp/frozen_model_v2a/jm-v2a_frozen.pb'
# # ModelPath = r'/home/wangyr/code/seg/fcos_v2a3_sapd.pb'
# # PATH_TO_LABELS = r"/home/wangyr/data/jm2021_adas_include_dongfeng/like13_7eval.pbtxt"
# # tf_queue = ['/home/wangyr/data/jm2021_adas_include_dongfeng/jm_val_remap_shuffle0.tfrecord']
# PATH_TO_LABELS = r"/home/wangyr/data/jm2021_adas_data/side15_14eval.pbtxt"
# tf_queue = ['/home/wangyr/data/jm2021_adas_data/jm_val_remap_shuffle0.tfrecord']
# # PATH_TO_LABELS = r"/data/adas/200w/side15_9eval.pbtxt"
# # tf_queue = ['/data/adas/200w/jm_val_remap_shuffle0.tfrecord']
# # PATH_TO_LABELS = r"/home/wangyr/data/jm_avm_include_dongfeng/avm5.pbtxt"
# # tf_queue = ['/home/wangyr/data/jm_avm_include_dongfeng/jm_val_remap_shuffle0.tfrecord']

# w = 576
# h = 320
# class_num = 6
# layer_num = 3
# ModelPath = r'/home/wangyr/code/automl/efficientdet/tmp/frozen_model_v2a/jm-v2a_frozen.pb'
# # ModelPath = r'/home/wangyr/code/seg/fcos_v2a_sapd.pb'
# PATH_TO_LABELS = r"/home/wangyr/data/jm2021_avm_sanyi/jm_avm_front/avm6eval.pbtxt"
# # tf_queue = ['/home/wangyr/data/jm2021_avm_sanyi/jm_avm_left_right/jm_val_remap_shuffle0.tfrecord']
# tf_queue = ['/home/wangyr/data/jm2021_avm_sanyi/jm_avm_front/jm_val_remap_shuffle0.tfrecord']
# # PATH_TO_LABELS = r"/home/wangyr/data/jm2021_adas_data/side15_14eval.pbtxt"
# # tf_queue = ['/home/wangyr/data/jm2021_adas_data/jm_val_remap_shuffle0.tfrecord']
# # PATH_TO_LABELS = r"/home/wangyr/data/jm_avm_include_dongfeng/avm5.pbtxt"
# # tf_queue = ['/home/wangyr/data/jm_avm_include_dongfeng/jm_val_remap_shuffle0.tfrecord']

w = 288
h = 160
class_num = 15
layer_num = 3
ModelPath = r'/home/wangyr/code/automl/efficientdet/tmp/frozen_model_v2tiny/jm-v2tiny_frozen.pb'
# PATH_TO_LABELS = r"/home/wangyr/data/jm2021_avm_sanyi/jm_avm_left_right/avm5eval.pbtxt"
# tf_queue = ['/home/wangyr/data/jm2021_avm_sanyi/jm_avm_left_right/jm_val_remap_shuffle0.tfrecord']
# tf_queue = ['/home/wangyr/data/jm2021_avm_sanyi/jm_avm_front/jm_val_remap_shuffle0.tfrecord']
PATH_TO_LABELS = r"/data/det_all/tiny9eval.pbtxt"
tf_queue = ['/data/det_all/jm_val_remap_shuffle0.tfrecord']


if quantize:
    for i in range(0, layer_num):
        name = 'BoxPredictor_%d/ClassPredictor/act_quant/FakeQuantWithMinMaxVars' % i
        preds_nanme.append(name)
        name = 'BoxPredictor_%d/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars' % i
        preds_nanme.append(name)
        # name = 'BoxPredictor_%d/CentrenessPredictor/act_quant/FakeQuantWithMinMaxVars' % i
        # preds_nanme.append(name)
else:
    for i in range(0, layer_num):
        name = 'BoxPredictor_%d/ClassOutput' % i
        preds_nanme.append(name)
        name = 'BoxPredictor_%d/BoxOutput' % i
        preds_nanme.append(name)
        # name = 'BoxPredictor_%d/ClassPredictor/BiasAdd' % i
        # preds_nanme.append(name)
        # name = 'BoxPredictor_%d/BoxEncodingPredictor/Relu' % i
        # preds_nanme.append(name)

stride = 4
double_stride = 8
fw = (w)//double_stride
fh = (h)//double_stride
include_background = False

score_threshold = 0.2  # 统一用0.01阈值来比较，其实当阈值为0时更高一点，但侯选框过多不方便计算

shape_list = [[fh*2, fw*2], [fh, fw], [(fh + 1) // 2, (fw + 1) // 2], [(fh + 3) // 4, (fw + 3) // 4], [(fh + 7) // 8, (fw + 7) // 8], [(fh + 15) // 16, (fw + 15) // 16]]
stride_list = [stride, stride<<1, stride<<2, stride<<3, stride<<4, stride<<5]
shape_list = shape_list[:layer_num]
stride_list = stride_list[:layer_num]

nms_thresh = 0.6

windows_length = 20
windows_thresh = 0.25
windows_scale = 1.2
windows_list = []
global_idx = 0
w_recall = 0.4

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'kitti_label_map.pbtxt')
# PATH_TO_LABELS = os.path.join('data', 'tage_label_map.pbtxt')
Eval_metrics_list = ['pascal_voc_detection_metrics', 'coco_detection_metrics']  # pascal_voc_detection_metrics, coco_detection_metrics

Box_filter = [0., 1.]
# Box_filter = [0., 0.1]
# Box_filter = [0.1, 0.2]
# Box_filter = [0.2, 1.]

# A dictionary of metric names to classes that implement the metric. The classes
# in the dictionary must implement
# utils.object_detection_evaluation.DetectionEvaluator interface.
EVAL_METRICS_CLASS_DICT = {
    'pascal_voc_detection_metrics':
        object_detection_evaluation.PascalDetectionEvaluator,
    'weighted_pascal_voc_detection_metrics':
        object_detection_evaluation.WeightedPascalDetectionEvaluator,
    'pascal_voc_instance_segmentation_metrics':
        object_detection_evaluation.PascalInstanceSegmentationEvaluator,
    'weighted_pascal_voc_instance_segmentation_metrics':
        object_detection_evaluation.WeightedPascalInstanceSegmentationEvaluator,
    'open_images_detection_metrics':
        object_detection_evaluation.OpenImagesDetectionEvaluator,
    'coco_detection_metrics':
        coco_evaluation.CocoDetectionEvaluator,
    'coco_mask_metrics':
        coco_evaluation.CocoMaskEvaluator,
    # 'coco_evaluation_all_frames':        coco_evaluation_all_frames.CocoEvaluationAllFrames,
}

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
name_list = label_map_util.create_categories_from_labelmap(PATH_TO_LABELS, use_display_name=True)
label_dict = label_map_util.get_label_map_dict(PATH_TO_LABELS, use_display_name=False)
if class_num is not None and class_num>0:
    name_list = name_list[:class_num+1] if 0 in category_index.keys() else name_list[:class_num]
    category_index = {x['id']:category_index[x['id']] for x in name_list}

# Evaluators = [EVAL_METRICS_CLASS_DICT[x](name_list) for x in Eval_metrics_list]

Evaluators = []
Evaluators.append(object_detection_evaluation.ObjectDetectionEvaluator(
        name_list,
        matching_iou_threshold=0.5,
        evaluate_corlocs=False,
        evaluate_precision_recall=True,
        metric_prefix='PascalBoxes',
        use_weighted_mean_ap=False))
Evaluators.append(EVAL_METRICS_CLASS_DICT['coco_detection_metrics'](name_list))


def center_scale_image(img, target_shape):
    ho,wo = img.shape[:2]
    _w, _h = target_shape
    upscale = 1
    img_upsample = img#cv2.resize(img, (wo*upscale, ho*upscale))

    x0 = np.arange(0, _w) / (_w - 1)
    x1 = np.arange(0, _h) / (_h - 1)
    y0 = 0.5*(1-np.tan(theta-x0*2*theta)/np.tan(theta))
    y1 = 0.5*(1-np.tan(theta-x1*2*theta)/np.tan(theta))
    yi = np.round(y1*(ho*upscale-1)).astype(np.int32)
    xi = np.round(y0*(wo*upscale-1)).astype(np.int32)
    xa,ya = np.meshgrid(xi, yi)
    imgscaled = img_upsample[ya, xa]

    return imgscaled

def is_box_valid(boxh, boxw, img_size, min_max_value=[0., 1.]):
    imgh, imgw = img_size
    valid = np.logical_and(boxh>0, boxw>0)
    large_side = np.maximum(boxh/imgh, boxw/imgw)
    return np.logical_and(np.logical_and(large_side>min_max_value[0], large_side<=min_max_value[1]), valid)


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(ModelPath, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def get_offset_by_windows(img_h, img_w, stride):
    offset_x = (((img_w-1) % stride) + 1)//2
    offset_y = (((img_h-1) % stride) + 1)//2
    return offset_x, offset_y

def compute_locations_per_level(stride, img_h, img_w, offset=None):
    if offset is None:
        offset = get_offset_by_windows(img_h, img_w, stride)
    range_x = np.arange(
        offset[0], img_w, step=stride,
        dtype=np.float32)
    range_y = np.arange(
        offset[1], img_h, step=stride,
        dtype=np.float32)
    shift_x, shift_y = np.meshgrid(range_x, range_y)
    shift_x = np.clip(shift_x, 0, img_w-1)
    shift_y = np.clip(shift_y, 0, img_h-1)
    locations = np.stack((shift_x, shift_y), axis=-1)
    # locations = tf.cast(locations, tf.float32)
    return locations


def py_cpu_nms(dets, thresh, refine=False):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    refine_dets = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w_ = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h_ = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w_ * h_
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        if refine:
            pos_idx = np.where(ious > thresh)[0]
            pos_index = np.concatenate([[i], index[pos_idx + 1]], axis=0)
            pos_det = dets[pos_index]
            _min_scores = score_threshold #np.min(pos_det[:,4])
            _weights = pos_det[:,4] - _min_scores
            _weights = _weights / np.maximum(np.sum(_weights), 1e-7)
            _refine_det = pos_det * _weights[:,None]
            _refine_det = np.sum(_refine_det, 0)
            _refine_det[4:] = pos_det[0,4:]
            refine_dets.append(_refine_det)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return np.asarray(refine_dets) if refine else dets[keep]

def py_cpu_nms_by_class(dets, thresh):
    dets_dict = {}
    for det in dets:
        if det[5] in dets_dict:
            dets_dict[det[5]].append(det)
        else:
            dets_dict[det[5]] = [det]

    result = []
    for key, value in dets_dict.items():
        dets_ = np.asarray(value)
        re = py_cpu_nms(dets_, thresh)
        result.append(re)
    if len(result)>0:
        result = np.concatenate(result, axis=0)
    else:
        result = np.zeros([0,6], np.float32)

    return result

def run_inference_for_single_image(sess, image, graph, imgid, use_windows = False):
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    image_scale_y = float(h) / float(height)
    image_scale_x = float(w) / float(width)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(float(height) * image_scale)
    scaled_width = int(float(width) * image_scale)
    image_padding = ((0,h-scaled_height),(0,w-scaled_width), (0,0))
    scaled_image_padding = np.asarray(image_padding) * [[1./image_scale], [1./image_scale], [1]]

    if center_scale:
      image_resized = center_scale_image(image, (w, h))
    else:
        scaled_image = cv2.resize(image, (scaled_width, scaled_height))
        image_resized = np.pad(scaled_image,image_padding,'constant', constant_values=0)
        # image_resized = cv2.resize(image, (w, h))

    image_np = (image_resized - 128.)/128.
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Get handles to input and output tensors
    # ops = tf.get_default_graph().get_operations()
    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in preds_nanme:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
    # image_tensor = graph.get_tensor_by_name('image_tensor:0')
    image_tensor = graph.get_tensor_by_name('normalized_input_image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: image_np_expanded})

    preds_value = [output_dict[x] for x in preds_nanme]

    scores = []
    ltrbs = []
    # centers = []
    locations = []

    num_preds_per_layer = len(preds_nanme) // layer_num
    for idx, shape in enumerate(shape_list):
      fh_, fw_ = shape[0]//2, shape[1]//2
      scores.append(np.reshape(preds_value[num_preds_per_layer*idx], [fh_*fw_, 4, -1]))
      ltrb_scale = 2**(idx+2) if ralative else 1
      ltrbs.append(np.reshape(preds_value[num_preds_per_layer*idx+1] * ltrb_scale, [fh_*fw_, 4, -1]))
      # centers.append(np.reshape(preds_value[num_preds_per_layer*idx+2], [fh_*fw_, 4]))

      # location_ = compute_locations_per_level(stride_list[idx]*2, h, w, offset = get_offset_by_windows(h, w, stride_list[idx]))
      # location_ = np.reshape(location_, [fh_*fw_, 1, 2])
      # location_ = np.tile(location_, [1,4,1])
      # location_ = location_ + [[0,0], [4,0], [0,4], [4,4]]

      location_ = compute_locations_per_level(stride_list[idx], h, w)
      location_ = np.reshape(location_, [fh_, 2, fw_, 2, 2])
      location_ = np.transpose(location_, [0, 2, 1, 3, 4])
      location_ = np.reshape(location_, [fh_ * fw_, 4, 2])

      locations.append(location_)


    scores = np.concatenate(scores, axis=0)
    ltrbs = np.concatenate(ltrbs, axis=0)
    # centers = np.concatenate(centers, axis=0)
    locations = np.concatenate(locations, axis=0)

    # sogmoid激活
    scores = 1 / (1 + np.exp(-scores))
    # centers = 1 / (1 + np.exp(-centers))

    if include_background:
      scores_max = np.max(scores[..., 1:], axis=-1)
      mixscore = scores_max# * centers
      classes = np.argmax(scores[..., 1:], axis=-1)+1   # classes 0 is background
    else:
      scores_max = np.max(scores, axis=-1)
      mixscore = scores_max# * centers
      classes = np.argmax(scores, axis=-1)+1  # classes 0 is background

    mask = (mixscore > score_threshold)

    scores_valid = mixscore[mask]
    classes_valid = classes[mask]
    ltrbs_valid = ltrbs[np.tile(mask[..., None], [1, 1, 4])]
    ltrbs_valid = np.reshape(ltrbs_valid, [-1, 4])
    ltrbs_valid = np.maximum(ltrbs_valid, 0)
    locations_valid = locations[np.tile(mask[..., None], [1, 1, 2])]
    locations_valid = np.reshape(locations_valid, [-1, 2])
    loc_x = locations_valid[:, 0]
    loc_y = locations_valid[:, 1]
    xmin = loc_x - ltrbs_valid[:, 0]
    ymin = loc_y - ltrbs_valid[:, 1]
    xmax = loc_x + ltrbs_valid[:, 2]
    ymax = loc_y + ltrbs_valid[:, 3]
    xmin = np.clip(xmin, 0, w)
    ymin = np.clip(ymin, 0, h)
    xmax = np.clip(xmax, 0, w)
    ymax = np.clip(ymax, 0, h)


    image_cpy = image.copy()
    # cv2.imwrite('/home/wangyr/temp/output/test.png', image_cpy[...,::-1])
    dets = np.stack([ymin, xmin, ymax, xmax, scores_valid, classes_valid], axis=1)
    # re = py_cpu_nms(dets, 0.6)
    re = py_cpu_nms_by_class(dets, nms_thresh)

    imgh, imgw = image.shape[0:2]
    if center_scale:
      re[:,0] = 0.5 * (1 - np.tan(theta - re[:,0] / (h) * 2 * theta) / np.tan(theta))
      re[:,2] = 0.5 * (1 - np.tan(theta - re[:,2] / (h) * 2 * theta) / np.tan(theta))
      re[:,1] = 0.5 * (1 - np.tan(theta - re[:,1] / (w) * 2 * theta) / np.tan(theta))
      re[:,3] = 0.5 * (1 - np.tan(theta - re[:,3] / (w) * 2 * theta) / np.tan(theta))
      scale = [[(imgh), (imgw), (imgh), (imgw), 1, 1]]
      re = re * scale - [scaled_image_padding[0, 0], scaled_image_padding[1, 0], scaled_image_padding[0, 0], scaled_image_padding[1, 0], 0, 0]
    else:
      re[:,:4] = re[:,:4] / image_scale - [scaled_image_padding[0, 0], scaled_image_padding[1, 0], scaled_image_padding[0, 0], scaled_image_padding[1, 0]]
      # re[:,:4] = (re[:,:4] - [image_padding[0,0], image_padding[1,0], image_padding[0,1], image_padding[1,1]]) / np.asarray([h,w,h,w])

    # this step is not necessary
    valid_box_mask = is_box_valid((re[:, 2]-re[:, 0]), (re[:, 3]-re[:, 1]), (imgh, imgw), Box_filter)

    # # filter rect
    # filters = np.logical_and(np.logical_and(0.5*(re[..., 2]+re[..., 0])<0.66*w, re[..., 5] != 7), re[..., 5] != 8)  #np.logical_and((re[..., 2]-re[..., 0])*(re[..., 3]-re[..., 1])<(0.33*w*h), 0.5*(re[..., 2]+re[..., 0])<0.66*w)
    # re = re[filters]

    # resize rect
    # scale = [[(imgh-1)/(h-1), (imgw-1)/(w-1), (imgh-1)/(h-1), (imgw-1)/(w-1), 1, 1]]
    # scale = [[(imgh), (imgw), (imgh), (imgw), 1, 1]]
    # re = re*scale - [scaled_image_padding[0,0], scaled_image_padding[1,0], scaled_image_padding[0,0], scaled_image_padding[1,0], 0, 0]

    global global_idx, Evaluators

    if any(valid_box_mask):
      re = re[valid_box_mask]
      for evaluator in Evaluators:
          evaluator.add_single_detected_image_info(
              image_id=imgid,
              detections_dict={
                  standard_fields.DetectionResultFields.detection_boxes:
                      re[..., :4],
                  standard_fields.DetectionResultFields.detection_scores:
                      re[..., 4],
                  standard_fields.DetectionResultFields.detection_classes:
                      re[..., 5]
              })

    for v in re:
      _re = np.asarray(v, dtype=np.int)
      y1, x1, y2, x2, s, c = _re[:6]
      cv2.rectangle(image_cpy, (x1, y1), (x2, y2), (0, 255, 0), 1)
    # plt.imshow(image_cpy)
    # plt.imsave('result.jpg', image_cpy)

    return image_cpy[...,::-1]


# video_path ="/home/wangyr/data/广德房屋误检35s处.ts"
# output_path = video_path.replace('.ts', '_predict.avi')
# videoCapture = cv2.VideoCapture(video_path)
# fps = videoCapture.get(cv2.CAP_PROP_FPS)
# size = (
#     int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
#     int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# )
#
# videoWriter = cv2.VideoWriter(
#     output_path,
#     # cv2.VideoWriter_fourcc("I", "4", "2", "0"),
#     cv2.VideoWriter_fourcc(*'XVID'),
#     # cv2.VideoWriter_fourcc(*'H264'),
#     fps,
#     size
# )


# filename_queue表示一个文件名队列，后面会讲到，比如我需要解析train.tfrecords文件的话，传入的就应该是训练的文件名队列
def read_and_decode(filename_queue, batch_size=32):
    """
    read and decode
    """

    # 创建一个tfrecords文件读取器并读取文件
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
    # width = tf.cast(parsed_features['image/width'], tf.int32)
    # height = tf.cast(parsed_features['image/height'], tf.int32)
    width = tf.cast(tf.shape(image)[1], tf.int32)
    height = tf.cast(tf.shape(image)[0], tf.int32)
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

with detection_graph.as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # success, frame = videoCapture.read()
        # while success:
        #     image = frame[..., ::-1]
        #     image_out = run_inference_for_single_image(sess, image, detection_graph, use_windows=True)
        #
        #
        #     videoWriter.write(image_out)
        #     success, frame = videoCapture.read()

        # init_op = tf.group(tf.global_variables_initializer(),
        #                    tf.local_variables_initializer())
        # sess.run(init_op)
        tf.global_variables_initializer().run(session=sess)
        tf.local_variables_initializer().run(session=sess)


        filename_queue = tf.train.string_input_producer(tf_queue, shuffle=False)#, capacity=1024, num_epochs=1)
        images, filename, source_id, width, height, cls, cls_name, xmin, xmax, ymin, ymax = read_and_decode(filename_queue)
        sample_num = []
        for tf_record in tf_queue:
            c = 0
            for record in tf.python_io.tf_record_iterator(tf_record):
                c += 1
            sample_num.append(c)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # for i in range(1):
        #     imgs = sess.run([images])
        #     plt.show()

        print('total pic number: %d' % sum(sample_num))

        try:
            idx = 0
            img_set = set()
            cls_list = [0]*30
            total_num = min(sum(sample_num), MaxEvalNum) if MaxEvalNum>0 else sum(sample_num)
            import time
            time_start = time.time()
            while not coord.should_stop() and idx<total_num:
                # Run training steps or whatever
                if idx % 100 == 0:
                    time_end = time.time()
                    print('[%d / %d] time: %0.2fs' % (idx, total_num, time_end - time_start))
                    time_start = time.time()
                imgs, imgname, imgid, w_, h_, c_, c_str, x1, x2, y1, y2 = sess.run([images, filename, source_id, width, height, cls, cls_name, xmin, xmax, ymin, ymax])
                if imgname in img_set:
                    print(str(imgname,'utf-8') + ' already exist!')
                    idx = idx + 1
                    continue
                img_set.add(imgname)

                c_refine = np.asarray([label_dict[str(x,'utf-8')] if str(x,'utf-8') in label_dict else -1 for x in c_str])
                for c_one in c_:
                    cls_list[c_one] = cls_list[c_one] + 1

                x1=np.maximum(x1, 0)
                y1=np.maximum(y1, 0)
                x2=np.minimum(x2, 1)
                y2=np.minimum(y2, 1)

                # assert np.all(np.stack([x1, x2, y1, y2], axis=-1)<=1)
                # assert np.all(np.stack([x1, x2, y1, y2], axis=-1)>=0)
                if not (np.all(x2>=x1) and np.all(y2>=y1)):
                    print('here')
                # assert np.all(x2>=x1) and np.all(y2>=y1)
                valid_box_mask = is_box_valid((y2 - y1), (x2 - x1), (h_, w_), Box_filter)
                valid_box_mask = np.logical_and(valid_box_mask, np.asarray(c_refine)>=0)

                # image_cpy = np.asarray(imgs)
                # for _y1, _x1, _y2, _x2, _c in zip(y1*h_,x1*w_,y2*h_,x2*w_,c_refine):
                #     y1i, x1i, y2i, x2i = int(_y1), int(_x1), int(_y2), int(_x2)
                #     cv2.rectangle(image_cpy, (x1i, y1i), (x2i, y2i), (0, 255, 0), 1)
                # # plt.imshow(image_cpy)
                # plt.imsave('result.jpg', image_cpy)

                if any(valid_box_mask):
                    for evaluator in Evaluators:
                        evaluator.add_single_ground_truth_image_info(
                            image_id=imgname,
                            groundtruth_dict={
                                standard_fields.InputDataFields.groundtruth_boxes:
                                    np.stack([y1[valid_box_mask]*h_, x1[valid_box_mask]*w_, y2[valid_box_mask]*h_, x2[valid_box_mask]*w_], axis=-1),
                                standard_fields.InputDataFields.groundtruth_classes: c_refine[valid_box_mask]
                            })
                else:
                    for evaluator in Evaluators:
                        evaluator.add_single_ground_truth_image_info(
                            image_id=imgname,
                            groundtruth_dict={
                                standard_fields.InputDataFields.groundtruth_boxes:
                                    np.zeros([0,4], dtype=np.float32),
                                standard_fields.InputDataFields.groundtruth_classes: np.zeros([0], dtype=np.float32)
                            })

                image_out = run_inference_for_single_image(sess, np.asarray(imgs), detection_graph, imgname, use_windows=False)
                idx=idx+1
                # plt.show()

        except tf.errors.OutOfRangeError:
            print
            'Done training -- epoch limit reached'
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # coord.request_stop()
        coord.join(threads)

        for id, value in category_index.items():
            print('%s number: %d'%(value['name'], cls_list[id]))

        for evaluator in Evaluators:
            metrics = evaluator.evaluate()
            print(metrics)

            if not any(['Score@' in x for x in metrics.keys()]):
                continue

            scores_map = {}
            precision_map = {}
            recall_map = {}
            for key, value in metrics.items():
                offset = key.find('Score@')
                if offset > 0:
                    scores_map[key[offset + 6:]] = value
                offset = key.find('Precision@')
                if offset > 0:
                    precision_map[key[offset + 10:]] = value
                offset = key.find('Recall@')
                if offset > 0:
                    recall_map[key[offset + 7:]] = value

            for key, s_ in scores_map.items():
                r_ = recall_map[key]
                p_ = precision_map[key]
                weighted_f1_ = 1 / (w_recall/np.maximum(r_, 1e-9) + (1-w_recall)/np.maximum(p_, 1e-9))
                f1_ = 2 / (1 / np.maximum(r_, 1e-9) + 1 / np.maximum(p_, 1e-9))
                if weighted_f1_.size > 0:
                    idx_ = np.argmax(weighted_f1_)
                    print(type(s_))
                        # print('%s: total=%d max_index=%d threshold=%f precision=%f recall=%f f1_@w_r%.2f=%f'%(key, np.size(s_), idx_,
                    #         s_[idx_] if np.size(s_)>1 else s_, p_[idx_] if np.size(p_)>1 else p_,
                        #         r_[idx_] if np.size(r_)>1 else r_, w_recall, weighted_f1_[idx_] if np.size(weighted_f1_)>1 else weighted_f1_[idx_]))
                    print('%s: total=%d max_index=%d threshold@w_r%.2f=%f precision=%f recall=%f f1_=%f'%(key, np.size(s_), idx_, w_recall,
                            s_[idx_] if np.size(s_)>1 else s_, p_[idx_] if np.size(p_)>1 else p_,
                            r_[idx_] if np.size(r_)>1 else r_, f1_[idx_] if np.size(f1_)>1 else f1_[idx_]))
