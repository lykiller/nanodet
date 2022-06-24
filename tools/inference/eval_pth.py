import os

import torch

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.data.dataset.coco import CocoDataset
from nanodet.evaluator.coco_detection import CocoDetectionEvaluator
import argparse
import os
import time

import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

model_dir = 'D:\\nanodet\\workspace\\rep_esnet\\jm_adas_coco\\nanodet-plus-m_320\\model_best'
train_model_name = 'nanodet_model_best.pth'
deploy_model_name = 'deploy_'+train_model_name
config_path = 'D:\\nanodet\\config\\rep_esnet\\jm2021_adas_coco\\nanodet-plus-m_320.yml'
load_config(cfg, config_path)


class Predictor(object):
    def __init__(self, cfg, model_path, deploy=True, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        logger = Logger(-1, cfg.save_dir, False)
        cfg.model.arch.backbone.update({'deploy': deploy})
        model = build_model(cfg.model)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        if checkpoint.__contains__('state_dict'):
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=True
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img


model_path = os.path.join(model_dir, deploy_model_name)
predict = Predictor(cfg, model_path, deploy=True)

val_images_dir = 'D:\\dataset\\jm2021_adas_coco\\val_images'
for img in os.listdir(val_images_dir):
    meta, results = predict.inference(os.path.join(val_images_dir, img))
    predict.visualize(results[0], meta, cfg.class_names, 0.35)
    cv2.waitKey(0)
