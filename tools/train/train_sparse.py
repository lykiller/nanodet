#!/usr/bin/env python
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from collections import OrderedDict, defaultdict

import hydra
import torch
import wandb
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch.cuda import amp
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import detectron2.utils.comm as d2_comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluators, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, get_event_storage


import tridet.modeling  # pylint: disable=unused-import
import tridet.utils.comm as comm
from tridet.data import build_test_dataloader, build_train_dataloader
from tridet.data.dataset_mappers import get_dataset_mapper
from tridet.data.datasets import random_sample_dataset_dicts, register_datasets
from tridet.evaluators import get_evaluator
from tridet.modeling import build_tta_model
from tridet.utils.s3 import sync_output_dir_s3
from tridet.utils.setup import setup
from tridet.utils.train import get_inference_output_dir, print_test_results
from tridet.utils.visualization import mosaic, save_vis
from tridet.utils.wandb import flatten_dict, log_nested_dict
from tridet.visualizers import get_dataloader_visualizer, get_predictions_visualizer

import numpy as np  #2022/5/13  add for test image and vidoe without gt
import cv2
from detectron2.structures.boxes import BoxMode
from tridet.structures.pose import Pose
from pyquaternion import Quaternion
#from tridet.structures.boxes3d import GenericBoxes3D
from tridet.utils.visualization import change_color_brightness, draw_text, fill_color_polygon
from tridet.utils.geometry import project_points3d
from detectron2.utils.visualizer import VisImage, _create_text_labels
from tridet.structures.boxes3d import Boxes3D, GenericBoxes3D
from detectron2.data import detection_utils as d2_utils

from sparse import *

LOG = logging.getLogger('tridet')



_class_names = ['car', 'Pedestrian', 'Cyclist', 'Van', 'Truck']


def pretty_render_3d_box(
    box3d,
    image,
    # camera,
    K,
    *,
    class_id=None,
    class_names=None,
    score=None,
    line_thickness=3,
    color=None,
    render_label=True
):
    """Render the bounding box on the image. NOTE: CV2 renders in place.

    Parameters
    ----------
    box3d: GenericBoxes3D

    image: np.uint8 array
        Image (H, W, C) to render the bounding box onto. We assume the input image is in *RGB* format

    K: np.ndarray
        Camera used to render the bounding box.

    line_thickness: int, default: 1
        Thickness of bounding box lines.

    font_scale: float, default: 0.5
        Font scale used in text labels.

    draw_axes: bool, default: False
        Whether or not to draw axes at centroid of box.
        Note: Bounding box pose is oriented such that x-forward, y-left, z-up.
        This corresponds to L (length) along x, W (width) along y, and H
        (height) along z.

    draw_text: bool, default: False
        If True, renders class name on box.
    """
    if not isinstance(box3d, GenericBoxes3D):
        raise ValueError(f'`box3d` must be a type of `Genericboxes3D`: {type(box3d).__str__()}')
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or len(image.shape) != 3 or image.shape[2] != 3):
        raise ValueError('`image` must be a 3-channel uint8 numpy array')

    points2d = project_points3d(box3d.corners[0].cpu().numpy(), K)
    corners = points2d.T

    # Draw the sides (first)
    for i in range(4):
        cv2.line(
            image, (int(corners.T[i][0]), int(corners.T[i][1])), (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
            color,
            thickness=line_thickness
        )
    # Draw front (in red) and back (blue) face.
    cv2.polylines(image, [corners.T[:4].astype(np.int32)], True, color, thickness=line_thickness)
    cv2.polylines(image, [corners.T[4:].astype(np.int32)], True, color, thickness=line_thickness)

    front_face_as_polygon = corners.T[:4].ravel().astype(int).tolist()
    fill_color_polygon(image, front_face_as_polygon, color, alpha=0.5)

    V = VisImage(img=image)

    if render_label:
        # Render text label. Mostly copied from Visualizer.overlay_instances()
        label = _create_text_labels([class_id], [score] if score is not None else None, class_names)[0]
        # bottom-right corner
        text_pos = tuple([points2d[:, 0].min(), points2d[:, 1].max()])
        horiz_align = "left"
        lighter_color = change_color_brightness(tuple([c / 255. for c in color]), brightness_factor=0.8)
        H, W = image.shape[:2]
        default_font_size = max(np.sqrt(H * W) // 90, 10)
        height_ratio = (points2d[:, 1].max() - points2d[:, 1].min()) / np.sqrt(H * W)
        font_size = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * default_font_size)

        draw_text(V.ax, label, text_pos, font_size=font_size, color=lighter_color, horizontal_alignment=horiz_align)

    image = V.get_image()
    return image

def convert_3d_box_to_kitti(box):
    """Convert a single 3D bounding box (GenericBoxes3D) to KITTI convention. i.e. for evaluation. We
    assume the box is in the reference frame of camera_2 (annotations are given in this frame).

    Usage:
        >>> box_camera_2 = pose_02.inverse() * pose_0V * box_velodyne
        >>> kitti_bbox_params = convert_3d_box_to_kitti(box_camera_2)

    Parameters
    ----------
    box: GenericBoxes3D
        Box in camera frame (X-right, Y-down, Z-forward)

    Returns
    -------
    W, L, H, x, y, z, rot_y, alpha: float
        KITTI format bounding box parameters.
    """
    assert len(box) == 1

    quat = Quaternion(*box.quat.cpu().tolist()[0])
    tvec = box.tvec.cpu().detach().numpy()[0]
    sizes = box.size.cpu().detach().numpy()[0]

    # Re-encode into KITTI box convention
    # Translate y up by half of dimension
    tvec += np.array([0., sizes[2] / 2.0, 0])

    inversion = Quaternion(axis=[1, 0, 0], radians=np.pi / 2).inverse
    quat = inversion * quat

    # Construct final pose in KITTI frame (use negative of angle if about positive z)
    if quat.axis[2] > 0:
        kitti_pose = Pose(wxyz=Quaternion(axis=[0, 1, 0], radians=-quat.angle), tvec=tvec)
        rot_y = -quat.angle
    else:
        kitti_pose = Pose(wxyz=Quaternion(axis=[0, 1, 0], radians=quat.angle), tvec=tvec)
        rot_y = quat.angle

    # Construct unit vector pointing in z direction (i.e. [0, 0, 1] direction)
    # The transform this unit vector by pose of car, and drop y component, thus keeping heading direction in BEV (x-z grid)
    v_ = np.float64([[0, 0, 1], [0, 0, 0]])
    v_ = (kitti_pose * v_)[:, ::2]

    # Getting positive theta angle (we define theta as the positive angle between
    # a ray from the origin through the base of the transformed unit vector and the z-axis
    theta = np.arctan2(abs(v_[1, 0]), abs(v_[1, 1]))

    # Depending on whether the base of the transformed unit vector is in the first or
    # second quadrant we add or subtract `theta` from `rot_y` to get alpha, respectively
    alpha = rot_y + theta if v_[1, 0] < 0 else rot_y - theta
    # Bound from [-pi, pi]
    if alpha > np.pi:
        alpha -= 2.0 * np.pi
    elif alpha < -np.pi:
        alpha += 2.0 * np.pi
    alpha = np.around(alpha, decimals=2)  # KITTI precision

    # W, L, H, x, y, z, rot-y, alpha
    return sizes[0], sizes[1], sizes[2], tvec[0], tvec[1], tvec[2], rot_y, alpha

def process_3d(inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input_per_image, pred_per_image in zip(inputs, outputs):
            pred_classes = pred_per_image['instances'].pred_classes
            pred_boxes = pred_per_image['instances'].pred_boxes.tensor
            pred_boxes3d = pred_per_image['instances'].pred_boxes3d
            # pred_boxes3d = pred_per_image['instances'].pred_box3d_as_vec
            scores = pred_per_image['instances'].scores
            scores_3d = pred_per_image['instances'].scores_3d

            file_name = input_per_image['file_name']
            #image_id = input_per_image['image_id']

            #intris = input_per_image['intrinsics']
            img = input_per_image['image'].permute(1,2,0).numpy()
            viz_image = img.copy()
            #img_2d = img.copy()
            if len(pred_boxes3d) == 0:
                return img.copy()

            assert len(pred_classes) == len(pred_boxes3d), "Number of class IDs must be same with number of 3D boxes."

            if isinstance(pred_boxes3d, Boxes3D):
                intrinsics = torch.unique(torch.inverse(pred_boxes3d.inv_intrinsics), dim=0)
                assert len(intrinsics) == 1, "Input 3D boxes must share intrinsics."
                _intrinsics = intrinsics[0]

                if intrinsics is not None:
                    assert torch.allclose(_intrinsics, torch.as_tensor(intrinsics).reshape(3, 3))
                intrinsics = _intrinsics
                K = intrinsics.detach().cpu().numpy().copy()
            else:
                assert intrinsics is not None
                K = np.float32(intrinsics).reshape(3, 3)

            # for class_id, box3d_as_vec, score, box2d in zip(pred_classes, pred_boxes3d, scores, pred_boxes):
            for k, (class_id, box3d, score_3d) in enumerate(zip(
                pred_classes, pred_boxes3d, scores_3d)
            ):
                if isinstance(box3d, GenericBoxes3D):
                    box3d = box3d.vectorize()[0]
                if isinstance(box3d, torch.Tensor):
                    box3d = box3d.detach().cpu().numpy()
                class_names = _class_names[class_id]
                box3d = GenericBoxes3D.from_vectors([box3d])
                if scores is not None:
                    score = scores[k]
                else:
                    score = None

                viz_image = pretty_render_3d_box(
                box3d,
                viz_image,
                # camera,
                K,
                class_id=class_id,
                color=(0,0,255),
                class_names=_class_names,
                score=score_3d,
                line_thickness=2,
                render_label=True
            )
        return viz_image


def process_2d(inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input_per_image, pred_per_image in zip(inputs, outputs):
            pred_classes = pred_per_image['instances'].pred_classes
            pred_boxes = pred_per_image['instances'].pred_boxes.tensor
            pred_boxes3d = pred_per_image['instances'].pred_boxes3d
            # pred_boxes3d = pred_per_image['instances'].pred_box3d_as_vec
            scores = pred_per_image['instances'].scores
            scores_3d = pred_per_image['instances'].scores_3d

            file_name = input_per_image['file_name']
            #image_id = input_per_image['image_id']

            # predictions
            predictions_kitti = []
            # for class_id, box3d_as_vec, score, box2d in zip(pred_classes, pred_boxes3d, scores, pred_boxes):
            for class_id, box3d, score_3d, box2d, score in zip(
                pred_classes, pred_boxes3d, scores_3d, pred_boxes, scores
            ):
                # class_name = self._metadata.thing_classes[class_id]
                class_name = _class_names[class_id]

                #box3d_as_vec = box3d.vectorize()[0].cpu().numpy()

                
                

                # prediction in KITTI format.
                W, L, H, x, y, z, rot_y, alpha = convert_3d_box_to_kitti(box3d)
                l, t, r, b = box2d.tolist()
                predictions_kitti.append([
                    class_name, -1, -1, alpha, l, t, r, b, H, W, L, x, y, z, rot_y,
                    float(score)
                ])
        return predictions_kitti

@hydra.main(config_path="configs/experiments/", config_name="dd3d_kitti_dla34_overfit")
def main(cfg):
    setup(cfg)

    if cfg.TEST_IMAGE:
        fileName = '/home/liuqingsong/data/simulation'
        output_name_3d = '/data/liuqingsong/project/pytorch/dd3d/test_res/3d'
        output_name_2d = '/data/liuqingsong/project/pytorch/dd3d/test_res/2d'
        intrinsics = np.array([[640, 0, 640],[0, 772, 360],[0, 0, 1]])
        intrinsics = np.reshape(intrinsics, (3, 3)).astype(np.float32)
        model = build_model(cfg)
        checkpoint_file = cfg.MODEL.CKPT
        if checkpoint_file:
            Checkpointer(model).load(checkpoint_file)
        listName = os.listdir(fileName)
        for i, name in enumerate(listName):
            datum = {}
            
            datum["intrinsics"] = torch.as_tensor(intrinsics)
            datum["inv_intrinsics"] = torch.as_tensor(np.linalg.inv(intrinsics))
            # datum['intrinsics'] = list(intrinsics.flatten())
            # Consistent with COCO format
            datum['file_name'] = os.path.join(fileName, name)   #2022/4/22  modify kitti to bdd

            image = cv2.imread(datum['file_name'])  #d2_utils.read_image(datum['file_name'], format=cfg.INPUT.FORMAT)
            image_show = cv2.imread(datum['file_name']) #[:,:,::-1]
            datum['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            data_dict = []
            data_dict.append(datum)
            model.training = False
            outputs = model(data_dict)
            result = process_3d(data_dict, outputs)
            output_dir_3d = os.path.join(output_name_3d, name)
            cv2.imwrite(output_dir_3d, result)

            result_2d = process_2d(data_dict, outputs)
            for _, res in enumerate(result_2d):
                cls = res[0]
                score = res[15]
                txt = '{}_{:.2f}'.format(cls, score)
                l, t, r, b = res[4:8]
                cv2.rectangle(image_show,(int(l),int(t)),(int(r),int(b)),(0,0,255),2)
                cv2.putText(image_show, txt, (int(l),int(t)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            output_dir_2d = os.path.join(output_name_2d, name)
            cv2.imwrite(output_dir_2d,image_show)

        return

    if cfg.TEST_VIDEO:
        return
    dataset_names = register_datasets(cfg)
    if cfg.ONLY_REGISTER_DATASETS:
        return {}, cfg
    LOG.info(f"Registered {len(dataset_names)} datasets:" + '\n\t' + '\n\t'.join(dataset_names))

    model = build_model(cfg)

    checkpoint_file = cfg.MODEL.CKPT
    if checkpoint_file:
        Checkpointer(model).load(checkpoint_file)

    if cfg.EVAL_ONLY:
        assert cfg.TEST.ENABLED, "'eval-only' mode is not compatible with 'cfg.TEST.ENABLED = False'."
        test_results = do_test(cfg, model, is_last=True)
        if cfg.TEST.AUG.ENABLED:
            test_results.update(do_test(cfg, model, is_last=True, use_tta=True))
        return test_results, cfg

    if comm.is_distributed():
        assert d2_comm._LOCAL_PROCESS_GROUP is not None
        # Convert all Batchnorm*D to nn.SyncBatchNorm.
        # For faster training, the batch stats are computed over only the GPUs of the same machines (usually 8).
        sync_bn_pg = d2_comm._LOCAL_PROCESS_GROUP if cfg.SOLVER.SYNCBN_USE_LOCAL_WORKERS else None
        model = SyncBatchNorm.convert_sync_batchnorm(model, process_group=sync_bn_pg)
        model = DistributedDataParallel(
            model,
            device_ids=[d2_comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.SOLVER.DDP_FIND_UNUSED_PARAMETERS
        )

    do_train(cfg, model)
    test_results = do_test(cfg, model, is_last=True)
    if cfg.TEST.AUG.ENABLED:
        test_results.update(do_test(cfg, model, is_last=True, use_tta=True))
    return test_results, cfg


def do_train(cfg, model):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = Checkpointer(model, './', optimizer=optimizer, scheduler=scheduler)
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    writers = [CommonMetricPrinter(max_iter)] if d2_comm.is_main_process() else []

    train_mapper = get_dataset_mapper(cfg, is_train=True)
    dataloader, dataset_dicts = build_train_dataloader(cfg, mapper=train_mapper)
    LOG.info("Length of train dataset: {:d}".format(len(dataset_dicts)))
    LOG.info("Starting training")
    storage = get_event_storage()

    if cfg.EVAL_ON_START:
        do_test(cfg, model)
        comm.synchronize()

    # In mixed-precision training, gradients are scaled up to keep them from being vanished due to half-precision.
    # They're scaled down again before optimizers use them to compute updates.
    scaler = amp.GradScaler(enabled=cfg.SOLVER.MIXED_PRECISION_ENABLED)

    # Accumulate gradients for multiple batches (as returned by dataloader) before calling optimizer.step().
    accumulate_grad_batches = cfg.SOLVER.ACCUMULATE_GRAD_BATCHES

    num_images_seen = 0
    # For logging, this stores losses aggregated from all workers in distributed training.
    batch_loss_dict = defaultdict(float)
    optimizer.zero_grad()
    for data, iteration in zip(dataloader, range(max_iter * accumulate_grad_batches)):
        iteration += 1
        # this assumes drop_last=True, so all workers has the same size of batch.
        num_images_seen += len(data) * d2_comm.get_world_size()
        if iteration % accumulate_grad_batches == 0:
            storage.step()

        with amp.autocast(enabled=cfg.SOLVER.MIXED_PRECISION_ENABLED):
            loss_dict = model(data)
        # Account for accumulated gradients.
        loss_dict = {name: loss / accumulate_grad_batches for name, loss in loss_dict.items()}
        losses = sum(loss_dict.values())
        # FIXME: First few iterations might give Inf/NaN losses when using mixed precision. What should be done?
        if not torch.isfinite(losses):
            LOG.critical(f"The loss DIVERGED: {loss_dict}")

        # Track total loss for logging.
        loss_dict_reduced = {k: v.item() for k, v in d2_comm.reduce_dict(loss_dict).items()}
        assert torch.isfinite(torch.as_tensor(list(loss_dict_reduced.values()))).all(), loss_dict_reduced
        for k, v in loss_dict_reduced.items():
            batch_loss_dict[k] += v

        # No amp version: leaving this here for legacy:
        # losses.backward()
        scaler.scale(losses).backward()

        if iteration % accumulate_grad_batches > 0:
            # Just accumulate gradients and move on to next batch.
            continue

        scaler.step(optimizer)
        storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()
        scaler.update()

        losses_reduced = sum(loss for loss in batch_loss_dict.values())
        storage.put_scalars(total_loss=losses_reduced, **batch_loss_dict)

        # Reset states.
        batch_loss_dict = defaultdict(float)
        optimizer.zero_grad()

        batch_iter = iteration // accumulate_grad_batches

        # TODO: probably check if the gradients contain any inf or nan, and only proceed if not.
        if batch_iter > 5 and (batch_iter % 20 == 0 or batch_iter == max_iter):
            # if batch_iter > -1 and (batch_iter % 1 == 0 or batch_iter == max_iter):
            for writer in writers:
                writer.write()
            # log epoch, # images seen
            if d2_comm.is_main_process() and cfg.WANDB.ENABLED:
                wandb.log({"epoch": 1 + num_images_seen // len(dataset_dicts)}, step=batch_iter)
                wandb.log({"num_images_seen": num_images_seen}, step=batch_iter)

        if cfg.VIS.DATALOADER_ENABLED and batch_iter % cfg.VIS.DATALOADER_PERIOD == 0 and d2_comm.is_main_process():
            dataset_name = cfg.DATASETS.TRAIN.NAME
            visualizer_names = MetadataCatalog.get(dataset_name).loader_visualizers
            viz_images = defaultdict(dict)
            for viz_name in visualizer_names:
                if (viz_name != visualizer_names[0]):           #2022/5/18  add for only do 2d train...
                    print(viz_name)
                    print(' is not loader...')
                    break
                viz = get_dataloader_visualizer(cfg, viz_name, dataset_name)
                for idx, x in enumerate(data):
                    viz_images[idx].update(viz.visualize(x))

            if cfg.WANDB.ENABLED:
                per_image_vis = [mosaic(list(viz_images[idx].values())) for idx in range(len(data))]
                wandb.log({
                    "dataloader": [wandb.Image(vis, caption=f"idx={idx}") for idx, vis in enumerate(per_image_vis)]
                },
                          step=batch_iter)
            save_vis(viz_images, os.path.join(os.getcwd(), "visualization"), "dataloader", step=batch_iter)

        if d2_comm.is_main_process():
            periodic_checkpointer.step(batch_iter - 1)  # (fvcore) model_0004999.pth checkpoints 5000-th iteration

        if cfg.SYNC_OUTPUT_DIR_S3.ENABLED and batch_iter > 0 and batch_iter % cfg.SYNC_OUTPUT_DIR_S3.PERIOD == 0:
            sync_output_dir_s3(cfg)

        if (cfg.TEST.EVAL_PERIOD > 0 and batch_iter % cfg.TEST.EVAL_PERIOD == 0 and batch_iter != max_iter) or \
            batch_iter in cfg.TEST.ADDITIONAL_EVAL_STEPS:
            do_test(cfg, model)
            d2_comm.synchronize()


def do_test(cfg, model, is_last=False, use_tta=False):
    if not cfg.TEST.ENABLED:
        LOG.warning("Test is disabled.")
        return {}

    dataset_names = [cfg.DATASETS.TEST.NAME]  # NOTE: only support single test dataset for now.

    if use_tta:
        LOG.info("Starting inference with test-time augmentation.")
        if isinstance(model, DistributedDataParallel):
            model.module.postprocess_in_inference = False
        else:
            model.postprocess_in_inference = False
        model = build_tta_model(cfg, model)

    test_results = OrderedDict()
    for dataset_name in dataset_names:
        # output directory for this dataset.
        dset_output_dir = get_inference_output_dir(dataset_name, is_last=is_last, use_tta=use_tta)

        # What evaluators are used for this dataset?
        evaluator_names = MetadataCatalog.get(dataset_name).evaluators
        evaluators = []
        for evaluator_name in evaluator_names:
            if(evaluator_name != evaluator_names[0]):          #2022/5/18  add for only do 2d train...
                print(evaluator_name)
                print('is not loader...')
                break
            evaluator = get_evaluator(cfg, dataset_name, evaluator_name, dset_output_dir)
            evaluators.append(evaluator)
        evaluator = DatasetEvaluators(evaluators)

        mapper = get_dataset_mapper(cfg, is_train=False)
        dataloader, dataset_dicts = build_test_dataloader(cfg, dataset_name, mapper)

        per_dataset_results = inference_on_dataset(model, dataloader, evaluator)
        #2022/5/11 add for only test dense depth...
        #continue
        if use_tta:
            per_dataset_results = OrderedDict({k + '-tta': v for k, v in per_dataset_results.items()})
        test_results[dataset_name] = per_dataset_results

        if cfg.VIS.PREDICTIONS_ENABLED and d2_comm.is_main_process():
            visualizer_names = MetadataCatalog.get(dataset_name).pred_visualizers
            # Randomly (but deterministically) select what samples to visualize.
            # The samples are shared across all visualizers and iterations.
            sampled_dataset_dicts, inds = random_sample_dataset_dicts(
                dataset_name, num_samples=cfg.VIS.PREDICTIONS_MAX_NUM_SAMPLES
            )

            viz_images = defaultdict(dict)
            for viz_name in visualizer_names:
                if(viz_name != visualizer_names[0]):      #2022/5/18  add for only train 2d det...
                    print(viz_name)
                    print('is not loader...')
                    break
                LOG.info(f"Running prediction visualizer: {viz_name}")
                visualizer = get_predictions_visualizer(cfg, viz_name, dataset_name, dset_output_dir)
                for x in tqdm(sampled_dataset_dicts):
                    sample_id = x['sample_id']
                    viz_images[sample_id].update(visualizer.visualize(x))

            save_vis(viz_images, dset_output_dir, "visualization")

            if cfg.WANDB.ENABLED:
                LOG.info(f"Uploading prediction visualization to W&B: {dataset_name}")
                for sample_id in viz_images.keys():
                    viz_images[sample_id] = mosaic(list(viz_images[sample_id].values()))
                step = get_event_storage().iter
                wandb.log({
                    f"{dataset_name}-predictions":
                    [wandb.Image(viz, caption=f"{sample_id}") for sample_id, viz in viz_images.items()]
                },
                          step=step)

    test_results = flatten_dict(test_results)
    log_nested_dict(test_results)
    if d2_comm.is_main_process():
        LOG.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_test_results(test_results)

    if use_tta:
        if isinstance(model, DistributedDataParallel):
            model.module.postprocess_in_inference = True
        else:
            model.postprocess_in_inference = True

    return test_results


#2022/5/19  add for sparse training...
save_dir = '/data/liuqingsong/project/pytorch/dd3d/sparse_checkpoint'
checkpoint_dir = ''          #2022/5/19  must def firstly!!!
def save_checkpoint(net, optim, lr, epoch, iter, sparse_value):
    print('Saving state, iter:', iter)
    file_name =  '{}_{:.2d}.pth'.format(epoch, sparse_value * 100)
    checkpoint = {'model':net.state_dict(), 'optimizer':optim.state_dict(), 
            'scheduler':lr, 'iteration':iter, 'epoch':epoch}
    checkpoint_dir = os.path.join(save_dir, file_name)
    torch.save(checkpoint, checkpoint_dir)
    return

epoches = 10
def do_train_sparse(cfg, model, optimizer, lr_schedule, dataloader, sparse_value):
    model.train()
    optimizer.zero_grad()

    #2022/5/19  calc sparsity...
    nonzeros, total = 0, 0
    for k, m in model.stat_dict():
        nonzeros += (m == 0).sum()
        total += m.numel()
    result_sparsity = float(total - nonzeros) / total
    print('sparsity: ', result_sparsity)

    #2022/5/19  sparse...
    weight_mask = global_prune_init(model.stat_dict(), sparse_value, 0.95)
    sparsify(model.stat_dict(), weight_mask)

    #2022/5/19  calc sparsity...
    nonzeros, total = 0, 0
    for k, m in weight_mask.items():
        nonzeros += (m == 0).sum()
        total += m.numel()
    result_sparsity = float(total - nonzeros) / total
    print('sparsity: ', result_sparsity)


    #2022/5/19  log file
    log_dir = '/data/liuqingsong/project/pytorch/dd3d/log'
    log_name = 'log_sparse_{:.2d}.txt'.format(sparse_value * 100)
    log_file = log_dir + '/' + log_name
    f_file = open(log_file, 'a')
    for epoch in range(epoches):
        for iter, inputs in enumerate(dataloader):
            loss_dict = model(inputs)
            losses = sum(loss_dict.values())
            losses.backward()
            lr_schedule.step()
            optimizer.step()
            sparsify(model.state_dict(), weight_mask)
            
            if iter % 20 == 0:
                for k, m in loss_dict.items():
                    log_loss = '{}: {:.5f},'.format(k, m)
                log_iter = 'epoch: {}, iter: {}, lr: {}'.format(epoch, iter, optimizer.param_groups[0]['lr'])
                log_img = '{}*{}'.format(inputs['image'][0].shape[1], inputs['image'][0].shape[2])
                if torch.cuda.is_available():
                    log_mem = 'max_mem: {}'.format(torch.cuda.max_memory_allocated()/1024/1024)
                log_sparse = 'sparse: {}'.format(result_sparsity)
                log = log_iter + log_loss + log_sparse + log_img + log_mem
                print(log)
                f_file.write(log + '\n')
        save_checkpoint(model, optimizer, lr_schedule, epoch, iter, sparse_value)
    f_file.close()
    return

#@hydra.main(config_path="configs/experiments/", config_name="dd3d_kitti_dla34_overfit")
def train_sparse(cfg, dataloader, sparse = 0.2):
    
    # dataset_names = register_datasets(cfg)
    # if cfg.ONLY_REGISTER_DATASETS:
    #     return {}, cfg
    # LOG.info(f"Registered {len(dataset_names)} datasets:" + '\n\t' + '\n\t'.join(dataset_names))

    model = build_model(cfg)

    checkpoint_file = checkpoint_dir
    
    if checkpoint_file:
        Checkpointer(model).load(checkpoint_file)

    if cfg.EVAL_ONLY:
        assert cfg.TEST.ENABLED, "'eval-only' mode is not compatible with 'cfg.TEST.ENABLED = False'."
        test_results = do_test(cfg, model, is_last=True)
        if cfg.TEST.AUG.ENABLED:
            test_results.update(do_test(cfg, model, is_last=True, use_tta=True))
        return test_results, cfg

    if comm.is_distributed():
        assert d2_comm._LOCAL_PROCESS_GROUP is not None
        # Convert all Batchnorm*D to nn.SyncBatchNorm.
        # For faster training, the batch stats are computed over only the GPUs of the same machines (usually 8).
        sync_bn_pg = d2_comm._LOCAL_PROCESS_GROUP if cfg.SOLVER.SYNCBN_USE_LOCAL_WORKERS else None
        model = SyncBatchNorm.convert_sync_batchnorm(model, process_group=sync_bn_pg)
        model = DistributedDataParallel(
            model,
            device_ids=[d2_comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.SOLVER.DDP_FIND_UNUSED_PARAMETERS
        )

    #2022/5/19  add for sparse train...
    model.train()

    #2022/5/19  need modify the flow for the sparse train...
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    
    LOG.info("Starting training")
    do_train_sparse(cfg, model, optimizer, scheduler, dataloader, sparse)
  
    test_results = do_test(cfg, model, is_last=True)
    if cfg.TEST.AUG.ENABLED:
        test_results.update(do_test(cfg, model, is_last=True, use_tta=True))

@hydra.main(config_path="configs/experiments/", config_name="dd3d_kitti_dla34_overfit")
def main_sparse(cfg):

    dataset_names = register_datasets(cfg)
    if cfg.ONLY_REGISTER_DATASETS:
        return {}, cfg
    LOG.info(f"Registered {len(dataset_names)} datasets:" + '\n\t' + '\n\t'.join(dataset_names))

    train_mapper = get_dataset_mapper(cfg, is_train=True)
    dataloader, dataset_dicts = build_train_dataloader(cfg, mapper=train_mapper)
    LOG.info("Length of train dataset: {:d}".format(len(dataset_dicts)))

    
    density = 1.0
    for i in range(10):
        density *= 0.8
        train_sparse(cfg, dataloader, 1 - density)
    return

if __name__ == '__main__':
    # main()  # pylint: disable=no-value-for-parameter
    # LOG.info("DONE.")
    main_sparse()
    