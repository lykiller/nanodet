import os

import torch

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight



model_dir = 'D:\\nanodet\\workspace\\rep_esnet\\jm_adas_coco\\nanodet-plus-m_320\\model_best'

train_model_name = 'nanodet_model_best.pth'
config_path = 'D:\\nanodet\\config\\rep_esnet\\jm2021_adas_coco\\nanodet-plus-m_320.yml'

def reparameter_train_model(config, model_path, input_shape=(320, 320)):
    logger = Logger(-1, config.save_dir, False)
    model = build_model(config.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, checkpoint, logger)
    if config.model.arch.backbone.name == "RepVGG":
        deploy_config = config.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert
        model = repvgg_det_model_convert(model, deploy_model)

    if config.model.arch.backbone.name == "RepESNet":
        deploy_config = config.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.esnet import rep_det_model_convert
        model = rep_det_model_convert(model, deploy_model)

    torch.save(model.state_dict(), os.path.join(os.path.dirname(model_path), 'deploy_'+os.path.basename(model_path)))

    return model


load_config(cfg, config_path)
reparameter_train_model(cfg, os.path.join(model_dir, train_model_name))
