# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from .custom_csp import CustomCspNet
from .efficientnet_lite import EfficientNetLite
from .ghostnet import GhostNet
from .mobilenetv2 import MobileNetV2
from .repvgg import RepVGG
from .resnet import ResNet
from .shufflenetv2 import ShuffleNetV2
from .shufflenetv3 import ShuffleNetV3
from .shufflenet_tiny import ShuffleNetTiny
from .shufflenet_tinyse import ShuffleNetTinySe
from .shufflenetse import ShuffleNetSe
from .shufflenet_lite import ShuffleNetLite
from .dsod_tiny import DsodTiny
from .dsod_tiny import DsodTinyV2
from .dsod_tiny import DsodTinyMnn
from .esnet import ESNet
from .mobilenet_one import MobileNetONE


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "ResNet":
        return ResNet(**backbone_cfg)
    elif name == "ShuffleNetV2":
        return ShuffleNetV2(**backbone_cfg)
    elif name == "GhostNet":
        return GhostNet(**backbone_cfg)
    elif name == "MobileNetV2":
        return MobileNetV2(**backbone_cfg)
    elif name == "EfficientNetLite":
        return EfficientNetLite(**backbone_cfg)
    elif name == "CustomCspNet":
        return CustomCspNet(**backbone_cfg)
    elif name == "RepVGG":
        return RepVGG(**backbone_cfg)
    elif name == "ShuffleNetV3":
        return ShuffleNetV3(**backbone_cfg)
    elif name == "ShuffleNetTiny":
        return ShuffleNetTiny(**backbone_cfg)
    elif name == "ShuffleNetTinySe":
        return ShuffleNetTinySe(**backbone_cfg)
    elif name == "ShuffleNetSe":
        return ShuffleNetSe(**backbone_cfg)
    elif name == "ShuffleNetLite":
        return ShuffleNetLite(**backbone_cfg)
    elif name == 'DsodTiny':
        return DsodTiny(**backbone_cfg)
    elif name == 'DsodTinyV2':
        return DsodTinyV2(**backbone_cfg)
    elif name == 'DsodTinyMnn':
        return DsodTinyMnn(**backbone_cfg)
    elif name == 'ESNet' or name == 'EsNet':
        return ESNet(**backbone_cfg)
    elif name == 'MobileNetONE':
        return MobileNetONE(**backbone_cfg)
    else:
        raise NotImplementedError
