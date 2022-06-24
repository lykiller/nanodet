# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Linear, Dropout
from torch.nn import AdaptiveAvgPool2d, MaxPool2d
from ..module.conv import RepConvBNLayer

from ..sparse_ops.syncbn_layer import SyncBatchNorm2d
import numpy as np

from ..module.se import SEModule
from ..module.conv import ConvBNLayer

MODEL_URLS = {
    "ESNet_x0_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_25_pretrained.pdparams",
    "ESNet_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_5_pretrained.pdparams",
    "ESNet_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_75_pretrained.pdparams",
    "ESNet_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x1_0_pretrained.pdparams",
}

MODEL_STAGES_PATTERN = {"ESNet": ["blocks[2]", "blocks[9]", "blocks[12]"]}

__all__ = list(MODEL_URLS.keys())


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ESBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, if_sparse=False):
        super().__init__()
        self.pw_1_1 = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            if_sparse=if_sparse
        )
        self.dw_1 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=1,
            groups=out_channels // 2,
            if_act=False,
            if_sparse=if_sparse
        )
        self.se = SEModule(out_channels)

        self.pw_1_2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            if_sparse=if_sparse
        )

    def forward(self, x):

        x1, x2 = torch.split(x, [x.shape[1] // 2, x.shape[1] // 2], dim=1)
        x2 = self.pw_1_1(x2)
        x3 = self.dw_1(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self.se(x3)
        x3 = self.pw_1_2(x3)
        x = torch.cat([x1, x3], dim=1)
        return channel_shuffle(x, 2)


class ESBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, if_sparse=False):
        super().__init__()

        # branch1
        self.dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            groups=in_channels,
            if_act=False,
            if_sparse=if_sparse
        )
        self.pw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1)
        # branch2
        self.pw_2_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1)
        self.dw_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=2,
            groups=out_channels // 2,
            if_act=False)
        self.se = SEModule(out_channels // 2)
        self.pw_2_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1)
        self.concat_dw = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            groups=out_channels)
        self.concat_pw = ConvBNLayer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.dw_1(x)
        x1 = self.pw_1(x1)
        x2 = self.pw_2_1(x)
        x2 = self.dw_2(x2)
        x2 = self.se(x2)
        x2 = self.pw_2_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.concat_dw(x)
        x = self.concat_pw(x)
        return x


class ESNet(nn.Module):
    def __init__(self,
                 class_num=1000,
                 scale=1.0,
                 dropout_prob=0.2,
                 class_expand=1280,
                 return_patterns=None,
                 return_stages=None,
                 model_size="1.0x",
                 out_stages=(2, 3, 4),
                 activation='ReLu',
                 if_sparse=False
                 ):
        super(ESNet, self).__init__()
        self.scale = scale
        self.class_num = class_num
        self.class_expand = class_expand
        stage_repeats = [3, 7, 3]
        # stage_out_channels = [-1, 24, make_divisible(116 * scale), make_divisible(232 * scale), make_divisible(464 * scale), 1024]
        if if_sparse:
            stage_out_channels = [-1, 24, 120, 240, 480, 1028]
        else:
            stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            if_sparse=if_sparse
        )
        self.max_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for stage_id, num_repeat in enumerate(stage_repeats):
            seq = []
            for i in range(num_repeat):
                if i == 0:
                    block = ESBlock2(
                        in_channels=stage_out_channels[stage_id + 1],
                        out_channels=stage_out_channels[stage_id + 2],
                        if_sparse=if_sparse
                    )
                else:
                    block = ESBlock1(
                        in_channels=stage_out_channels[stage_id + 2],
                        out_channels=stage_out_channels[stage_id + 2],
                        if_sparse=if_sparse
                    )
                seq.append(block)
            setattr(self, stage_names[stage_id], nn.Sequential(*seq))

    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.stage2(x)
        output.append(x)
        x = self.stage3(x)
        output.append(x)
        x = self.stage4(x)
        output.append(x)
        return tuple(output)


class RepESBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.pw_1_1 = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1)
        self.dw_1 = RepConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=1,
            groups=out_channels // 2,
            if_act=False,
            deploy=deploy
        )
        self.se = SEModule(out_channels)

        self.pw_1_2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1)

    def forward(self, x):

        x1, x2 = torch.split(x, [x.shape[1] // 2, x.shape[1] // 2], dim=1)
        x2 = self.pw_1_1(x2)
        x3 = self.dw_1(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self.se(x3)
        x3 = self.pw_1_2(x3)
        x = torch.cat([x1, x3], dim=1)
        return channel_shuffle(x, 2)


class RepESBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()

        # branch1
        self.dw_1 = RepConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            groups=in_channels,
            if_act=False,
            deploy=deploy
        )
        self.pw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1)
        # branch2
        self.pw_2_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1)
        self.dw_2 = RepConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=2,
            groups=out_channels // 2,
            if_act=False,
            deploy=deploy
        )
        self.se = SEModule(out_channels // 2)
        self.pw_2_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1)
        self.concat_dw = RepConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            groups=out_channels,
            deploy=deploy
        )
        self.concat_pw = ConvBNLayer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.dw_1(x)
        x1 = self.pw_1(x1)
        x2 = self.pw_2_1(x)
        x2 = self.dw_2(x2)
        x2 = self.se(x2)
        x2 = self.pw_2_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.concat_dw(x)
        x = self.concat_pw(x)
        return x


class RepESNet(nn.Module):
    def __init__(self,
                 class_num=1000,
                 scale=1.0,
                 dropout_prob=0.2,
                 class_expand=1280,
                 return_patterns=None,
                 return_stages=None,
                 model_size="1.0x",
                 out_stages=(2, 3, 4),
                 activation='ReLu',
                 deploy=False
                 ):
        super(RepESNet, self).__init__()
        self.scale = scale
        self.class_num = class_num
        self.class_expand = class_expand
        stage_repeats = [3, 7, 3]
        stage_out_channels = [-1, 24, make_divisible(116 * scale), make_divisible(232 * scale),
            make_divisible(464 * scale), 1024]
        stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        self.conv1 = RepConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            deploy=deploy
        )
        self.max_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for stage_id, num_repeat in enumerate(stage_repeats):
            seq = []
            for i in range(num_repeat):
                if i == 0:
                    block = RepESBlock2(
                        in_channels=stage_out_channels[stage_id + 1],
                        out_channels=stage_out_channels[stage_id + 2],
                        deploy=deploy
                    )
                else:
                    block = RepESBlock1(
                        in_channels=stage_out_channels[stage_id + 2],
                        out_channels=stage_out_channels[stage_id + 2],
                        deploy=deploy
                    )
                seq.append(block)
            setattr(self, stage_names[stage_id], nn.Sequential(*seq))

    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.stage2(x)
        output.append(x)
        x = self.stage3(x)
        output.append(x)
        x = self.stage4(x)
        output.append(x)
        return tuple(output)


def rep_det_model_convert(model, deploy_model):
    converted_weights = {}
    deploy_model.load_state_dict(model.state_dict(), strict=False)
    for name, module in model.backbone.named_modules():
        if hasattr(module, "rep_net_convert"):
            kernel, bias = module.rep_net_convert()
            converted_weights[name + ".rbr_reparam.weight"] = kernel
            converted_weights[name + ".rbr_reparam.bias"] = bias
    del model
    for name, param in deploy_model.backbone.named_parameters():
        if converted_weights.__contains__(name):

            print("deploy param: ", name, param.size(), np.mean(converted_weights[name]))
            param.data = torch.from_numpy(converted_weights[name]).float()
    return deploy_model
