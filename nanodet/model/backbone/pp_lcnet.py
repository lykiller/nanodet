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
from torch import ParameterDict
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Dropout, Linear
from torch.regularizer import L2Decay

from nanodet.model.module.theseus import TheseusLayer
# from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "PPLCNet_x0_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams",
    "PPLCNet_x0_35":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams",
    "PPLCNet_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams",
    "PPLCNet_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams",
    "PPLCNet_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams",
    "PPLCNet_x1_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams",
    "PPLCNet_x2_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams",
    "PPLCNet_x2_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams"
}

MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

__all__ = list(MODEL_URLS.keys())

# Each element(list) represents a depthwise block, which is composed of k, in_c, out_c, s, use_se.
# k: kernel_size
# in_c: input channel number in depthwise block
# out_c: output channel number in depthwise block
# s: stride in depthwise block
# use_se: whether to use SE block

NET_CONFIG = {
    # k, in_c, out_c, s, use_se
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1,
                 lr_mult=1.0
                 ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)

        self.bn = nn.BatchNorm2d(num_filters)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False,
                 lr_mult=1.0):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels,
            lr_mult=lr_mult)
        if use_se:
            self.se = SEModule(num_channels, lr_mult=lr_mult)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1,
            lr_mult=lr_mult)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x




class PPLCNet(nn.Module):
    def __init__(self,
                 stages_pattern,
                 scale=1.0,
                 class_num=1000,
                 dropout_prob=0.2,
                 class_expand=1280,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 stride_list=[2, 2, 2, 2, 2],
                 use_last_conv=True,
                 return_patterns=None,
                 return_stages=None,
                 **kwargs):
        super().__init__()
        self.scale = scale
        self.class_expand = class_expand
        self.lr_mult_list = lr_mult_list
        self.use_last_conv = use_last_conv
        self.stride_list = stride_list
        self.net_config = NET_CONFIG
        if isinstance(self.lr_mult_list, str):
            self.lr_mult_list = eval(self.lr_mult_list)

        assert isinstance(self.lr_mult_list, (
            list, tuple
        )), "lr_mult_list should be in (list, tuple) but got {}".format(
            type(self.lr_mult_list))
        assert len(self.lr_mult_list
                   ) == 6, "lr_mult_list length should be 6 but got {}".format(
                       len(self.lr_mult_list))

        assert isinstance(self.stride_list, (
            list, tuple
        )), "stride_list should be in (list, tuple) but got {}".format(
            type(self.stride_list))
        assert len(self.stride_list
                   ) == 5, "stride_list length should be 5 but got {}".format(
                       len(self.stride_list))

        for i, stride in enumerate(stride_list[1:]):
            self.net_config["blocks{}".format(i + 3)][0][3] = stride
        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=stride_list[0],
            lr_mult=self.lr_mult_list[0])

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[1])
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[2])
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[3])
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[4])
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[5])
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks6"])
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.use_last_conv:
            self.last_conv = nn.Conv2d(
                in_channels=make_divisible(self.net_config["blocks6"][-1][2] *
                                           scale),
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False)
            self.hardswish = nn.Hardswish()
            self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        else:
            self.last_conv = None
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.fc = Linear(
            self.class_expand if self.use_last_conv else
            make_divisible(self.net_config["blocks6"][-1][2]), class_num)

        super().init_res(
            stages_pattern,
            return_patterns=return_patterns,
            return_stages=return_stages)

    def forward(self, x):
        x = self.conv1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        x = self.avg_pool(x)
        if self.last_conv is not None:
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x





