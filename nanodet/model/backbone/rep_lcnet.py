from __future__ import absolute_import, division, print_function

import torch.nn as nn

from torch.nn import Dropout

from ..module.conv import RepConvBNLayer, ConvBNLayer, RepConvBNLayer5x5
from ..module.se import SEModule

MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}


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


class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 stride,
                 kernel_size,
                 use_se=False,
                 lr_mult=1.0,
                 deploy=False
                 ):
        super().__init__()
        self.use_se = use_se
        if kernel_size == 3:
            self.dw_conv = RepConvBNLayer(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=input_channels,
                deploy=deploy
            )
        else:
            self.dw_conv = RepConvBNLayer5x5(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=input_channels,
                deploy=deploy
            )
        if use_se:
            self.se = SEModule(input_channels)
        self.pw_conv = ConvBNLayer(
            in_channels=input_channels,
            kernel_size=1,
            out_channels=output_channels,
            stride=1,
            )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class RepLCNet(nn.Module):
    def __init__(self,
                 scale=1.0,
                 dropout_prob=0.2,
                 class_expand=1280,
                 use_last_conv=False,
                 deploy=False,
                 **kwargs
                 ):

        super().__init__()
        self.scale = scale
        self.class_expand = class_expand
        self.use_last_conv = use_last_conv
        self.net_config = NET_CONFIG
        self.deploy = deploy

        self.conv1 = RepConvBNLayer(
            in_channels=3,
            kernel_size=3,
            out_channels=make_divisible(16 * scale),
            stride=2,
            deploy=deploy
        )

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                input_channels=make_divisible(in_c * scale),
                output_channels=make_divisible(out_c * scale),
                kernel_size=k,
                stride=s,
                use_se=se,
                deploy=deploy)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                input_channels=make_divisible(in_c * scale),
                output_channels=make_divisible(out_c * scale),
                kernel_size=k,
                stride=s,
                use_se=se,
                deploy=deploy)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                input_channels=make_divisible(in_c * scale),
                output_channels=make_divisible(out_c * scale),
                kernel_size=k,
                stride=s,
                use_se=se,
                deploy=deploy)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                input_channels=make_divisible(in_c * scale),
                output_channels=make_divisible(out_c * scale),
                kernel_size=k,
                stride=s,
                use_se=se,
                deploy=deploy)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                input_channels=make_divisible(in_c * scale),
                output_channels=make_divisible(out_c * scale),
                kernel_size=k,
                stride=s,
                use_se=se,
                deploy=deploy)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks6"])
        ])

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
            self.dropout = Dropout(p=dropout_prob)
        else:
            self.last_conv = None

    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        output.append(x)
        x = self.blocks5(x)
        output.append(x)
        x = self.blocks6(x)
        if self.last_conv is not None:
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        output.append(x)
        return tuple(output)
