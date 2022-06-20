from __future__ import absolute_import, division, print_function

import warnings
import torch
import torch.nn as nn
import numpy as np
from ..module.activation import act_layers


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        activation="ReLU",
        deploy=False
    ):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.deploy = deploy

        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.conv_aux1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, (1, 3), stride, (0, 1), groups=groups, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.conv_aux2 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, (3, 1), stride, (1, 0), groups=groups, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.act_layers = act_layers(activation)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x0 = self.conv(x)
            x1 = self.conv_aux1(x)
            x2 = self.conv_aux2(x)

            x = torch.add(x0, x1)
            x = torch.add(x, x2)

        return self.act_layers(x)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv)
        kernel1x3, bias1x3 = self._fuse_bn_tensor(self.conv_aux1)
        kernel3x1, bias3x1 = self._fuse_bn_tensor(self.conv_aux2)
        return (
            kernel3x3 + self._pad_1x3_to_3x3_tensor(kernel1x3) + self._pad_3x1_to_3x3_tensor(kernel3x1),
            bias3x3 + bias1x3 + bias3x1,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _pad_1x3_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [0, 0, 1, 1])

    def _pad_3x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 0, 0])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def rep_mobilenet_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, activation="ReLU", deploy=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU(inp, hidden_dim, kernel_size=1, activation=activation)
            )
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation=activation,
                    deploy=deploy
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetONE(nn.Module):
    def __init__(
        self,
        width_mult=1.0,
        out_stages=(1, 2, 4, 6),
        last_channel=320,
        activation="ReLU",
        act=None,
        deploy=False
    ):
        super(MobileNetONE, self).__init__()
        # TODO: support load torchvison pretrained weight
        assert set(out_stages).issubset(i for i in range(7))
        self.width_mult = width_mult
        self.out_stages = out_stages
        input_channel = 32
        self.last_channel = last_channel
        self.activation = activation
        self.deploy = deploy
        if act is not None:
            warnings.warn(
                "Warning! act argument has been deprecated, " "use activation instead!"
            )
            self.activation = act
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        self.input_channel = int(input_channel * width_mult)
        self.first_layer = ConvBNReLU(
            3, self.input_channel, stride=2, activation=self.activation
        )
        # building inverted residual blocks
        for i in range(7):
            name = "stage{}".format(i)
            setattr(self, name, self.build_mobilenet_stage(stage_num=i))

        self._initialize_weights()

    def build_mobilenet_stage(self, stage_num):
        stage = []
        t, c, n, s = self.interverted_residual_setting[stage_num]
        output_channel = int(c * self.width_mult)
        for i in range(n):
            if i == 0:
                stage.append(
                    InvertedResidual(
                        self.input_channel,
                        output_channel,
                        s,
                        expand_ratio=t,
                        activation=self.activation,
                        deploy=self.deploy
                    )
                )
            else:
                stage.append(
                    InvertedResidual(
                        self.input_channel,
                        output_channel,
                        1,
                        expand_ratio=t,
                        activation=self.activation,
                        deploy=self.deploy
                    )
                )
            self.input_channel = output_channel
        if stage_num == 6:
            last_layer = ConvBNReLU(
                self.input_channel,
                self.last_channel,
                kernel_size=1,
                activation=self.activation,
            )
            stage.append(last_layer)
        stage = nn.Sequential(*stage)
        return stage

    def forward(self, x):
        x = self.first_layer(x)
        output = []
        for i in range(0, 7):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            # print(x.shape)
            if i in self.out_stages:
                output.append(x)

        return tuple(output)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
