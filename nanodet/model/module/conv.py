"""
ConvModule refers from MMDetection
RepVGGConvModule refers from RepVGG: Making VGG-style ConvNets Great Again
"""
import warnings

import numpy as np
import torch
import torch.nn as nn

from .activation import act_layers
from .init_weights import constant_init, kaiming_init
from .norm import build_norm_layer
from torch.nn import BatchNorm2d, Conv2d
from ..sparse_ops.sparse_ops import SparseConv


class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str): activation layer, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        activation="ReLU",
        inplace=True,
        order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert activation is None or isinstance(activation, str)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.conv = nn.Conv2d(  #
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if self.activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "relu"
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, norm=True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and self.activation:
                x = self.act(x)
        return x


class DepthwiseConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias="auto",
        norm_cfg=dict(type="BN"),
        activation="ReLU",
        inplace=True,
        order=("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act"),
    ):
        super(DepthwiseConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 6
        assert set(order) == {
            "depthwise",
            "dwnorm",
            "act",
            "pointwise",
            "pwnorm",
            "act",
        }

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.depthwise.in_channels
        self.out_channels = self.pointwise.out_channels
        self.kernel_size = self.depthwise.kernel_size
        self.stride = self.depthwise.stride
        self.padding = self.depthwise.padding
        self.dilation = self.depthwise.dilation
        self.transposed = self.depthwise.transposed
        self.output_padding = self.depthwise.output_padding

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            _, self.dwnorm = build_norm_layer(norm_cfg, in_channels)
            _, self.pwnorm = build_norm_layer(norm_cfg, out_channels)

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        if self.activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "relu"
        kaiming_init(self.depthwise, nonlinearity=nonlinearity)
        kaiming_init(self.pointwise, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.dwnorm, 1, bias=0)
            constant_init(self.pwnorm, 1, bias=0)

    def forward(self, x, norm=True):
        for layer_name in self.order:
            if layer_name != "act":
                layer = self.__getattr__(layer_name)
                x = layer(x)
            elif layer_name == "act" and self.activation:
                x = self.act(x)
        return x


class RepVGGConvModule(nn.Module):
    """
    RepVGG Conv Block from paper RepVGG: Making VGG-style ConvNets Great Again
    https://arxiv.org/abs/2101.03697
    https://github.com/DingXiaoH/RepVGG
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        activation="ReLU",
        padding_mode="zeros",
        deploy=False,
        **kwargs
    ):
        super(RepVGGConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation

        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )

        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=padding_11,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )
            print("RepVGG Block, identity = ", self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you
    #   do to the other models.  May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

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

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )


class RepConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 if_act=True,
                 deploy=False):
        super().__init__()
        self.if_act = if_act
        self.deploy = deploy
        assert kernel_size == 3
        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=True,
            )
        else:
            self.conv = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    groups=groups,
                    bias=False),
                BatchNorm2d(out_channels)
            )

            self.conv_aux1_3 = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 3),
                    stride=stride,
                    padding=(0, 1),
                    groups=groups,
                    bias=False
                ),
                BatchNorm2d(out_channels)
            )

            self.conv_aux3_1 = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 1),
                    stride=stride,
                    padding=(1, 0),
                    groups=groups,
                    bias=False
                ),
                BatchNorm2d(out_channels)
            )

            self.conv_aux3_3 = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=(1, 1),
                    groups=groups,
                    bias=False
                ),
                BatchNorm2d(out_channels)
            )

            nn.init.kaiming_normal_(self.conv[0].weight.data)
        if if_act:
            self.act_layers = nn.Hardswish()

    def forward(self, x):
        if self.deploy:
            x = self.rbr_reparam(x)
        else:
            x0 = self.conv(x)
            x1_3 = self.conv_aux1_3(x)
            x3_1 = self.conv_aux3_1(x)
            x3_3 = self.conv_aux3_3(x)

            x = torch.add(x0, x1_3)
            x = torch.add(x, x3_1)
            x = torch.add(x, x3_3)

        if self.if_act:
            return self.act_layers(x)
        else:
            return x

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv)
        kernel1x3, bias1x3 = self._fuse_bn_tensor(self.conv_aux1_3)
        kernel3x1, bias3x1 = self._fuse_bn_tensor(self.conv_aux3_1)
        kernel3x3_aux, bias3x3_aux = self._fuse_bn_tensor(self.conv_aux3_3)
        return (
            kernel3x3 + self._pad_1x3_to_3x3_tensor(kernel1x3) + self._pad_3x1_to_3x3_tensor(kernel3x1) + kernel3x3_aux,
            bias3x3 + bias1x3 + bias3x1 + bias3x3_aux,
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

    def rep_net_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )


class RepConvBNLayer5x5(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 if_act=True,
                 deploy=False):
        super().__init__()
        self.if_act = if_act
        self.deploy = deploy
        assert kernel_size == 5
        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=True,
            )
        else:
            self.conv = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    groups=groups,
                    bias=False),
                BatchNorm2d(out_channels)
            )

            self.conv_aux1_5 = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 5),
                    stride=stride,
                    padding=(0, 2),
                    groups=groups,
                    bias=False
                ),
                BatchNorm2d(out_channels)
            )

            self.conv_aux5_1 = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(5, 1),
                    stride=stride,
                    padding=(2, 0),
                    groups=groups,
                    bias=False
                ),
                BatchNorm2d(out_channels)
            )

            self.conv_aux3_3 = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=(1, 1),
                    groups=groups,
                    bias=False
                ),
                BatchNorm2d(out_channels)
            )

            self.conv_aux3_5 = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 5),
                    stride=stride,
                    padding=(1, 2),
                    groups=groups,
                    bias=False
                ),
                BatchNorm2d(out_channels)
            )

            self.conv_aux5_3 = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(5, 3),
                    stride=stride,
                    padding=(2, 1),
                    groups=groups,
                    bias=False
                ),
                BatchNorm2d(out_channels)
            )

            self.conv_aux5_5 = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(5, 5),
                    stride=stride,
                    padding=(2, 2),
                    groups=groups,
                    bias=False
                ),
                BatchNorm2d(out_channels)
            )

            nn.init.kaiming_normal_(self.conv[0].weight.data)
        if if_act:
            self.act_layers = nn.Hardswish()

    def forward(self, x):
        if self.deploy:
            x = self.rbr_reparam(x)
        else:
            x0 = self.conv(x)
            x1_5 = self.conv_aux1_5(x)
            x5_1 = self.conv_aux5_1(x)
            x3_3 = self.conv_aux3_3(x)
            x5_3 = self.conv_aux5_3(x)
            x3_5 = self.conv_aux3_5(x)
            x5_5 = self.conv_aux5_5(x)

            x = torch.add(x0, x1_5)
            x = torch.add(x, x5_1)
            x = torch.add(x, x3_3)
            x = torch.add(x, x3_5)
            x = torch.add(x, x5_3)
            x = torch.add(x, x5_5)

        if self.if_act:
            return self.act_layers(x)
        else:
            return x

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv)
        kernel1x3, bias1x3 = self._fuse_bn_tensor(self.conv_aux1_3)
        kernel3x1, bias3x1 = self._fuse_bn_tensor(self.conv_aux3_1)
        kernel3x3_aux, bias3x3_aux = self._fuse_bn_tensor(self.conv_aux3_3)
        return (
            kernel3x3 + self._pad_1x3_to_3x3_tensor(kernel1x3) + self._pad_3x1_to_3x3_tensor(kernel3x1) + kernel3x3_aux,
            bias3x3 + bias1x3 + bias3x1 + bias3x3_aux,
        )

    def _pad_3x3_to_5x5_tensor(self, kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            return nn.functional.pad(kernel3x3, [1, 1, 1, 1])

    def _pad_1x5_to_5x5_tensor(self, kernel1x5):
        if kernel1x5 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x5, [0, 0, 2, 2])

    def _pad_5x1_to_5x5_tensor(self, kernel5x1):
        if kernel5x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel5x1, [2, 2, 0, 0])

    def _pad_3x5_to_5x5_tensor(self, kernel3x5):
        if kernel3x5 is None:
            return 0
        else:
            return nn.functional.pad(kernel3x5, [0, 0, 1, 1])

    def _pad_5x3_to_5x5_tensor(self, kernel5x3):
        if kernel5x3 is None:
            return 0
        else:
            return nn.functional.pad(kernel5x3, [1, 1, 0, 0])

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
                    (self.in_channels, input_dim, 5, 5), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 2, 2] = 1
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

    def rep_net_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 if_act=True,
                 if_sparse=False):
        super().__init__()

        if if_sparse:
            self.conv = SparseConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False)

        else:
            self.conv = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False)
        self.bn = BatchNorm2d(out_channels)

        nn.init.kaiming_normal_(self.conv.weight.data)
        self.if_act = if_act
        self.hardswish = nn.Hardswish()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.hardswish(x)
        return x
