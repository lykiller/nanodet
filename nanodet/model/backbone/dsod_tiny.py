import torch
import torch.nn as nn
import numpy as np
from itertools import repeat
import torch.nn.functional as F


def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


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


def blur_depth_wise_conv(i, o, stride=(2, 2), padding=(0, 0), bias=False):

    a = [1, 2, 1]
    k = np.array(a, dtype=np.float32)
    k = k[:, None] * k[None, :]
    k = k / np.sum(k)
    k = np.tile(k[np.newaxis, np.newaxis, :, :], (i, 1, 1, 1))
    dw_conv = nn.Conv2d(i, o, 3, stride, padding, bias=bias, groups=i)
    dw_conv.weight.data = torch.Tensor(k)
    return dw_conv


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs # 测试阶段
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    # 以样本为单位生成模块是否被drop的01向量
    binary_tensor = torch.floor(random_tensor)
    # 因为越往后越容易被drop，所以没有被drop的值就要通过除keep_prob来放大
    output = inputs / keep_prob * binary_tensor
    return output


class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """

    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])  # 默认沿着中间所有的shape

        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


class SeBlock(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DropBlock(nn.Module):
    def __init__(self, block_size: int, p: float = 0.5):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: torch.Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class TransDownBlock(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            offset=0,
            depth_conv=False,
            kernel_multi=None,
            se=False,
    ):
        super(TransDownBlock, self).__init__()
        self.se = se
        self.depth_conv = depth_conv
        self.output_channels = output_channels
        self.offset = offset
        if se:
            self.se_block = SeBlock(input_channels)
        if depth_conv:
            self.depth_conv = nn.ModuleList()
            if isinstance(kernel_multi, list):
                for k in kernel_multi:
                    if k == 0:
                        self.depth_conv.append(nn.MaxPool2d(3, 2))
                    else:
                        self.depth_conv.append(depthwise_conv(input_channels, input_channels, k, 2))
            else:
                self.depth_conv.append(depthwise_conv(input_channels, input_channels, 3, 2))

        if output_channels:
            self.conv = depthwise_conv(input_channels, output_channels, 1)

    def forward(self, inputs):
        x = inputs
        if self.offset:
            _, x = torch.split(x, [self.offset, -self.offset], dim=1)

        if self.se:
            x = self.se_block(x)

        if self.depth_conv:
            xlist = []
            for depth_conv in self.depth_conv:
                xone = depth_conv(x)
                xlist.append(xone)
            x = torch.cat(xlist, dim=1)
        # print(x.shape)
        if self.output_channels:
            # print(self.output_channels)
            x = self.conv(x)
        # print(x.shape)
        return x


class DenseBlockB(nn.Module):
    def __init__(
            self,
            input_channels,
            growth_rate,
            use_conv=False,
            kernel_multi=1,
            drop_connect_rate=0,
            res2net_style=False,
            b_concat=True,
            is_training=None,
            return_value=False,
            short_cut=None,
            concat_short_cut=None
    ):
        super(DenseBlockB, self).__init__()
        self.res2net_style = res2net_style
        self.drop_connect_rate = drop_connect_rate
        self.kernel_multi = kernel_multi
        self.b_concat = b_concat
        self.short_cut = short_cut
        self.concat_short_cut = concat_short_cut

        if drop_connect_rate > 0:
            self.drop_block = DropBlock(block_size=5, p=drop_connect_rate)
        self.conv0 = nn.Conv2d(input_channels, growth_rate, 1, 1)
        self.multi_kernel_conv = nn.ModuleList()

        if isinstance(kernel_multi, tuple):
            for i, k in enumerate(kernel_multi):
                if k != 0:
                    self.multi_kernel_conv.append(self.conv2d(growth_rate, growth_rate, k, 1, use_conv))
        else:
            for i in range(kernel_multi):
                self.multi_kernel_conv.append(self.conv2d(growth_rate, growth_rate, 3, 1, use_conv))

            self.kernel_multi = range(kernel_multi)

        self.conv1 = nn.Conv2d(growth_rate*len(kernel_multi), growth_rate, 1, 1)

    @staticmethod
    def conv2d(input_channels, growth_rate, kernel, stride, use_conv):
        if use_conv:
            return nn.Conv2d(input_channels, growth_rate, 3, stride, padding=1)
        else:
            return nn.Conv2d(input_channels, input_channels, kernel, stride, padding=int((kernel-1)/2), groups=input_channels)

    def forward(self, x, concat_shortcut=None, short_cut=None):
        residual = x
        if self.drop_connect_rate > 0:
            residual = self.drop_block(x)
        residual = self.conv0(residual)
        residual_cpy = residual
        xlist = []
        if self.res2net_style:
            for i, k in enumerate(self.kernel_multi):
                if k == 0:
                    xone = residual_cpy
                else:
                    xone = residual if len(xlist) == 0 else torch.add(residual, xlist[-1])
                    xone = self.multi_kernel_conv[i](xone)
                xlist.append(xone)
            if concat_shortcut is not None:
                xlist.append(concat_shortcut)

        else:
            for i, k in enumerate(self.kernel_multi):
                if k == 0:
                    xone = residual
                else:
                    xone = self.multi_kernel_conv[i-1](residual)
                xlist.append(xone)

        if concat_shortcut is not None:
            xlist.append(concat_shortcut)
        xnew = xlist[0] if len(xlist) == 1 else torch.cat(xlist, dim=1)

        if short_cut is not None:
            xnew = torch.add(xnew, short_cut)

        return torch.cat([x, self.conv1(xnew)], dim=1)


class TransitionLayerMine(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            down_sample=True,
            drop_connect_rate=0,
            pool_k=2,
            is_conv_pool=False,
            is_avg_pool=False,
            is_training=True

    ):
        super(TransitionLayerMine, self).__init__()
        self.down_sample = down_sample
        self.is_avg_pool = is_avg_pool
        self.conv2d_0 = nn.Conv2d(input_channels, output_channels, 1, 1)
        self.drop_connect_rate = drop_connect_rate
        self.is_training = is_training

        if down_sample:
            self.down_sample_layer = blur_depth_wise_conv(output_channels, output_channels, padding=1)
        elif is_avg_pool:
            self.down_sample_layer = nn.AvgPool2d(pool_k, 2)
        else:
            self.down_sample_layer = nn.MaxPool2d(pool_k, 2)

    def forward(self, inputs, concat_shortcut=None, short_cut=None):
        x = inputs
        if concat_shortcut is not None:
            x = torch.cat([x, concat_shortcut], dim=1)

        if self.drop_connect_rate > 0:
            x = drop_connect(x, self.drop_connect_rate, self.is_training)

        x = self.conv2d_0(x)
        if short_cut is not None:
            x = torch.add(x, short_cut)
        value = x
        x = self.down_sample_layer(x)
        return x


class ShortCut(nn.Module):
    def __init__(
            self,
            input_channels,
            mid_channels,
            output_channels
    ):
        super(ShortCut, self).__init__()
        self.max_pool2d_0 = nn.MaxPool2d(2, 2)
        self.conv2d_0 = nn.Conv2d(input_channels, mid_channels, (1, 1), 2, 0)
        self.max_pool2d_1 = nn.MaxPool2d(2, 2)
        self.max_pool2d_2 = nn.MaxPool2d(2, 2)

        self.conv2d_1 = nn.Conv2d(3 * mid_channels, mid_channels, 1, 2)

        self.dw_conv2d_0 = depthwise_conv(mid_channels, mid_channels, 3, 1)
        self.dw_conv2d_1 = depthwise_conv(mid_channels, mid_channels, 3, 1)

        self.conv2d_2 = nn.Conv2d(input_channels, output_channels, 1, 1)
        self.up_sample = nn.Upsample()

    def forward(self, x):
        x = self.max_pool2d_0(x)
        x = self.conv2d_0(x)
        x2 = self.max_pool2d_1(x)
        x3 = self.max_pool2d_2(x2)
        x = torch.cat([x, x2, x3], dim=-1)
        x = self.conv2d_1(x)
        x2 = self.dw_conv2d_0(x)
        x = self.dw_conv2d_1(torch.add(x, x2))
        x = self.conv2d_2(x)
        x = self.up_sample(x)
        return x


class StemBlockMine(nn.Module):
    def __init__(
            self,
            init_channels,
            out_channels,
    ):
        super(StemBlockMine, self).__init__()
        self.conv2d_0 = nn.Conv2d(3, init_channels, 3, 2, padding=1)
        self.dw_conv = depthwise_conv(init_channels, init_channels, 3, 1, padding=1)
        self.conv2d_1 = nn.Conv2d(init_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv2d_0(x)

        x = self.dw_conv(x)

        x = self.conv2d_1(x)
        return x


class Stage2(nn.Module):
    def __init__(
            self,
            input_channels,
            growth_rate,
            repeats=3,
            kernel_multi=1,
            is_training=True
    ):
        super(Stage2, self).__init__()
        self.con2d_0 = nn.Conv2d(input_channels, growth_rate, 1, 1)
        self.con2d_1 = nn.Conv2d(input_channels, growth_rate, 1, 1)
        self.repeats = repeats
        for i in range(repeats-1):
            name = 'stage_{}'.format(i)
            input_channels = growth_rate * (i + 1)
            seq = nn.Sequential(
                depthwise_conv(input_channels, input_channels, 3, 1, padding=1),
                nn.Conv2d(input_channels, growth_rate, 1, 1)
            )
            setattr(self, name, seq)

    def forward(self, x):

        net1 = self.con2d_0(x)

        residual = self.con2d_1(x)

        re_list = []
        for i in range(self.repeats - 1):
            x = residual if len(re_list) == 0 else torch.cat([residual, re_list[-1]], dim=1)
            stage_name = 'stage_{}'.format(i)
            x = getattr(self, stage_name)(x)
            re_list.append(x)
        re_list.append(net1)
        output = torch.cat(re_list, dim=1)

        return output


class DsodTiny(nn.Module):
    def __init__(
            self,
            model_size="1.0x",
            out_stages=(2, 3, 4),
            repeats=(5, 5, 4),
            kernel_multi=(0, 3, 5),
            is_training=True,
            activation='LeakyReLU'

    ):

        super(DsodTiny, self).__init__()
        if model_size == "0.5x":
            growth_rate = 8
        elif model_size == "1.0x":
            growth_rate = 16
        elif model_size == "1.5x":
            growth_rate = 24
        elif model_size == "2.0x":
            growth_rate = 32
        else:
            raise NotImplementedError

        self.stem_block = StemBlockMine(growth_rate, growth_rate)
        self.trans_down_block1 = TransDownBlock(growth_rate, growth_rate * 2)
        self.stage2 = Stage2(growth_rate * 2, growth_rate, 3, 1)
        # print(input_channels, output_channels)
        self.trans_down_block2 = TransDownBlock(growth_rate * 3, growth_rate * 3)

        self.max_pool2d_1 = nn.MaxPool2d(2, 2)
        self.max_pool2d_2 = nn.MaxPool2d(2, 2)
        self.max_pool2d_3 = nn.MaxPool2d(2, 2)

        stage_names = ["stage{}b".format(i) for i in [2, 3, 4]]
        sp_block_names = ["sp_block{}b".format(i) for i in [2, 3, 4]]
        init_input_channels = [48, 64, 160]
        sp_output_channels = [64, 160, 176]

        for j, stage_name, sp_block_name, r, input_channels, output_channels in zip(range(3), stage_names, sp_block_names, repeats, init_input_channels, sp_output_channels):
            seq = []
            for i in range(r):
                seq.append(DenseBlockB(input_channels + i*growth_rate, growth_rate, kernel_multi=kernel_multi, is_training=is_training))
            setattr(self, stage_name, nn.Sequential(*seq))

            input_channels = sum(init_input_channels[:j+1]) + r * growth_rate
            sp_block = TransitionLayerMine(input_channels, output_channels)
            setattr(self, sp_block_name, sp_block)

    def forward(self, x):
        output = []
        stem_block_output = self.stem_block(x)
        net = stem_block_output
        net = self.trans_down_block1(net)
        net = self.stage2(net)
        net = self.trans_down_block2(net)

        c2_3 = net
        c2_4 = self.max_pool2d_1(c2_3)
        c2_5 = self.max_pool2d_2(c2_4)

        net = self.stage2b(net)
        net = self.sp_block2b(net)
        output.append(net)
        c3_4 = net
        c3_5 = self.max_pool2d_3(c3_4)

        net = self.stage3b(net)
        net = torch.cat([c2_4, net], dim=1)
        net = self.sp_block3b(net)
        output.append(net)
        c4_5 = net
        net = self.stage4b(net)
        net = torch.cat([c2_5, c3_5, net], dim=1)
        net = self.sp_block4b(net)
        output.append(net)
        return output


class ShortCutInfoBuild(nn.Module):
    def __init__(
            self,
            input_channels,
            mid_channels,
            output_channels
    ):
        super(ShortCutInfoBuild, self).__init__()
        self.max_pool2d_0 = nn.MaxPool2d(2, 2)
        self.conv2d_0 = nn.Conv2d(input_channels, output_channels, 1, 1)
        self.max_pool2d_1 = nn.MaxPool2d(3, 1)
        self.max_pool2d_2 = nn.MaxPool2d(3, 1)

        cat_input_channels = 3*mid_channels
        self.conv2d_1 = nn.Conv2d(cat_input_channels, mid_channels, 1, 1)
        self.dw_conv2d_0 = depthwise_conv(mid_channels, mid_channels, 3, 1)
        self.dw_conv2d_1 = depthwise_conv(mid_channels, mid_channels, 3, 1)

        self.conv2d_2 = nn.Conv2d(mid_channels, output_channels, 1, 1)


    def forward(self, x):
        shape = x.shape
        x = self.max_pool2d_0(x)
        x = self.conv2d_0(x)
        x2 = self.max_pool2d_1(x)
        x3 = self.max_pool2d_2(x)
        x = torch.cat([x, x2, x3], dim=1)

        x = self.conv2d_1(x)
        x2 = self.dw_conv2d_0(x)
        x = self.dw_conv2d_1(torch.add(x, x2))

        x = self.conv2d_2(x)
        x = nn.Upsample(shape)
        return x


class DsodTinyV2(nn.Module):
    def __init__(
            self,
            model_size='1.0x',
            is_training=True,
            repeats=(5, 5, 4),
            out_stages=(2, 3, 4),
            grow_rates=None,
            kernel_multi=(0, 3, 5),
            sib=False,
            enable_se=False,
            res2net_style=True,
            b_concat=False,
            activation='ReLu'
    ):
        super(DsodTinyV2, self).__init__()
        if model_size == "0.5x":
            growth_rate = [8, 16, 16]
        elif model_size == "1.0x":
            growth_rate = [16, 32, 32]
        elif model_size == "1.5x":
            growth_rate = [24, 48, 48]
        elif model_size == "2.0x":
            growth_rate = [32, 64, 64]
        else:
            raise NotImplementedError

        self.sib = sib
        self.stem_block = StemBlockMine(growth_rate[0], growth_rate[0])
        self.trans_down_block1 = TransDownBlock(growth_rate[0], growth_rate[0] * 2)
        self.stage2 = Stage2(growth_rate[0] * 2, growth_rate[0], 3, 1)
        self.trans_down_block2 = TransDownBlock(growth_rate[0] * 3, growth_rate[0] * 3)

        stage_names = ["stage{}b".format(i) for i in [2, 3, 4]]
        sp_block_names = ["sp_block{}b".format(i) for i in [2, 3, 4]]
        init_input_channels = [48, 64, 128]
        sp_output_channels = [64, 128, 128]

        if sib:
            len_kernel = len(kernel_multi)
            self.short_cut3 = ShortCutInfoBuild(init_input_channels[0], growth_rate[0],
                                                growth_rate[0] * len_kernel)
            self.short_cut4 = ShortCutInfoBuild(init_input_channels[1], growth_rate[1],
                                                growth_rate[1] * len_kernel)
            self.short_cut5 = ShortCutInfoBuild(init_input_channels[2], growth_rate[2],
                                                growth_rate[2] * len_kernel)
        else:
            self.short_cut3 = None

        for j, stage_name, sp_block_name, r, input_channels, output_channels \
                in zip(range(3), stage_names, sp_block_names, repeats, init_input_channels, sp_output_channels):
            seq = []
            for i in range(r):
                seq.append(DenseBlockB(input_channels + i*growth_rate[j], growth_rate[j], kernel_multi=kernel_multi, is_training=is_training))
            setattr(self, stage_name, nn.Sequential(*seq))

            input_channels = input_channels + r * growth_rate[j]
            sp_block = TransitionLayerMine(input_channels, output_channels)
            setattr(self, sp_block_name, sp_block)

    def forward(self, x):
        output = []
        stem_block_output = self.stem_block(x)
        net = stem_block_output
        net = self.trans_down_block1(net)
        net = self.stage2(net)
        net = self.trans_down_block2(net)

        if self.sib:
            short_cut3 = self.short_cut3(net)
        else:
            short_cut3 = None
        net = self.stage2b(net, short_cut3)
        net = self.sp_block2b(net)
        output.append(net)

        if self.sib:
            short_cut4 = self.short_cut4(net)
        else:
            short_cut4 = None
        net = self.stage3b(net, short_cut4)
        net = self.sp_block3b(net)
        output.append(net)

        if self.sib:
            short_cut5 = self.short_cut5(net)
        else:
            short_cut5 = None
        net = self.stage4b(net, short_cut5)
        net = self.sp_block4b(net)
        output.append(net)
        return output


class DsodTinyMnn(nn.Module):
    def __init__(
            self,
            model_size='',
            repeats=None,
            is_training=True
    ):
        super(DsodTinyMnn, self).__init__()

        self.stem_block = StemBlockMine(growth_rate, growth_rate)
        self.trans_down_block1 = TransDownBlock(growth_rate, growth_rate * 2)
        self.stage2 = Stage2(growth_rate * 2, growth_rate, 3, 1)
        # print(input_channels, output_channels)
        self.trans_down_block2 = TransDownBlock(growth_rate * 3, growth_rate * 3)

        self.max_pool2d_1 = nn.MaxPool2d(2, 2)
        self.max_pool2d_2 = nn.MaxPool2d(2, 2)
        self.max_pool2d_3 = nn.MaxPool2d(2, 2)

        stage_names = ["stage{}b".format(i) for i in [2, 3, 4]]
        sp_block_names = ["sp_block{}b".format(i) for i in [2, 3, 4]]
        init_input_channels = [48, 64, 160]
        sp_output_channels = [64, 160, 176]

        for j, stage_name, sp_block_name, r, input_channels, output_channels in zip(range(3), stage_names, sp_block_names, repeats, init_input_channels, sp_output_channels):
            seq = []
            for i in range(r):
                seq.append(DenseBlockB(input_channels + i*growth_rate, growth_rate, kernel_multi=kernel_multi, is_training=is_training))
            setattr(self, stage_name, nn.Sequential(*seq))

            input_channels = sum(init_input_channels[:j+1]) + r * growth_rate
            sp_block = TransitionLayerMine(input_channels, output_channels)
            setattr(self, sp_block_name, sp_block)

    def forward(self, x):
        output = []
        stem_block_output = self.stem_block(x)
        net = stem_block_output
        net = self.trans_down_block1(net)
        net = self.stage2(net)
        net = self.trans_down_block2(net)

        c2_3 = net
        c2_4 = self.max_pool2d_1(c2_3)
        c2_5 = self.max_pool2d_2(c2_4)

        net = self.stage2b(net)
        net = self.sp_block2b(net)
        output.append(net)
        c3_4 = net
        c3_5 = self.max_pool2d_3(c3_4)

        net = self.stage3b(net)
        net = torch.cat([c2_4, net], dim=1)
        net = self.sp_block3b(net)
        output.append(net)
        c4_5 = net
        net = self.stage4b(net)
        net = torch.cat([c2_5, c3_5, net], dim=1)
        net = self.sp_block4b(net)
        output.append(net)
        return output






