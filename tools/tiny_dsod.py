import collections
import tensorflow as tf
import time
from datetime import datetime

slim = tf.contrib.slim
drop_rate=0.1#1.0/16
efficient_dense_enable = False


def channel_shuffle(x, groups):
    """

    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    groups: int
        number of groups per channel


    Returns
    -------
        channel shuffled output tensor


    Examples
    --------
    Example for a 1D Array with 3 groups

    >>> d = np.array([0,1,2,3,4,5,6,7,8])
    >>> x = np.reshape(d, (3,3))
    >>> x = np.transpose(x, [1,0])
    >>> x = np.reshape(x, (9,))
    '[0 1 2 3 4 5 6 7 8] --> [0 3 6 1 4 7 2 5 8]'


    """
    if (groups != 1):
        height, width, in_channels = x.shape.as_list()[1:]
        channels_per_group = in_channels // groups

        # x = K.reshape(x, [-1, height, width, groups, channels_per_group])
        # x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
        # x = K.reshape(x, [-1, height, width, in_channels])

        s = tf.shape(x)
        x = tf.reshape(x, [-1, groups, channels_per_group])
        x = tf.transpose(x, (0, 2, 1))  # transpose
        x = tf.reshape(x, s)
    return x


def se_unit_(features, ratio=4, up_align=True):
    sz = features.shape.as_list()
    # features_group = tf.split(features_group, group, axis=-1)
    assert sz[2]>3 and sz[3]>3
    # x = slim.avg_pool2d(features, (sz[-3], sz[-2]), stride=(1, 1), padding='VALID')
    x = tf.reduce_mean(features, axis=[1,2], keep_dims=True)

    mid_channel = (sz[-1]//ratio + 15) // 16 * 16 if up_align else sz[-1]//ratio

    x = slim.conv2d(x, mid_channel, kernel_size=1, activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm)
    x = slim.conv2d(x, sz[-1], kernel_size=1, activation_fn=None, normalizer_fn=None)
    x = tf.sigmoid(x)

    features = features * x
    return features



def blur_depthwise_conv2d(x, stride=(2,2),normalizer_fn=None, trainable=False):
    import numpy as np
    a = [1, 2, 1]
    k = np.array(a, dtype=np.float32)
    k = k[:, None] * k[None, :]
    k = k / np.sum(k)
    k = np.tile(k[:,:,np.newaxis, np.newaxis], (1, 1, x.get_shape().as_list()[-1], 1))
    # k = tf.tile(k[:, :, None, None], (1, 1, x.get_shape().as_list()[-1], 1))
    x = slim.separable_conv2d(x, None, [3, 3], stride=stride,activation_fn=tf.nn.relu6,normalizer_fn=normalizer_fn,trainable=trainable,
                              padding='SAME', biases_initializer=tf.zeros_initializer(),
                              weights_initializer = tf.constant_initializer(k))
    return x

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing an Xception block.
  Its parts are:
    scope: The scope of the block.
    unit_fn: The Xception unit function which takes as input a tensor and
      returns another tensor with the output of the Xception unit.
    args: A list of length equal to the number of units in the block. The list
      contains one dictionary for each unit in the block to serve as argument to
      unit_fn.
  """



@slim.add_arg_scope
def stem_block_mine(inputs,
               num_init_channel=16,
               num_out_channel=32,
               outputs_collections=None,
               scope=None):
    with tf.variable_scope(scope, 'stem_block_mine', [inputs]) as sc:
        net = slim.conv2d(inputs, num_init_channel, [3, 3], stride=2, scope='conv0')
        # net = slim.conv2d(net, num_init_channel, [1, 1], stride=1, scope='conv1')

        net = slim.separable_conv2d(net, None, [3, 3], stride=1, scope="Dwconv0")

        net = slim.conv2d(net, num_out_channel, [1, 1], stride=1, scope='conv2')

        # net = slim.separable_conv2d(net, None, [3, 3], stride=2, depth_multiplier=3, scope="Dwconv1")

        # net = channel_shuffle(net, 4)

        # # output = slim.avg_pool2d(net, [2, 2], stride=2, scope='maxpool0')
        # net = slim.max_pool2d(net, [3, 3], 1, scope='pool', padding='SAME')
        # output = blur_depthwise_conv2d(net)
        output=net

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


@slim.add_arg_scope
def drop_connect(inputs, is_training, drop_connect_rate):
  """Apply drop connect."""
  if not is_training or drop_connect_rate==0.:
    return inputs

  # Compute keep_prob
  # TODO(tanmingxing): add support for training progress.
  keep_prob = 1.0 - drop_connect_rate

  # Compute drop_connect tensor
  batch_size = tf.shape(inputs)[0]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  output = tf.div(inputs, keep_prob) * binary_tensor
  return output

@slim.add_arg_scope
def drop_spatial(inputs, is_training, drop_connect_rate):
  """Apply drop connect."""
  if not is_training or drop_connect_rate==0.:
    return inputs

  # Compute keep_prob
  # TODO(tanmingxing): add support for training progress.
  keep_prob = 1.0 - drop_connect_rate

  # Compute drop_connect tensor
  channels = tf.shape(inputs)[-1]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform([1, 1, 1, channels], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  output = tf.div(inputs, keep_prob) * binary_tensor
  return output

@slim.add_arg_scope
def drop_block(inputs, is_training, drop_connect_rate, scale=True, block_size=3):
  """Apply drop connect."""
  if not is_training or drop_connect_rate==0.:
    return inputs

  # Compute keep_prob
  # TODO(tanmingxing): add support for training progress.
  keep_prob = 1.0 - drop_connect_rate

  batch_size, h, w, c = inputs.shape.as_list()
  gamma = (1. - keep_prob) * (w * h) / (block_size ** 2) / ((w - block_size + 1) * (h - block_size + 1))
  sampling_mask_shape = tf.stack([1, h - block_size + 1, w - block_size + 1, c])
  noise_dist = tf.distributions.Bernoulli(probs=gamma)
  mask = noise_dist.sample(sampling_mask_shape)
  br = (block_size - 1) // 2
  tl = (block_size - 1) - br
  pad_shape = [[0, 0], [tl, br], [tl, br], [0, 0]]
  mask = tf.pad(mask, pad_shape)
  mask = tf.nn.max_pool2d(mask, [1, block_size, block_size, 1], [1, 1, 1, 1], 'SAME')
  mask = tf.cast(1 - mask, tf.float32)
  output = tf.multiply(inputs, mask)
  if scale:
    # output = output * (tf.to_float(tf.size(mask)) / tf.maximum(tf.reduce_sum(mask), 1e-7))
    output = output * (tf.to_float(c) / tf.maximum(tf.reduce_sum(mask, axis=-1, keepdims=True), 1e-7))
  return output

@slim.add_arg_scope
def transition_layer_mine(inputs, num_channel=128, short_cut=None, concat_shortcut=None, is_convpool=False, is_avgpool=False, down_sample=True, se=False, drop_connect_rate=0, pool_k=2,
                          return_value=False,outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'sp_block', [inputs]) as sc:
        x = inputs
#         output_channel = inputs.get_shape().as_list()[-1]
#         x = slim.separable_conv2d(x, None, [3, 3], stride =1,scope="Dwconv")
        if concat_shortcut is not None:
            x = tf.concat([x, concat_shortcut], axis=-1, name='concat_short_cut')
        if se:
            x = se_unit_(x)
        if drop_connect_rate>0:
            x = drop_connect(x, drop_connect_rate=drop_connect_rate)
        x = slim.conv2d(x, num_channel, [1,1], stride=1, scope='conv')
        if short_cut is not None:
            x = tf.add(x, short_cut, name='add_trans_shortcut')
        x = slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name+'_value',
                                            x)
        value = x
        if down_sample:
            if is_convpool:
                # x = slim.separable_conv2d(x, None, [3, 3], stride=2, scope="Dwconvpool", padding='SAME')
                x = blur_depthwise_conv2d(x, normalizer_fn=slim.batch_norm, trainable=False)
                # x = slim.separable_conv2d(x, None, [2, 2], stride=2, activation_fn=tf.nn.relu6, normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                #               weights_initializer = tf.constant_initializer(0.25))
                # x = slim.conv2d(x, num_channel, [2, 2], stride=2, scope='conv_stride2')
            elif is_avgpool:
                x = slim.avg_pool2d(x, [pool_k, pool_k], stride=2, scope='avgpool', padding='SAME')
            else:
                x = slim.max_pool2d(x, [pool_k, pool_k], stride=2, scope='maxpool', padding='SAME')

        # x = slim.conv2d(x, num_channel, [1,1], stride=1, scope='conv')

    if return_value:
        return slim.utils.collect_named_outputs(outputs_collections,sc.name,x), value
    else:
        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            x)



@slim.add_arg_scope
def trans_down_block(inputs,
                  offset=0,
                  out_channel=None,
                  depth_conv=False,
                  kernel_multi=1,
                  se=False,
                  shuffle=False,
                  outputs_collections=None,
                  scope=None):
    with tf.variable_scope(scope, 'sp_block', [inputs]) as sc:
        x = inputs
        if offset != 0:
            l = x.shape.as_list()[-1]
            _, x = tf.split(x, [offset, l-offset], axis=-1)
        if se:
            x = se_unit_(x)
        x = slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name+'_value', x)
        if depth_conv:
            if isinstance(kernel_multi, list):
                xlist = []
                for k in kernel_multi:
                    if k==0:
                        xone = slim.max_pool2d(x, [3, 3], 2, padding='SAME')
                    else:
                        xone = slim.separable_conv2d(x, None, [k, k], stride=2, depth_multiplier=1, padding='SAME')#, normalizer_fn=None, activation_fn=None)
                    xlist.append(xone)
                x = xone if len(kernel_multi) == 1 else tf.concat(xlist, axis=-1)
                # x = slim.batch_norm(x)
                # x = tf.nn.relu6(x)
            else:
                x = slim.separable_conv2d(x, None, [3, 3], stride=2, depth_multiplier=kernel_multi)  # , normalizer_fn=None, activation_fn=None)

        if out_channel is not None:
            x = slim.conv2d(x, out_channel, [1, 1], stride=1, scope='conv')
            # x1 = blur_depthwise_conv2d(x1)
            # x1 = slim.avg_pool2d(x1, (3,3), stride=(2,2), padding='SAME')
        output = x

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)



@slim.add_arg_scope
def dense_block_b(inputs,
                 growth_rate,
                 use_conv3x3=False,
                 kernel_multi=1,
                 short_cut=None,
                 concat_shortcut=None,
                 se=False,
                 drop_connect_rate=0,
                 res2net_style=False,
                 bconcat=True,
                 is_training=None,
                 return_value=False,
                 outputs_collections=None,
                 scope=None):
    with tf.variable_scope(scope, 'dense_block', [inputs]) as sc:

        def fun_warp(x):
            residual = x
            if drop_connect_rate > 0:
                # residual = drop_spatial(residual, drop_connect_rate=drop_connect_rate)
                residual = drop_block(residual, drop_connect_rate=drop_connect_rate, block_size=5)
            residual = slim.conv2d(residual, growth_rate, [1,1], stride=1, scope='conv_1x1')
            residual_cpy = residual
            # if use_conv3x3:
            #     xnew = slim.conv2d(residual, growth_rate, [3, 3], stride=1, scope='conv_3x3')
            #     xnew = tf.add(residual, xnew, name='add_shortcut')
            #     xnew = slim.conv2d(xnew, growth_rate, [3, 3], stride=1, scope='conv_3x3_2')
            if isinstance(kernel_multi, list):
                xlist = []
                if res2net_style:
                    for i, k in enumerate(kernel_multi):
                        if k == 0:
                            xone = residual_cpy
                        else:
                            xone = residual if len(xlist)==0 else tf.add(residual, xlist[-1], name='add_shortcut')#slim.conv2d(xlist[-1], growth_rate, [1,1], stride=1)#tf.add(residual, xlist[-1], name='add_shortcut')
                            # if k==5:
                            #     xone = slim.separable_conv2d(residual, None, [3, 3], stride=1, rate=[2, 2],scope="dwconv_3X3_%d" % i, depth_multiplier=1)
                            if use_conv3x3:
                                xone = slim.conv2d(xone, growth_rate, [3, 3], stride=1, scope="conv_3X3_%d"%i)
                            else:
                                xone = slim.separable_conv2d(xone, None, [k, k], stride=1, scope="dwconv_3X3_%d"%i, depth_multiplier=1)#, normalizer_fn=None, activation_fn=None)
                        xlist.append(xone)
                else:
                    for i, k in enumerate(kernel_multi):
                        if k==0:
                            xone = residual
                        # elif k==5:
                        #     xone = slim.separable_conv2d(residual, None, [3, 3], stride=1, rate=[2,2], scope="dwconv_3X3_%d"%i, depth_multiplier=1)#, normalizer_fn=None, activation_fn=None)
                        # #     xone = slim.max_pool2d(residual, [2, 2], stride=2)
                        # #     xone = slim.separable_conv2d(xone, None, [3, 3], stride=1, scope="dwconv_3X3_%d" % i, depth_multiplier=1)
                        # #     xone = tf.image.resize_bilinear(xone, size=[tf.shape(residual)[1], tf.shape(residual)[2]])
                        else:
                            if use_conv3x3:
                                xone = slim.conv2d(residual, growth_rate, [3, 3], stride=1, scope="conv_3X3_%d"%i)
                            else:
                                xone = slim.separable_conv2d(residual, None, [k, k], stride=1, scope="dwconv_3X3_%d"%i, depth_multiplier=1)#, normalizer_fn=None, activation_fn=None)
                        xlist.append(xone)
                if concat_shortcut is not None:
                    xlist.append(concat_shortcut)
                xnew = xlist[0] if len(xlist)==1 else tf.concat(xlist, axis=-1)
            else:
                if res2net_style:
                    xlist = []
                    for i in range(kernel_multi):
                        xone = residual if len(xlist)==0 else tf.add(residual, xlist[-1], name='add_shortcut')#slim.conv2d(xlist[-1], growth_rate, [1,1], stride=1)#tf.add(residual, xlist[-1], name='add_shortcut')
                        if use_conv3x3:
                            xone = slim.conv2d(xone, growth_rate, [3, 3], stride=1, scope="conv_3X3_%d"%i)
                        else:
                            xone = slim.separable_conv2d(xone, None, [3, 3], stride=1, scope="dwconv_3X3_%d"%i, depth_multiplier=1)#, normalizer_fn=None, activation_fn=None)
                        xlist.append(xone)
                else:
                    if use_conv3x3:
                        xone = slim.conv2d(residual, growth_rate*kernel_multi, [3, 3], stride=1, scope="conv_3X3")
                    else:
                        xone = slim.separable_conv2d(residual, None, [3, 3], stride=1, scope="dwconv_3X3", depth_multiplier=kernel_multi)  # , normalizer_fn=None, activation_fn=None)
                xnew = [xone, residual_cpy] if bconcat else [xone]
                if concat_shortcut is not None:
                    xnew.append(concat_shortcut)
                xnew = xnew[0] if len(xnew)==1 else tf.concat(xnew, axis=-1)
            # if residual.shape.as_list()[-1] != growth_rate:
            #     residual = slim.conv2d(residual, growth_rate, [1,1], stride=1, scope='conv_1x1_')
            if short_cut is not None:
                xnew = tf.add(xnew, short_cut, name='short_cut_add')
            if se:
                xnew = se_unit_(xnew)
            if xnew.shape.as_list()[-1] != growth_rate:
                xnew = slim.conv2d(xnew, growth_rate, [1,1], stride=1, scope='conv_1x1_')
            return xnew

        fun_warp = tf.contrib.layers.recompute_grad(fun_warp) if efficient_dense_enable and is_training else fun_warp


        # residual = slim.separable_conv2d(inputs, None, [3, 3], stride=1, scope="dwconv_3X3", depth_multiplier=multi)
        # residual = slim.conv2d(residual, growth_rate, [1, 1], stride=1, scope='conv_1x1_')

        value = fun_warp(inputs)

        output = tf.concat([inputs, value], axis=3)
    if return_value:
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output), value
    else:
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)



@slim.add_arg_scope
def stage2(inputs,
           growth_rate,
           repeat=3,
           kernel_multi=1,
           is_training=None,
           outputs_collections=None):
    relist=[]
    net1 = slim.conv2d(inputs, growth_rate, [1, 1], stride=1)
    residual = slim.conv2d(inputs, growth_rate, [1, 1], stride=1)
    for i in range(repeat-1):
        with tf.variable_scope('denselayer_stage2_{}'.format(i), use_resource=True):
            x = residual if len(relist)==0 else tf.add(residual, relist[-1], name='add_stage2%d_'%i)
            x = slim.separable_conv2d(x, None, [3, 3], stride=1, depth_multiplier=kernel_multi)
            x = slim.conv2d(x, growth_rate, [1, 1], stride=1)
            relist.append(x)

    output = tf.concat(relist + [net1], axis=3)
    return slim.utils.collect_named_outputs(outputs_collections,
                                            'stage2_concat',
                                            output)


def tiny_dsod_v1tiny_mnn(inputs,
              dropout_keep_prob=0.999,
              is_training=True,
              reuse=None,
              override_params=None,
              scope='tiny_dsod_mine'):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))
    end_points = {}
    with tf.variable_scope(scope, 'tiny_dsod_mine', [inputs]) as scope:
        end_points_collection = scope.original_name_scope + '_end_points'
        with slim.arg_scope(
                [slim.conv2d, slim.avg_pool2d, slim.max_pool2d, dense_block_b, trans_down_block, transition_layer_mine, stage2],
                outputs_collections=[end_points_collection]):
            stem_block_output = stem_block_mine(inputs, num_init_channel=16, num_out_channel=16)
            net = stem_block_output
            net = trans_down_block(net, out_channel=32, depth_conv=True, kernel_multi=[3,3,3])
            net = stage2(net, growth_rate=16, kernel_multi=[3], repeat=3, is_training=is_training)
            net = trans_down_block(net, out_channel=48, depth_conv=True, kernel_multi=[3,3])

            C2_3 = net
            C2_4 = slim.avg_pool2d(C2_3, [2,2], stride=2)
            C2_5 = slim.avg_pool2d(C2_4, [2,2], stride=2)
            with tf.variable_scope("stage2b", 'block', [net]) as sc:
                for i in range(5):
                    with tf.variable_scope('unit_%d' % (i + 1), values=[net], use_resource=True):
                        net = dense_block_b(net, growth_rate=16, kernel_multi=[0, 3, 5], is_training=is_training)

            with tf.variable_scope("sp_block2b", 'block', [net]) as sc:
                with tf.variable_scope('unit_1'):
                    net = transition_layer_mine(net, num_channel=64, is_avgpool=True, se=False)

            C3_4 = net
            C3_5 = slim.avg_pool2d(C3_4, [2,2], stride=2)
            with tf.variable_scope("stage3b", 'block', [net]) as sc:
                for i in range(5):
                    with tf.variable_scope('unit_%d' % (i + 1), values=[net], use_resource=True):
                        net = dense_block_b(net, growth_rate=36, kernel_multi=[0, 3, 5], is_training=is_training)

            net = tf.concat([C2_4, net], axis=-1)
            with tf.variable_scope("sp_block3b", 'block', [net]) as sc:
                with tf.variable_scope('unit_1'):
                    net = transition_layer_mine(net, num_channel=160, is_avgpool=True, se=False)

            C4_5 = net
            with tf.variable_scope("stage4", 'block', [net]) as sc:
                for i in range(4):
                    with tf.variable_scope('unit_%d' % (i + 1), values=[net], use_resource=True):
                        net = dense_block_b(net, growth_rate=48, kernel_multi=[0, 3, 5], is_training=is_training)

            net = tf.concat([C2_5, C3_5, net], axis=-1)
            with tf.variable_scope("sp_block4", 'block', [net]) as sc:
                with tf.variable_scope('unit_1'):
                    net = transition_layer_mine(net, num_channel=176, is_avgpool=True, se=False,
                                                down_sample=False)


            outputs_collections=None
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points

def tiny_dsod_v1tiny(inputs,
              dropout_keep_prob=0.999,
              is_training=True,
              reuse=None,
              override_params=None,
              scope='tiny_dsod_mine'):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))
    end_points = {}
    with tf.variable_scope(scope, 'tiny_dsod_mine', [inputs]) as scope:
        end_points_collection = scope.original_name_scope + '_end_points'
        with slim.arg_scope(
                [slim.conv2d, slim.avg_pool2d, slim.max_pool2d, stem_block_mine, dense_block_b, trans_down_block, transition_layer_mine, stage2],
                outputs_collections=[end_points_collection]):
            stem_block_output = stem_block_mine(inputs, num_init_channel=16, num_out_channel=16)
            net = stem_block_output
            net = trans_down_block(net, out_channel=32, depth_conv=True, kernel_multi=3)
            net = stage2(net, growth_rate=16, kernel_multi=1, repeat=3, is_training=is_training)
            net = trans_down_block(net, out_channel=48, depth_conv=True, kernel_multi=2)

            C2_3 = net
            C2_4 = slim.max_pool2d(C2_3, [2,2], stride=2)
            C2_5 = slim.max_pool2d(C2_4, [2,2], stride=2)
            with tf.variable_scope("stage2b", 'block', [net]) as sc:
                for i in range(5):
                    with tf.variable_scope('unit_%d' % (i + 1), values=[net], use_resource=True):
                        net = dense_block_b(net, growth_rate=16, kernel_multi=[0, 3, 5], is_training=is_training)

            with tf.variable_scope("sp_block2b", 'block', [net]) as sc:
                with tf.variable_scope('unit_1'):
                    net = transition_layer_mine(net, num_channel=64, is_avgpool=False, se=False)

            C3_4 = net
            C3_5 = slim.max_pool2d(C3_4, [2,2], stride=2)
            with tf.variable_scope("stage3b", 'block', [net]) as sc:
                for i in range(5):
                    with tf.variable_scope('unit_%d' % (i + 1), values=[net], use_resource=True):
                        net = dense_block_b(net, growth_rate=36, kernel_multi=[0, 3, 5], is_training=is_training)

            net = tf.concat([C2_4, net], axis=-1)
            with tf.variable_scope("sp_block3b", 'block', [net]) as sc:
                with tf.variable_scope('unit_1'):
                    net = transition_layer_mine(net, num_channel=160, is_avgpool=False, se=False)

            C4_5 = net
            with tf.variable_scope("stage4", 'block', [net]) as sc:
                for i in range(4):
                    with tf.variable_scope('unit_%d' % (i + 1), values=[net], use_resource=True):
                        net = dense_block_b(net, growth_rate=48, kernel_multi=[0, 3, 5], is_training=is_training)

            net = tf.concat([C2_5, C3_5, net], axis=-1)
            with tf.variable_scope("sp_block4", 'block', [net]) as sc:
                with tf.variable_scope('unit_1'):
                    net = transition_layer_mine(net, num_channel=176, is_avgpool=False, se=False,
                                                down_sample=False)


            outputs_collections=None
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points

def tiny_dsod_v2tiny(inputs,
              dropout_keep_prob=0.999,
              is_training=True,
              reuse=None,
              override_params=None,
              scope='tiny_dsod_mine'):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))
    with tf.variable_scope(scope, 'tiny_dsod_mine', [inputs]) as scope:
        end_points_collection = scope.original_name_scope + '_end_points'
        with slim.arg_scope(
                [slim.conv2d, slim.avg_pool2d, slim.max_pool2d, stem_block_mine, dense_block_b, trans_down_block, transition_layer_mine, stage2],
                outputs_collections=[end_points_collection]):
            stem_block_output = stem_block_mine(inputs, num_init_channel=16, num_out_channel=16)
            net = stem_block_output
            net = trans_down_block(net, out_channel=None, depth_conv=True, kernel_multi=[2,2])
            # net = slim.separable_conv2d(net, None, [3, 3], stride=2, depth_multiplier=2)  # , normalizer_fn=None, activation_fn=None)

            net = stage2(net, growth_rate=16, kernel_multi=1, repeat=3, is_training=is_training)
            net = trans_down_block(net, out_channel=None, depth_conv=True, kernel_multi=1)
            # net = slim.separable_conv2d(net, None, [3, 3], stride=2, depth_multiplier=1)  # , normalizer_fn=None, activation_fn=None)
            # net = slim.conv2d(net, 40, [1, 1], stride=1)

            h, w = input_shape[1:3]

            repeats=[4,5,6]
            graw_rates=[16,32,32]
            num_channels=[64,128,128]
            kernel_multi=[3,3,0]#[0, 3, 5]
            SIB = False
            enable_se = False
            res2net_style = True
            bconcat = False

            def shortcut_info_build(input, mid_channels, output_channels):
                # x = slim.separable_conv2d(input, None, [3, 3], stride=2, padding='SAME')
                x = slim.max_pool2d(input, [2, 2], stride=2, padding='SAME')
                x = slim.conv2d(x, mid_channels, [1, 1], stride=1)
                x2 = slim.max_pool2d(x, [3, 3], stride=1, padding='SAME')
                x3 = slim.max_pool2d(x2, [3, 3], stride=1, padding='SAME')
                x = tf.concat([x,x2,x3], axis=-1)
                x = slim.conv2d(x, mid_channels, [1, 1], stride=1)#, normalizer_fn=None, activation_fn=tf.nn.relu6)
                x2 = slim.separable_conv2d(x, None, [3, 3], stride=1, padding='SAME')
                x = slim.separable_conv2d(tf.add(x, x2, name='add_res2'), None, [3, 3], stride=1, padding='SAME')
                # # x3 = slim.separable_conv2d(x, None, [3, 3], stride=1, rate=[2, 2], depth_multiplier=1)
                # x = tf.maximum(x, x2, name='max_12')
                # x = tf.maximum(x, x3, name='max_23')
                x = slim.conv2d(x, output_channels, [1, 1], stride=1)#, normalizer_fn=None, activation_fn=tf.nn.relu6)
                x = tf.image.resize_bilinear(x, size=[tf.shape(input)[1],  tf.shape(input)[2]])
                return x

            shortcut3 = None if SIB==False else shortcut_info_build(net, graw_rates[0], graw_rates[0]*len(kernel_multi))
            with tf.variable_scope("stage2b", 'block', [net]) as sc:
                for i in range(repeats[0]):
                    with tf.variable_scope('unit_%d' % (i + 1), values=[net], use_resource=True):
                        net = dense_block_b(net, growth_rate=graw_rates[0], kernel_multi=kernel_multi, short_cut=shortcut3, res2net_style=res2net_style, bconcat=bconcat, drop_connect_rate=drop_rate/2,is_training=is_training)

            with tf.variable_scope("sp_block2b", 'block', [net]) as sc:
                with tf.variable_scope('unit_1'):
                    net = transition_layer_mine(net, num_channel=num_channels[0], is_convpool=False, se=enable_se)

            shortcut4 = None if SIB==False else shortcut_info_build(net, graw_rates[1], graw_rates[1]*len(kernel_multi))
            with tf.variable_scope("stage3b", 'block', [net]) as sc:
                for i in range(repeats[1]):
                    with tf.variable_scope('unit_%d' % (i + 1), values=[net], use_resource=True):
                        net = dense_block_b(net, growth_rate=graw_rates[1], kernel_multi=kernel_multi, short_cut=shortcut4, res2net_style=res2net_style, bconcat=bconcat, drop_connect_rate=drop_rate, is_training=is_training)

            with tf.variable_scope("sp_block3b", 'block', [net]) as sc:
                with tf.variable_scope('unit_1'):
                    net = transition_layer_mine(net, num_channel=num_channels[1], is_convpool=False, se=enable_se)

            shortcut5 = None if SIB==False else shortcut_info_build(net, graw_rates[2], graw_rates[2]*len(kernel_multi))
            with tf.variable_scope("stage4", 'block', [net]) as sc:
                for i in range(repeats[2]):
                    with tf.variable_scope('unit_%d' % (i + 1), values=[net], use_resource=True):
                        net = dense_block_b(net, growth_rate=graw_rates[2], kernel_multi=kernel_multi, short_cut=shortcut5, res2net_style=res2net_style, bconcat=bconcat, drop_connect_rate=drop_rate, is_training=is_training)

            with tf.variable_scope("sp_block4", 'block', [net]) as sc:
                with tf.variable_scope('unit_1'):
                    net = transition_layer_mine(net, num_channel=num_channels[2], is_convpool=False, se=enable_se, down_sample=False)

            outputs_collections=None
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points

def tiny_dsod_arg_scope(
    is_training=True,
    weight_decay=0.00001,
    stddev=0.09,
    regularize_depthwise=True,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
  """Defines the default MobilenetV1 arg scope.
  Args:
    is_training: Whether or not we're training the model. If this is set to
      None, the parameter is not added to the batch_norm arg_scope.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
  Returns:
    An `arg_scope` to use for the mobilenet v1 model.
  """
  batch_norm_params = {
      'center': True,
      'scale': True,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'updates_collections': batch_norm_updates_collections,
  }
  if is_training is not None:
    batch_norm_params['is_training'] = is_training

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm, padding='SAME'):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.batch_norm, slim.dropout, drop_connect, drop_spatial, drop_block], is_training=is_training):
          with slim.arg_scope([slim.separable_conv2d],
                          weights_regularizer=depthwise_regularizer) as sc:
            return sc


   
def tinydsod_v1tiny(images, training, override_params=None):
    with slim.arg_scope(tiny_dsod.tiny_dsod_arg_scope(is_training=training, batch_norm_decay=0.99)):
        logits, image_features = tiny_dsod.tiny_dsod_v1tiny(images, is_training=training, override_params=override_params)
    return logits, image_features

def tinydsod_v2tiny(images, training, override_params=None):
    with slim.arg_scope(tiny_dsod.tiny_dsod_arg_scope(is_training=training, batch_norm_decay=0.99)):
        logits, image_features = tiny_dsod.tiny_dsod_v2tiny(images, is_training=training, override_params=override_params)
    return logits, image_features
