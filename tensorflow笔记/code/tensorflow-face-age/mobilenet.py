# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py

from collections import namedtuple
import functools
import tensorflow as tf
slim = tf.contrib.slim

# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# MOBILENETV1_CONV_DEFS specifies the MobileNet body
MOBILENETV1_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]

def _fixed_padding(inputs, kernel_size, rate=1):
    # 但使用prepad输入，以便输出尺寸与使用'SAME'填充时相同
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                             kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                    [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs

def mobilenet_v1_base(inputs,
                      final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      use_explicit_padding=False,
                      scope=None):
    """
    :param inputs: [batch_size, height, width, channels]
    :param final_endpoint:神经网络最后一层scope name
    :param min_depth:所有卷积运算的小通道数，当depth_multiplier < 1时强制执行，depth_multiplier >= 1不强制执行
    :param depth_multiplier:所有卷积运算的通道系数，（0,1],模型压缩参数
    :param conv_defs:指定网络体系结构的ConvDef命名元组列表
    :param output_stride:int，指定请求的输入与输出空间分辨率之比。允许值为8（精确完全卷积模式），16（快速完全卷积模式），32（分类模式）
    :param use_explicit_padding:使用'VALID'填充进行卷积，但使用prepad输入，以便输出尺寸与使用'SAME'填充时相同。
    :param scope:可选的variable_scope
    :return:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.
    """
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = MOBILENETV1_CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    padding = 'SAME'
    if use_explicit_padding:
        padding = 'VALID'
    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
            # The current_stride variable keeps track of the output stride of the
            # activations, i.e., the running product of convolution strides up to the
            # current network layer. This allows us to invoke atrous convolution
            # whenever applying the next convolution would result in the activations
            # having output stride larger than the target output_stride.
            current_stride = 1

            # 卷积压缩率
            rate = 1

            net = inputs
            for i, conv_def in enumerate(conv_defs):
                # 定义scope
                end_point_base = 'Conv2d_%d' % i

                if output_stride is not None and current_stride == output_stride:
                    # 如果指定了output_stride,那么我们需要使用stride = 1的atrous卷积，并将atrous rate乘以当前单位的步幅，以便在后续层中使用。
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride

                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel)
                    net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                                      stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                elif isinstance(conv_def, DepthSepConv):
                    #深度卷积
                    end_point = end_point_base + '_depthwise'
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel, layer_rate)
                    net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                depth_multiplier=1,
                                                stride=layer_stride,
                                                rate=layer_rate,
                                                normalizer_fn=slim.batch_norm,
                                                scope=end_point)

                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                    #点卷积
                    end_point = end_point_base + '_pointwise'
                    net = slim.conv2d(net, depth(conv_def.depth), [1, 1],
                                      stride=1,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def mobilenet_v1(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV1',
                 global_pool=False):
    """
    :param inputs:[batch_size, height, width, channels]
    :param num_classes:分类数，如果为0或者None，则省略logits图层，并返回logits图层的输入要素（在dropout之前）
    :param dropout_keep_prob:激活百分比
    :param is_training:是否训练
    :param min_depth:所有卷积运算的小通道数，当depth_multiplier < 1时强制执行，depth_multiplier >= 1不强制执行
    :param depth_multiplier:所有卷积运算的通道系数，（0,1],模型压缩参数
    :param conv_defs:指定网络体系结构的ConvDef命名元组列表
    :param prediction_fn:从logits中获取预测的函数
    :param spatial_squeeze:如果为True，则logits的形状为[B，C]，如果false logits的形状为[B，1,1，C]，其中B为batch_size，C为类别数
    :param reuse:是否应该重用网络及其变量。 允许给出能够重用“scope”。
    :param scope:可选的variable_scope
    :param global_pool:可选的布尔标志，用于控制logits图层之前的avgpooling。
    如果为false或未设置，则使用固定窗口完成池化，将默认大小的输入减少到1x1，而较大的输入会导致更大的输出。
    如果为true，则将任何输入大小合并为1x1。
    :return:
    net:如果num_classes是非零整数，则带有logits的2D Tensor（pre-softmax激活）;如果num_classes为0或None，则为logits层的非dropout输入。
    end_points:从网络组件到相应激活的字典。
    """
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))
    with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
            net, end_points = mobilenet_v1_base(inputs, scope=scope,
                                                min_depth=min_depth,
                                                depth_multiplier=depth_multiplier,
                                                conv_defs=conv_defs)
        with tf.variable_scope('Logits'):
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            else:
                # Pooling with a fixed kernel size.
                kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                      scope='AvgPool_1a')
                end_points['AvgPool_1a'] = net
            if not num_classes:
                return net, end_points

            # 1 x 1 x 1024
            #net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
            # 使用1x1卷积替代全连接层
            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                 normalizer_fn=None, scope='Conv2d_1c_1x1')
            if spatial_squeeze:
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        end_points['Logits'] = logits
        if prediction_fn:
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points

mobilenet_v1.default_image_size = 224

# functools，用于高阶函数：指那些作用于函数或者返回其它函数的函数，通常只要是可以被当做函数调用的对象就是这个模块的目标
def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func

# 模型压缩
mobilenet_v1_075 = wrapped_partial(mobilenet_v1, depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(mobilenet_v1, depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(mobilenet_v1, depth_multiplier=0.25)

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """
    定义内核大小，用于池化层
    如果输入图像的形状在图形构建时未知，则此函数假定输入图像足够大
    :param input_tensor: [batch_size, height, width, channels]
    :param kernel_size:[kernel_height, kernel_width]
    :return:
    pooling 的kernel大小
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out

def mobilenet_v1_arg_scope(
    is_training=True,
    weight_decay=0.00004,
    stddev=0.09,
    regularize_depthwise=False,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
    """
    定义默认的MobilenetV1 参数域
    :param is_training:我们是否正在训练模型。 如果将此值设置为None，则该参数不会添加到batch_norm arg_scope中。
    :param weight_decay:模型正规化的权重。
    :param stddev:权重初始化的标准差
    :param regularize_depthwise:是否在depthwise上应用正则化
    :param batch_norm_decay:batch norm的移动平均值
    :param batch_norm_epsilon:小浮点数增加到方差，以避免在batch norm中除以零
    :param batch_norm_updates_collections:batch norm 更新操作集合。
    :return:
    用于mobilenet v1模型的`arg_scope`。
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
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc

