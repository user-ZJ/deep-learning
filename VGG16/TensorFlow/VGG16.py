import tensorflow as tf
'''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_trainable: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
'''
def conv_layer(layer_name, x, out_channels, kernel_size=[3 ,3], stride=[1 ,1 ,1 ,1], is_trainable=True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',dtype=tf.float32,trainable=is_trainable,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()) # default is uniform distribution initialization
        b = tf.get_variable(name='biases',dtype=tf.float32,
                            trainable=is_trainable,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))

        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')


        x = tf.nn.relu(x, name='relu')

        return x

'''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
'''
def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x


'''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
'''
def fc_layer(layer_name, x, out_nodes, keep_prob=0.8):
    shape = x.get_shape()
    # 处理没有预先做flatten的输入
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))

        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, keep_prob)

        return x


def vgg16_net(resized_input_tensor,n_classes, is_trainable=True):
    #height, width = 224,224
    #with tf.Graph().as_default() as graph:
        # resized_input_tensor = tf.placeholder(shape=[None, height, width, 3],dtype=tf.float32)
        # input_groud_truth = tf.placeholder(shape=[None,n_classes],dtype=tf.float32)
    #with tf.name_scope('VGG16'):
    x = conv_layer('conv1_1', resized_input_tensor, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    x = conv_layer('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    with tf.name_scope('pool1'):
        x = pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
    x = conv_layer('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    x = conv_layer('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    with tf.name_scope('pool2'):
        x = pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
    x = conv_layer('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    x = conv_layer('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    x = conv_layer('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    with tf.name_scope('pool3'):
        x = pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
    x = conv_layer('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    x = conv_layer('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    x = conv_layer('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    with tf.name_scope('pool4'):
        x = pool('pool4', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
    x = conv_layer('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    x = conv_layer('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    x = conv_layer('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=is_trainable)
    with tf.name_scope('pool5'):
        x = pool('pool5', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
    x = fc_layer('fc6', x, out_nodes=4096)
    assert x.get_shape().as_list()[1:] == [4096]
    x = fc_layer('fc7', x, out_nodes=4096)
    x = fc_layer('fc8', x, out_nodes=n_classes)
    x = tf.nn.softmax(x)
    # loss = tf.reduce_mean(-tf.reduce_sum(input_groud_truth * tf.log(x), reduction_indices=[1]))
    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    # train_step = optimizer.minimize(tf.convert_to_tensor(loss))

    return x
