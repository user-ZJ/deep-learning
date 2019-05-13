import logging

import tensorflow as tf

slim = tf.contrib.slim

layers_dict = dict()



def clipped_relu(inputs):
    return tf.clip_by_value(inputs,0,20)


def identity_block(inputs, kernel_size, filters, unit):
    with tf.variable_scope('unit_{}'.format(unit), [inputs]):
        net = slim.conv2d(inputs,filters,[kernel_size,kernel_size],stride=1,padding='same',activation_fn=None,scope='conv_2a')
        net = slim.batch_norm(net, scope='conv_2a_bn')
        net = clipped_relu(net)
        net = slim.conv2d(net, filters, [kernel_size, kernel_size], padding='same',activation_fn=None,
                          scope='conv_2b')
        net = slim.batch_norm(net, scope='conv_2b_bn')
        net = tf.add(net,inputs,name="add")
        net = clipped_relu(net)
        return net



def conv_and_res_block(inputs, filters, block_id):
    with tf.variable_scope('res_block_{}'.format(block_id), [inputs]):
        net = slim.conv2d(inputs,filters,[5,5],stride=2,padding='same',activation_fn=None,scope='conv{}-s'.format(filters))
        net = slim.batch_norm(net,scope='conv{}-s_bn'.format(filters))
        net = clipped_relu(net)
        for i in range(3):
            net = identity_block(net, kernel_size=3, filters=filters, unit=i)
        return net


def cnn_component(inputs):
    net = conv_and_res_block(inputs, 64, block_id=1)
    net = conv_and_res_block(net, 128, block_id=2)
    net = conv_and_res_block(net, 256, block_id=3)
    net = conv_and_res_block(net, 512, block_id=4)
    return net


def rescnn(inputs,vector_size=512):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with tf.variable_scope( 'rescnn', [inputs]) as sc:
            net = cnn_component(inputs)
            #net = tf.reshape(net,(2048,128))
            net = tf.reshape(net,(-1, 4, 2048))
            net = tf.reduce_mean(net,axis=1)
            net = slim.fully_connected(net,vector_size,activation_fn=None,scope='affine')
            net = tf.nn.l2_normalize(net,name='ln')
            return net