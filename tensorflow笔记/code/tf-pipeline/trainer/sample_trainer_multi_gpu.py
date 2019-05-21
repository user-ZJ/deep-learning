#coding=utf-8
import argparse
import logging
import random
import re
from datetime import datetime

import numpy as np
import time
import os
import tensorflow as tf
from six.moves import xrange

from data_generator.sample_data_generator import SampleDataGenerator
from utils.generic_utils import initialize_logger, mkdir, get_available_gpus
from models import rescnn
from utils.triplet_loss import batch_hard_triplet_loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sample models on TIMIT/VCTK data')
    parser.add_argument('--tfrecord_dir', type=str,required=True, help="Train TfRecord file path")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--model_type', type=str, default='rescnn', help="Type of model to run",
                        choices=['dnn', 'rnn', 'cnn','rescnn'])
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--num_threads', type=int, default=4, help="Number of threads for enqueue operation")
    args = parser.parse_args()

    np.random.rand(42)
    random.seed(42)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


    now = int(time.time())
    runs_dir = os.path.join('.', 'runs_dir', str(now))
    mkdir(runs_dir)
    initialize_logger(runs_dir)

    num_gpus = len(get_available_gpus())
    assert args.batch_size%num_gpus == 0,"batch_size%num_gpu must be 0"

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    data_generator = SampleDataGenerator(
        sess,
        tfrecord_dir=args.tfrecord_dir,
        batch_size=args.batch_size,
        model_type=args.model_type,
        num_threads=args.num_threads
    )

    INITIAL_LEARNING_RATE = 1e-4
    MOVING_AVERAGE_DECAY = 0.9999


    with tf.device('/cpu:0'):
        global_step = tf.train.get_or_create_global_step()
        opt = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE)
        inputs = tf.placeholder(dtype=tf.float32,shape=[None,64,64,1],name="inputs")
        labels = tf.placeholder(dtype=tf.float32, shape=[None, 109], name="labels")
        inputs_splits = tf.split(inputs,num_gpus,axis=0)
        labels_splits = tf.split(labels,num_gpus,axis=0)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % (i)) as scope:
                        logits = rescnn.rescnn(inputs_splits[i])
                        loss = batch_hard_triplet_loss(tf.argmax(labels, axis=1), logits)
                        tf.summary.scalar(scope+"loss",loss)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)


        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        for grad,var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/gradients',grad)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # 模型保存
        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(runs_dir, 'log/train'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(runs_dir, 'log/valid'))

        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        total_step = args.epochs * data_generator.train_spe
        for step in range(total_step):
            start_time = time.time()
            train_data,train_label=data_generator.generator('train').__next__()
            validation_data, validation_labels = data_generator.generator('validation').__next__()
            train_summary_str,_, train_loss = sess.run([summary_op,train_op, loss],feed_dict={inputs:train_data,labels:train_label})
            duration = time.time() - start_time
            if step % 100 == 0:
                valid_summary_str,valid_loss = sess.run([summary_op,loss], feed_dict={inputs: train_data, labels: train_label})
                num_examples_per_step = data_generator.batch_size
                examples_per_sec = num_examples_per_step / (duration + 1e-4)
                sec_per_batch = float(duration)
                logging.info('{}: epoch:{:d} {:d}/{:d}, train/valid loss = {:4f}/{:4f} ({:1f} examples/sec; {:3f} ' 'sec/batch)'.format(
                        datetime.now(), int(step / data_generator.train_spe + 1), int(step % data_generator.train_spe),
                        int(data_generator.train_spe), train_loss, valid_loss, examples_per_sec, sec_per_batch))
                train_writer.add_summary(train_summary_str, step)
                valid_writer.add_summary(valid_summary_str, step)
            if step % 1000 == 0 or (step + 1) == total_step:
                logging.info("save model at {} step".format(step))
                saver.save(sess, os.path.join(runs_dir, "checkpoints") + '/model.ckpt', global_step=step)

        data_generator.clean_up()
        coord.request_stop()
        coord.join(threads)