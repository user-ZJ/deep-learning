import argparse
import random
from datetime import datetime

import numpy as np
import time
import os
import tensorflow as tf

from data_generator.sample_data_generator import SampleDataGenerator
from utils.generic_utils import initialize_logger, mkdir
from models import rescnn
from utils.triplet_loss import batch_hard_triplet_loss

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


    now = int(time.time())
    runs_dir = os.path.join('.', 'runs_dir', str(now))
    mkdir(runs_dir)
    initialize_logger(runs_dir)

    sess = tf.Session()
    data_generator = SampleDataGenerator(
        sess,
        tfrecord_dir=args.tfrecord_dir,
        batch_size=args.batch_size,
        model_type=args.model_type,
        num_threads=args.num_threads
    )
    #shape, num_labels, train_spe, validation_spe, test_spe = data_generator.get_data_info()

    if args.model_type == 'rescnn':
        inputs = tf.placeholder(dtype=tf.float32,shape=[None,64,64,1],name="inputs")
        labels = tf.placeholder(dtype=tf.float32, shape=[None,109], name="labels")
        logits = rescnn.rescnn(inputs)

    # 定义损失函数
    #total_loss = tf.losses.softmax_cross_entropy(labels, outputs)
    loss = batch_hard_triplet_loss(tf.argmax(labels, axis=1), logits)
    tf.summary.scalar('loss',loss)

    # 定义学习率衰减
    learning_rate = tf.train.exponential_decay(
        1e-4,  # 初始学习率
        tf.train.get_or_create_global_step(),  # 训练step
        1000,#300000//args.batch_size,  # 衰减速度，每decay_steps轮，学习率乘以_LEARNING_RATE_DECAY_FACTOR
        0.94,  # 衰减系数
        staircase=True)
    tf.summary.scalar('learning_rate',learning_rate)
    #定义global_step计算训练步数
    global_step = tf.train.get_or_create_global_step()
    # 定义优化器
    #optz = lambda learning_rate: tf.train.GradientDescentOptimizer(learning_rate)
    #train_opt = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optz, clip_gradients=4.)
    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    # 创建tensorboard日志
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(runs_dir,'log/train'), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(runs_dir,'log/valid'))

    # 初始化所有变量
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)



    total_step = args.epochs*data_generator.train_spe
    for step in range(total_step):
        # 生成batch_size数据
        train_data, train_labels = data_generator.generator('train').__next__()
        validation_data, validation_labels = data_generator.generator('validation').__next__()
        start_time = time.time()
        train_summary_str,_, train_loss = sess.run([summary_op,train_opt, loss],feed_dict={inputs:train_data,labels:train_labels})
        duration = time.time() - start_time
        if step % 100 == 0:
            valid_summary_str,valid_loss = sess.run([summary_op,loss],feed_dict={inputs:validation_data,labels:validation_labels})
            num_examples_per_step = data_generator.batch_size
            examples_per_sec = num_examples_per_step / (duration + 1e-4)
            sec_per_batch = float(duration)
            format_str = ('%s: epoch:%d %d/%d, train/valid loss = %.4f/%.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
            print(format_str % (datetime.now(), step/data_generator.train_spe+1,step%data_generator.train_spe,data_generator.train_spe,
                                train_loss, valid_loss,examples_per_sec, sec_per_batch))
            train_writer.add_summary(train_summary_str, step)
            valid_writer.add_summary(valid_summary_str, step)
        if step % 1000 == 0 or (step + 1) == total_step:
            print("save model at %d step" % step)
            saver.save(sess, os.path.join(runs_dir,"checkpoints") + '/model.ckpt', global_step=step)

    coord.request_stop()
    coord.join(threads)

