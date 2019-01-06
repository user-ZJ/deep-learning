import json

from datetime import datetime
import tensorflow as tf
import time

import mobilenet
import os
from data import distorted_inputs
from six.moves import xrange
import numpy as np

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('train_dir', './AgeGenderDeepLearning/Folds/tf/test_fold_is_0','Training directory')
flags.DEFINE_integer('num_preprocess_threads', 4,'Number of preprocessing threads')
flags.DEFINE_integer('batch_size', 10,'Batch size')
flags.DEFINE_integer('image_size', 224,'Image size')
flags.DEFINE_float('depth_multiplier', 1.0,'Depth multiplier for mobilenet')
flags.DEFINE_integer('num_classes', 9,'num_classes')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('checkpoint_dir', './log','Directory for writing training checkpoints and logs')
flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('log_every_n_steps', 50, 'Number of steps per log')
flags.DEFINE_integer('number_of_steps', 5000,'Number of training steps to perform before stopping')
flags.DEFINE_integer('save_summaries_secs', 30,'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 30,'How often to save checkpoints, secs')
flags.DEFINE_string('logdir', './log', 'log dir')
flags.DEFINE_string('fine_tune_checkpoint', '','Checkpoint from which to start finetuning.')
tf.app.flags.DEFINE_boolean('log_device_placement', False,"""Whether to log device placement.""")

FLAGS = flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94

def get_learning_rate():
  if FLAGS.fine_tune_checkpoint:
    # If we are fine tuning a checkpoint we need to start at a lower learning
    # rate since we are farther along on training.
    return 1e-4
  else:
    return 0.045

def get_quant_delay():
  if FLAGS.fine_tune_checkpoint:
    # We can start quantizing immediately if we are finetuning.
    return 0
  else:
    # We need to wait for the model to train a bit before we quantize if we are
    # training from scratch.
    return 250000

def build_model():
    """
    为模型创建图，用于训练和量化
    :return:
    g: Graph with fake quantization ops and batch norm folding suitable for
    training quantized weights.
    train_tensor: Train op for execution during training.
    """
    g = tf.Graph()
    # tf.train.replica_device_setter为分布式部署方案
    with g.as_default(),tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        # 1.获取image和label
        images, labels, _ = distorted_inputs(FLAGS.train_dir, FLAGS.batch_size, FLAGS.image_size,
                                             FLAGS.num_preprocess_threads,is_train=True)
        with slim.arg_scope(mobilenet.mobilenet_v1_arg_scope(is_training=True)):
            logits, _ = mobilenet.mobilenet_v1(
                images,
                is_training=True,
                depth_multiplier=FLAGS.depth_multiplier,
                num_classes=FLAGS.num_classes)
        # 2.定义loss
        final_result = tf.nn.softmax(logits,name='final_result')
        total_loss = tf.losses.sparse_softmax_cross_entropy(labels, final_result)

        # 调用重写器，生成量化图，从而允许更好的模型精度。
        if FLAGS.quantize:
            tf.contrib.quantize.create_training_graph(quant_delay=get_quant_delay())
        # 3.获取total loss 可以加正则化项
        # total_loss = tf.losses.get_total_loss(name='total_loss')
        # 4.定义学习率衰减，在decay_steps之前学习率不衰减
        num_epochs_per_decay = 2.5
        imagenet_size = 1271167
        decay_steps = int(imagenet_size / FLAGS.batch_size * num_epochs_per_decay)
        # 指数衰减学习率方法
        # decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
        learning_rate = tf.train.exponential_decay(
            get_learning_rate(), #初始学习率
            tf.train.get_or_create_global_step(), #训练step
            decay_steps, # 衰减速度，每decay_steps轮，学习率乘以_LEARNING_RATE_DECAY_FACTOR
            _LEARNING_RATE_DECAY_FACTOR, #衰减系数
            staircase=True)
        # 5.定义优化器
        global_step = tf.train.get_or_create_global_step()
        optz = lambda learning_rate:tf.train.GradientDescentOptimizer(learning_rate)
        # 6.创建训练器，使用优化器优化loss
        train_opt = tf.contrib.layers.optimize_loss(total_loss,global_step,learning_rate,optz,clip_gradients=4.)

        init = tf.global_variables_initializer()
        return g,init,train_opt,total_loss


def get_checkpoint_init_fn():
    """
    如果提供了检查点，则返回检查点init_fn
    :return:
    """
    if FLAGS.fine_tune_checkpoint:
        exclusions = ['MobilenetV1/Logits',
                      'MobilenetV1/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclusions)
        # tf.assign:通过将 "value" 赋给 "ref" 来更新 "ref",将global step赋值为0
        global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
        # 当加载float类型模型，量化参数min/max不存在
        # 通过设置ignore_missing_vars = True来忽略恢复期间丢失的变量
        slim_init_fn = slim.assign_from_checkpoint_fn(
            FLAGS.fine_tune_checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)

        def init_fn(sess):
            slim_init_fn(sess)
            # If we are restoring from a floating point model, we need to initialize
            # the global step to zero for the exponential decay to result in
            # reasonable learning rates.
            sess.run(global_step_reset)

        return init_fn
    else:
        return None



def main(_):
    input_file = os.path.join(FLAGS.train_dir, 'md.json')
    print(input_file)
    with open(input_file, 'r') as f:
        md = json.load(f)
    g,init,train_op,total_loss = build_model()
    sess = tf.Session(graph=g,config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
    # get_checkpoint_init_fn()
    with g.as_default():
        # 模型保存和日志保存
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('log', sess.graph)
        #7.训练模型
        sess.run(init)
#        checkpoint_init_fn = get_checkpoint_init_fn()
#        checkpoint_init_fn(sess)
        tf.train.start_queue_runners(sess=sess)
        for step in xrange(FLAGS.number_of_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, total_loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                tf.summary.scalar('total_loss',total_loss)
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.3f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch))

            # Loss only actually evaluated every 100 steps?
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 100 == 0 or (step + 1) == FLAGS.number_of_steps:
                print("save model at %d step" % step)
                saver.save(sess, FLAGS.checkpoint_dir+'/model.ckpt', global_step=step)


if __name__ == '__main__':
    tf.app.run()

