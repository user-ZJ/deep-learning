# coding=utf-8
import argparse
import json
import logging
import random

import numpy as np
import tensorflow as tf

from utils.generic_utils import initialize_logger,run_once,threadsafe_generator


class SampleDataGenerator(object):
    """
    1. 处理非常大的tfrecord数据集，无法完全加载到内存
    2. 创建复杂的tensorflow pipeline,如数据增强
    3. 使用缓存、多线程、队列构建高效的数据生成器
    """
    def __init__(self,session,tfrecord_dir, batch_size=64, model_type='dnn', num_threads=4):
        """
         :param train_filenames: Train TfRecord files
         :param validation_filenames: validation TfRecord files
         :param test_filenames: Test TfRecord files
         :param batch_size: Batch size
         :param model_type: Type of model consuming the data.
             If it is dnn 2D tensor is returned (batch_size, time * freq)
             If it is rnn 3D tensor is returned (batch_size, time, freq)
             If it is cnn 4D tensor is returned (batch_size, time, freq, 1)
         :param num_threads: Number of threads for enqueueing data
         """
        random.seed(42)
        np.random.seed(42)

        self.session = session
        # self.train_filenames = train_filenames
        # self.validation_filenames = validation_filenames
        # self.test_filenames = test_filenames
        self.tfrecord_dir = tfrecord_dir
        self.batch_size = batch_size
        self.model_type = model_type
        self.num_threads = num_threads

        self.get_data_info()

        self.train_x_batch, self.train_y_batch = self.prepare_dataset(self.train_tfrecord_list)
        self.validation_x_batch, self.validation_y_batch = self.prepare_dataset(self.validation_tfrecord_list)
        self.test_x_batch, self.test_y_batch = self.prepare_dataset(self.test_tfrecord_list)

        self.coordinator = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coordinator, sess=self.session)

    @run_once
    def get_data_info(self):
        """
        获取数据集信息，包括shape，label数，每轮迭代次数
        Returns shape of data, number of labels, steps per epoch of training, validation and test
        """
        # dataset = tf.data.TFRecordDataset(self.train_filenames)
        # dataset = dataset.map(self.parser)
        # dataset = dataset.take(4)
        # iterator = dataset.make_one_shot_iterator()
        # sample_data = self.session.run(
        #     iterator.get_next()
        # )
        #
        # train_spe = int(np.ceil(self.count_samples(self.train_filenames) * 1.0 / self.batch_size))  #每轮训练batch数
        # validation_spe = int(np.ceil(self.count_samples(self.validation_filenames) * 1.0 / self.batch_size))
        # test_spe = int(np.ceil(self.count_samples(self.test_filenames) * 1.0 / self.batch_size))
        with open(self.tfrecord_dir+"/datainfo.json", 'r', encoding='UTF-8') as f:
            datainfo = json.load(f)
            shape = datainfo['data_shape']
            shape.append(1)
            self.shape = shape
            self.num_labels = datainfo['num_labels']
            self.train_samples = datainfo['train_samples']
            self.validation_samples = datainfo['validation_samples']
            self.test_samples = datainfo['test_samples']
            self.train_spe = self.train_samples//self.batch_size
            self.validation_spe = self.validation_samples//self.batch_size
            self.test_spe = self.test_samples//self.batch_size
            self.train_tfrecord_list = datainfo['train_tfrecord_list']
            self.validation_tfrecord_list = datainfo['validation_tfrecord_list']
            self.test_tfrecord_list = datainfo['test_tfrecord_list']

        logging.info("Shape of input data: {}".format(self.shape))
        logging.info("Number of labels in input data: {}".format(self.num_labels))
        logging.info("Steps per epoch - Train: {}, Validation: {}, Test: {}".format(
            self.train_spe, self.validation_spe, self.test_spe))

    def prepare_dataset(self, filename):
        """
        生成batch数据
        Datset transformation pipeline
        :param filename: TfRecord filename
        :return: iterator of x, y batch nodes where x: feature, y: label
        """
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(self.parser)
        # dataset = dataset.shuffle(buffer_size=1024, seed=42)
        dataset = dataset.repeat(count=-1)
        # dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        feature, label_id = iterator.get_next()
        feature, label_id = tf.train.shuffle_batch(
            tensors=[feature, label_id], shapes=[self.shape, [self.num_labels]],
            batch_size=self.batch_size,
            capacity=2048,
            min_after_dequeue=1024,
            enqueue_many=False,
            num_threads=self.num_threads
        )
        return feature, label_id

    @threadsafe_generator
    def generator(self, identifier='train'):
        """
        调用prepare_dataset生成batch size数据
        :param identifier:
        :return:
        """
        if identifier == 'train':
            x_batch, y_batch = self.train_x_batch, self.train_y_batch
        elif identifier == 'validation':
            x_batch, y_batch = self.validation_x_batch, self.validation_y_batch
        else:
            x_batch, y_batch = self.test_x_batch, self.test_y_batch

        while not self.coordinator.should_stop():
            yield self.session.run([x_batch, y_batch])

    def parser(self, record):
        """
        Parse a TfRecord
        :param record: A TfRecord
        :return: x node: Feature, y node: Label
        """
        features = tf.parse_single_example(record, features={
            'feature': tf.FixedLenFeature([], tf.string),
            'label_id': tf.FixedLenFeature([], tf.string),
            't_dim': tf.FixedLenFeature([], tf.int64),
            'f_dim': tf.FixedLenFeature([], tf.int64),
            'file_id': tf.FixedLenFeature([], tf.int64),
            'label_name': tf.FixedLenFeature([], tf.string),
        })
        feature = tf.decode_raw(features['feature'], tf.float32)
        label_id = tf.decode_raw(features['label_id'], tf.float32) #one-hot后label
        t_dim = tf.cast(features['t_dim'], tf.int32)
        f_dim = tf.cast(features['f_dim'], tf.int32)
        feature = self.transform_input(tf.reshape(feature, tf.stack([t_dim, f_dim])))
        return feature, label_id


    def transform_input(self, inp):
        """
        修改inputs数据格式，匹配不同模型
        :param inp:
        :return:
        """
        if self.model_type == 'dnn':
            return tf.layers.flatten(inp)
        elif self.model_type == 'cnn':
            return tf.expand_dims(inp, -1)
        elif self.model_type == 'rescnn':
            return tf.expand_dims(inp, -1)
        else:
            return inp

    def clean_up(self):
        """
        停止数据生成
        :return:
        """
        self.coordinator.request_stop()
        self.coordinator.join(self.threads)

    def get_complete_data(self, filename):
        """
        获取tfrecord中所有数据
        :param filename:
        :return:
        """
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(self.parser)
        dataset = dataset.shuffle(buffer_size=1024, seed=42)
        iterator = dataset.make_one_shot_iterator()
        xt, yt = iterator.get_next()
        xs, ys = [], []
        while True:
            try:
                x, y = self.session.run([xt, yt])
                xs.append(x)
                ys.append(y)
            except:
                break
        return np.array(xs), np.array(ys)

    def get_data_iterator(self, filename):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(self.parser)
        dataset = dataset.shuffle(buffer_size=1024, seed=42)
        iterator = dataset.make_one_shot_iterator()
        return iterator


if __name__ == '__main__':
    initialize_logger(None)

    parser = argparse.ArgumentParser(description='Run sample baseline models on TIMIT/VCTK data')
    parser.add_argument('--tfrecord_dir', type=str,required=True, help="Train TfRecord file path")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--model_type', type=str, default='dnn', help="Type of model to run",
                        choices=['dnn', 'rnn', 'cnn'])
    args = parser.parse_args()

    sess = tf.Session()

    # args.train_filenames = tf.data.Dataset.from_tensor_slices(np.array(train_file_list))
    # args.validation_filenames = tf.data.Dataset.from_tensor_slices(np.array(validation_file_list))
    # args.test_filenames = tf.data.Dataset.from_tensor_slices(np.array(test_file_list))
    # print(args.test_filenames)
    batchDataGenerator = SampleDataGenerator(
        sess,
        args.tfrecord_dir,
        args.batch_size, args.model_type
    )
    batchDataGenerator.get_data_info()
    features,labels = batchDataGenerator.generator().__next__()
    batchDataGenerator.clean_up()
    print(features.shape,labels.shape)
    sess.close()