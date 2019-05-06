import argparse
import logging
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.utils import to_categorical
from tqdm import tqdm

from feature_extractor.feature_extractors import LogMelFeatureExtractor, MfccFeatureExtractor
from utils.audio_utils import audio_predicate, read_wav, resample
from utils.generic_utils import initialize_logger, mkdir, list_files

IMAGE_SIZE=28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
TRAIN_FILE = "train.tfrecords"
VALIDATION_FILE="validation.tfrecords"


class SampleTFRecordCreator(object):

    def __init__(self,dir_path,feature_extractor,num_labels,tfrecords_dir=None,processes=8,context=100,target_sr=16000,samples_per_speaker=None):
        """
        1. 创建tfrecord目录
        2. 获取文件列表及标签
        3. 将数据划分为训练集，验证集，测试集
        :param dir_path:数据存放路径
        :param num_labels:选取的标签个数，对于大型数据集，选取部分标签的数据进行快速调试
        :param tfrecords_dir:tfrecord文件存放目录
        :param dir_path:
        """

        random.seed(42)
        np.random.seed(42)
        self.dir_path = dir_path
        self.feature_extractor = feature_extractor
        self.context = context
        self.validation_frac = 0.1
        self.test_frac = 0.2
        self.target_sr = target_sr
        self.processes = processes
        self.samples_per_speaker = samples_per_speaker
        if self.samples_per_speaker is None or self.samples_per_speaker<=0:
            self.samples_per_speaker = float("inf")
        if tfrecords_dir is not None:
            self.tfrecords_dir = tfrecords_dir
            mkdir(self.tfrecords_dir)
        else:
            self.tfrecords_dir = os.path.join('.', 'tfrecords_dir', str(int(time.time())))
            mkdir(self.tfrecords_dir)
        # 获取所有文件列表和标签列表
        self.all_files = list(list_files(self.dir_path, lambda x: audio_predicate(x)))
        self.all_labels = list(set([self.label_extractor(f) for f in self.all_files]))
        if num_labels < len(self.all_labels):
            self.num_labels = num_labels
        else:
            self.num_labels = len(self.all_labels)
        logging.info("Reading audio files from: {}".format(self.dir_path))
        logging.info("Using {} labels".format(self.num_labels))
        logging.info("Number of context frames: {}".format(self.context))
        logging.info("Data Fractions - Train: {}, Val: {}, Test: {}".format((1 - self.validation_frac - self.test_frac),
                                                                            self.validation_frac, self.test_frac))
        #选取num_label个标签，从大量数据中抽取部分数据用作调试
        self.label_dict = self.choose_labels()
        logging.debug("Labels: {}".format(self.label_dict))
        #获取选中标签的所有文件，并统计每个标签文件个数
        self.id_filename_dict,self.label_count_dict = self.get_dictionaries()
        logging.info("Number of files: {}".format(len(self.id_filename_dict)))
        # 将选取的文件划分为训练集、验证集、测试集，对每个标签中的文件，按比例划分
        self.train_files, self.validation_files, self.test_files = self.divide_data()
        logging.info("Number of Train Files: {}".format(len(self.train_files)))
        logging.info("Number of Dev Files: {}".format(len(self.validation_files)))
        logging.info("Number of Test Files: {}".format(len(self.test_files)))
        logging.info("Input dimensions: {}".format(self.feature_extractor.dim()[1]))

        self.train_samples = 0
        self.validation_samples = 0
        self.test_samples = 0

    @staticmethod
    def _int64_feature(value):
        """Wrapper for inserting int64 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        """Wrapper for inserting bytes features into Example proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _convert_to_tfrecord(self,flag,files):
        """
        提取files中的feature，写入tfrecord文件中
        :param flag: train/validation/test
        :param files:文件id列表
        :return:
        """
        tfrecords_list = []
        count = 0 #统计处理文件的数量
        pbar = tqdm(total=len(files), desc="Creating: {} tfrecord".format(flag))
        # 单线程
        # for i in files:
        #     if count % 1000 == 0:
        #         filename = os.path.join(self.tfrecords_dir, flag + "_" + str(count) + ".tfrecord")
        #         tfrecords_list.append(filename)
        #         writer = tf.python_io.TFRecordWriter(filename)
        #     examples = self.get_examples(i)
        #     for example in examples:
        #         writer.write(example)
        #     count += 1
        #     if count % 1000 == 0:
        #         writer.close()
        #     pbar.update()
        # writer.close()
        # pbar.close()
        # return tfrecords_list,count
        #多线程
        executor = ThreadPoolExecutor(max_workers=self.processes)
        examples_feature = {executor.submit(self.get_examples, i): i for i in files}
        for feature in as_completed(examples_feature):
            if count % 1000 == 0:
                filename = os.path.join(self.tfrecords_dir,flag+"_"+str(count)+".tfrecord")
                tfrecords_list.append(filename)
                writer = tf.python_io.TFRecordWriter(filename)
            i = examples_feature[feature]
            try:
                examples = feature.result()
            except Exception as exc:
                del examples_feature[feature]
                print('%r generated an exception: %s' % (self.id_filename_dict[i], exc))
            else:
                del examples_feature[feature]
                for example in examples:
                    writer.write(example)
                count += 1
                if count % 1000 ==0:
                    writer.close()
            pbar.update()
        writer.close()
        executor.shutdown()
        pbar.close()
        return tfrecords_list,count

    def get_examples(self, i):
        """
        1. 提取音频特征,并将特征分成长度相等的段
        2. 将音频特征及标签等信息写入example
        :param i: 文件id
        :return:
        """
        examples = []
        tup = self.featurize(i)
        label_name = self.label_extractor(self.id_filename_dict[i])
        if tup is None:
            return examples
        feature,label_id=tup  #feature为(*,context,mel)
        for j in range(feature.shape[0]):
            example=tf.train.Example(features=tf.train.Features(feature={
                'feature': self._bytes_feature(feature[j].tostring()),
                'label_id': self._bytes_feature(to_categorical(label_id, self.num_labels).astype(np.float32).tostring()), #label的onehot编码
                't_dim': self._int64_feature(feature[j].shape[0]),
                'f_dim': self._int64_feature(feature[j].shape[1]),
                'file_id': self._int64_feature(i),  # useful for computing clip level statistics
                'label_name': self._bytes_feature(label_name.encode('utf-8')),
            }))
            examples.append(example.SerializeToString())
        return examples

    def label_extractor(self, filepath):
        """
        提取文件对应的标签
        :param filepath:
        :return:
        """
        return os.path.basename(os.path.split(filepath)[0])

    def choose_labels(self):
        random.shuffle(self.all_labels)
        if len(self.all_labels) > self.num_labels:
            label_dict = dict(zip(self.all_labels[:self.num_labels],range(self.num_labels)))
        else:
            label_dict = dict(zip(self.all_labels,range(len(self.all_labels))))
        return label_dict

    def get_dictionaries(self):
        """
        :return:
        1.创建文件名-文件ID的词典
        2. 创建标签和标签对应的文件个数的词典
        """
        selectedfiles = list(filter(lambda x: self.label_extractor(x) in self.label_dict, self.all_files))
        random.shuffle(selectedfiles)
        filelen = len(selectedfiles)
        id_filename_dict = dict(zip(range(filelen),selectedfiles))
        label_count_dict = Counter([self.label_extractor(f) for f in selectedfiles])
        return id_filename_dict,label_count_dict

    def divide_data(self):
        """
        对于每个标签中的数据，按比例划分为训练集、验证集、测试集
        :return:
        """
        train_files, validation_files, test_files, = [], [], []
        curr_count = defaultdict(float)  #计算每个标签中有多少个文件被分配
        keys = list(self.id_filename_dict.keys())
        random.shuffle(keys)
        train_frac = 1.0 - self.validation_frac - self.test_frac
        for key in keys:
            f = self.id_filename_dict[key]
            label = self.label_extractor(f)
            if curr_count[label] < train_frac * min(self.label_count_dict[label], self.samples_per_speaker):
                train_files.append(key)
            elif curr_count[label] < (1.0 - self.test_frac) * min(self.label_count_dict[label],self.samples_per_speaker):
                validation_files.append(key)
            elif curr_count[label] <= 1.0 * min(self.label_count_dict[label], self.samples_per_speaker):
                test_files.append(key)
            curr_count[label] += 1.0
        return train_files, validation_files, test_files

    def create_tfrecords(self):
        self.train_tfrecord_list,self.train_samples = self._convert_to_tfrecord("train",self.train_files)
        self.validation_tfrecord_list, self.validation_samples = self._convert_to_tfrecord("validation", self.validation_files)
        self.test_tfrecord_list, self.test_samples = self._convert_to_tfrecord("test",self.test_files)
        logging.info("file of records - Train: {}, Validation: {}, Test: {}".format(
            self.train_tfrecord_list, self.validation_tfrecord_list, self.test_tfrecord_list))
        logging.info("Number of records - Train: {}, Validation: {}, Test: {}".format(
            self.train_samples, self.validation_samples, self.test_samples))

    def featurize(self, i):
        """
       1. 分帧
       2. 重采样为8K 16bit 单通道数据
       3. 提取特征
       :param i: File id
       :return: If successful returns (feature, label_id) otherwise returns None
       """
        filename = self.id_filename_dict[i]
        label = self.label_extractor(filename)
        y,sr,channel=read_wav(filename,dtype="float32")
        if self.target_sr is not None and self.target_sr != sr:
            y, sr = resample(y, sr, self.target_sr)
        feature = self.feature_extractor(y=y, sr=sr)
        if feature is not None:
            return feature,self.label_dict[label]



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create TfRecords of data sets')
    parser.add_argument('--dir_path', type=str, required=True,help="Dataset path")
    parser.add_argument('--num_labels', type=int, default=50, help='Number of labels to select')
    parser.add_argument('--features', default='logmel', help='Type of featurizer to use', choices=['logmel', 'mfcc'])
    parser.add_argument('--mels', type=int, default=40, help='Number of mels banks to compute')
    parser.add_argument('--mfccs', type=int, default=13, help='Number of mfccs to compute')
    parser.add_argument('--context', type=int, default=100, help='Number of frames as context')
    parser.add_argument('--stride', type=int, default=2, help='Stride for running window in feature extraction')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes to spin for TfRecord creation')
    parser.add_argument('--samples_per_speaker', default=-1, type=int, help='Number of samples per speaker')
    args = parser.parse_args()

    stride = args.stride
    mels = args.mels  # 梅尔滤波器个数
    mfccs = args.mfccs
    context = args.context  # 上下文帧数
    num_labels = args.num_labels
    dir_path = args.dir_path
    processes = args.processes

    if args.features == 'logmel':
        feature_extractor = LogMelFeatureExtractor(mels=mels, context=context, log_mel=True, stride=stride)
    else:
        feature_extractor = MfccFeatureExtractor(mfccs=mfccs, context=context, stride=stride)


    now = int(time.time())
    tfrecords_dir = os.path.join('E:/训练数据包/','tfrecords_dir', str(now))

    mkdir(tfrecords_dir)
    initialize_logger(tfrecords_dir)

    creator = SampleTFRecordCreator(dir_path=dir_path,feature_extractor=feature_extractor,
                                    num_labels=num_labels,tfrecords_dir=tfrecords_dir,processes=4,context=context,
                                    target_sr=16000,samples_per_speaker=args.samples_per_speaker)

    creator.create_tfrecords()