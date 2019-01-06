from datetime import datetime
import sys
import threading

import tensorflow as tf
import os
import random
import numpy as np
from six.moves import xrange
from ImageCoder import ImageCoder
import json

RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224

tf.app.flags.DEFINE_string('fold_dir', './AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_0',
                           'the fold contains image-label map')
tf.app.flags.DEFINE_string('data_dir', './aligned',
                           'Data directory')
tf.app.flags.DEFINE_string('output_dir', './AgeGenderDeepLearning/Folds/tf/test_fold_is_0',
                           'Output directory')
tf.app.flags.DEFINE_string('train_list', 'age_train.txt',
                           'Training list')
tf.app.flags.DEFINE_string('valid_list', 'age_val.txt',
                           'Test list')
tf.app.flags.DEFINE_integer('train_shards', 10,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('valid_shards', 2,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 2,
                            'Number of threads to preprocess the images.')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _is_png(filename):
    return '.png' in filename

def _process_image(filename, coder):
    """Process a single image file.
        Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    image = coder.resample_jpeg(image_data)
    return image, RESIZE_HEIGHT, RESIZE_WIDTH


def _convert_to_example(filename, image_buffer, label, height, width):
    """Build an Example proto for an example.
        Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        height: integer, image height in pixels
        width: integer, image width in pixels
        Returns:
        Example proto
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/class/label': _int64_feature(label),
        'image/filename': _bytes_feature(str.encode(os.path.basename(filename))),
        'image/encoded': _bytes_feature(image_buffer),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width)
    }))
    return example


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,labels, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.
        Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).batch编号：0,1
        ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.batch图片编号列表
        name: string, (validation train)
        filenames: list of strings; picture list
        labels: list of integer; label list
        num_shards: number of tfrecord file.
    """
    num_threads = len(ranges)
    assert not num_shards % num_threads
    # 每个线程处理的tfrecord数
    num_shards_per_batch = int(num_shards / num_threads)
    # 计算每个tfrecord中存在哪些图片信息
    shard_ranges = np.linspace(ranges[thread_index][0],ranges[thread_index][1],num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = int(labels[i])

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer, label,
                                          height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _find_image_files(list_file, data_dir):
    files_labels = [l.strip().split(' ') for l in tf.gfile.FastGFile(list_file, 'r').readlines()]
    labels = []
    filenames = []
    for path, label in files_labels:
        jpeg_file_path = '%s/%s' % (data_dir, path)
        if os.path.exists(jpeg_file_path):
            filenames.append(jpeg_file_path)
            labels.append(label)
    unique_labels = set(labels)
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]
    return filenames, labels


def _process_image_files(name, filenames, labels, num_shards):
    """Process and save list of images as TFRecord of Example protos.
        Args:
        name: string, (validation train)
        filenames : picture list
        labels: label list
        num_shards: number of tfrecord file.
    """
    assert len(filenames) == len(labels)
    # num_threads=2,产生len为3的等差数列[0  642 1284]，将数据分为[[0, 642], [642, 1284]]两部分，分别用两个线程处理
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    # Coordinator：协调器，协调线程间的关系可以视为一种信号量，用来做同步
    coord = tf.train.Coordinator()

    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()



def _process_dataset(name, filename, directory, num_shards):
    """Process a complete data set and save it as a TFRecord.
        Args:
        name: string, (validation train)
        filename : age_train.txt age_val.txt
        directory: string, root path to the data set.
        num_shards: number of tfrecord file.
    """
    filenames, labels = _find_image_files(filename, directory)
    _process_image_files(name, filenames, labels, num_shards)
    unique_labels = set(labels)
    return len(labels), unique_labels

def main(_):
    assert not FLAGS.train_shards % FLAGS.num_threads,("Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert not FLAGS.valid_shards % FLAGS.num_threads, ('Please make the FLAGS.num_threads commensurate with FLAGS.valid_shards')
    print('Saving results to %s' % FLAGS.output_dir)

    if os.path.exists(FLAGS.output_dir) is False:
        print('creating %s' % FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)

    valid, valid_outcomes = _process_dataset('validation', '%s/%s' % (FLAGS.fold_dir, FLAGS.valid_list), FLAGS.data_dir,
                                             FLAGS.valid_shards)
    train, train_outcomes = _process_dataset('train', '%s/%s' % (FLAGS.fold_dir, FLAGS.train_list), FLAGS.data_dir,
                                             FLAGS.train_shards)
    
    if len(valid_outcomes) != len(valid_outcomes | train_outcomes):
        print('Warning: unattested labels in training data [%s]' % (
            ', '.join((valid_outcomes | train_outcomes) - valid_outcomes)))

    output_file = os.path.join(FLAGS.output_dir, 'md.json')

    md = {'num_valid_shards': FLAGS.valid_shards,
          'num_train_shards': FLAGS.train_shards,
          'valid_counts': valid,
          'train_counts': train,
          'timestamp': str(datetime.now()),
          'nlabels': len(train_outcomes)}
    with open(output_file, 'w') as f:
        json.dump(md, f)



if __name__=='__main__':
    tf.app.run()