import tensorflow as tf
import os
from ImageCoder import ImageCoder

def parse_example_proto(example_serialized):
    feature = {
        'image/class/label': tf.FixedLenFeature([],dtype=tf.int64,default_value=-1),
        'image/filename': tf.FixedLenFeature([],dtype=tf.string),
        'image/encoded': tf.FixedLenFeature([],dtype=tf.string),
        'image/height': tf.FixedLenFeature([],dtype=tf.int64),
        'image/width': tf.FixedLenFeature([],dtype=tf.int64)
    }
    features = tf.parse_single_example(example_serialized, feature)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    return features['image/encoded'], label, features['image/filename']

def data_files(data_dir, subset):
    """返回tfrecord文件列表
    Returns:
      tfrecord文件名的列表.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    if subset not in ['train', 'validation']:
        raise ValueError('Invalid subset!')
    tf_record_pattern = os.path.join(data_dir, '%s-*' % subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    print(data_files)
    if not data_files:
      raise ValueError('No files found for data dir %s at %s' % (subset, data_dir))
    return data_files


def image_preprocessing(image_buffer, image_size, train):
    """
    对单张图片进行解码
    :param image_buffer: JPEG encoded string Tensor
    :param image_size:图片大小
    :param train:bool，训练集还是验证集
    :param thread_id:
    :return:
    包含适当缩放图像的3-D float Tensor
    """
    coder = ImageCoder()
    image = coder.preprocess_image(image_buffer,image_size,train)
    return image


def batch_inputs(data_dir, batch_size, image_size, train, num_preprocess_threads=4,
                 num_readers=1, input_queue_memory_factor=8):
    with tf.name_scope('batch_processing'):
        if train:
            # 获取tfrecord文件名列表
            files = data_files(data_dir, 'train')
            # tf.train.string_input_producer 函数会使用提供的文件列表（string_tensor）创建一个输入队列
            # 对files进行混淆
            filename_queue = tf.train.string_input_producer(files,shuffle=True,capacity=16)
        else:
            files = data_files(data_dir, 'validation')
            filename_queue = tf.train.string_input_producer(files,shuffle=False,capacity=1)
        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', num_preprocess_threads)

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        #每一个tfrecord文件大概图片数量
        examples_per_shard = 1024
        #随机混淆图片，需要在良好的混淆和内存占用之间平衡
        # 224 * 224 * 3 * 4byte = 500K
        # examples_per_shard * input_queue_memory_factor * 500K = 4GB
        min_queue_examples = examples_per_shard * input_queue_memory_factor
        if train:
            #创建随机队列
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard + 3 * batch_size,
                dtypes=[tf.string])
        #创建一个tfrecord的reader，填充队列
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))
                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

        images_labels_fnames = []
        for thread_id in range(num_preprocess_threads):
            #解析序列化的示例proto以提取图像和元数据。
            image_buffer, label_index, fname = parse_example_proto(example_serialized)
            #对图片进行解码
            image = image_preprocessing(image_buffer, image_size, train)
            images_labels_fnames.append([image, label_index, fname])
            images, label_index_batch, fnames = tf.train.batch_join(
                images_labels_fnames,
                batch_size=batch_size,
                capacity=2 * num_preprocess_threads * batch_size)
        #在可视化工具中显示训练图像
        #tf.summary.image('images', images, 20)
        return images, label_index_batch, fnames


def distorted_inputs(data_dir, batch_size=128, image_size=224, num_preprocess_threads=4,is_train=False):
    # 产生batch_size张图片
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels, filenames = batch_inputs(
            data_dir, batch_size, image_size, train=is_train,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=1)
    return images, labels, filenames