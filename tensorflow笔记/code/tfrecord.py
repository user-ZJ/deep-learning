#参考https://blog.csdn.net/dbsdzxq/article/details/79872465
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE=28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
TRAIN_FILE = "train.tfrecords"
VALIDATION_FILE="validation.tfrecords"

# 1.创建数据属性
#生成整数型的属性
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

#生成字符串型的属性
def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_tfrecord(data_set,name):
    '''
        将数据填入到tf.train.Example的协议缓冲区（protocol buffer)中，将协议缓冲区序列
        化为一个字符串，通过tf.python_io.TFRecordWriter写入TFRecords文件
    '''
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples
    if images.shape[0] != num_examples:
        raise ValueError('Imagessize %d does not match label size %d.' \
                         % (images.shape[0], num_examples))
    rows = images.shape[1]  # 28
    cols = images.shape[2]  # 28
    depth = images.shape[3]  # 1 是黑白图像

    dir = "tfrecords"
    if not os.path.exists(dir):
        os.mkdir(dir)
    filename = os.path.join(dir, name + '.tfrecords')
    # 使用下面语句就会将三个文件存储为一个TFRecord文件,当数据量较大时，最好将数据写入多个文件
    # filename="C:/Users/dbsdz/Desktop/TF练习/TFRecord"
    print('Writing', filename)
    # 2.创建tfrecord文件
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()  # 将图像矩阵化为一个字符串

        # 3. 将数据转化为example数据格式
        # 写入协议缓冲区，height、width、depth、label编码成int 64类型，image——raw编码成二进制
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        # 4. 写入tfrecord文件
        writer.write(example.SerializeToString())  # 序列化字符串
    writer.close()

def read_and_decode(filename_queue):     #输入文件名队列
    # 1. 创建tfrecord阅读器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 解析一个example,如果需要解析多个样例，使用parse_example函数
    features = tf.parse_single_example(
        serialized_example,
        # 必须写明feature里面的key的名称
        features={
            # TensorFlow提供两种不同的属性解析方法，一种方法是tf.FixedLenFeature,
            # 这种方法解析的结果为一个Tensor。另一个方法是tf.VarLenFeature,
            # 这种方法得到的解析结果为SparseTensor,用于处理稀疏数据。
            # 这里解析数据的格式需要和上面程序写入数据的格式一致
            'image_raw': tf.FixedLenFeature([], tf.string),  # 图片是string类型
            'label': tf.FixedLenFeature([], tf.int64),  # 标记是int64类型
        })
    # 对于BytesList,要重新进行编码，把string类型的0维Tensor变成uint8类型的一维Tensor
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_PIXELS])
    # tensor("input/DecodeRaw:0",shape=(784,),dtype=uint8)

    # image张量的形状为：tensor("input/sub:0",shape=(784,),dtype=float32)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # 把label从uint8类型转换为int32类性
    # label张量的形状为tensor（“input/cast_1:0",shape=(),dtype=int32)
    label = tf.cast(features['label'], tf.int32)
    return image,label


def read_from_tfrecord(file,batch_size,num_epochs):
    # train：选择输入训练数据/验证数据
    # batch_size:训练的每一批有多少个样本
    # num_epochs:训练次数，过几遍数据，设置为0/None表示永远训练下去

    '''
       返回结果： A tuple (images,labels)
       *images:类型为float，形状为【batch_size,mnist.IMAGE_PIXELS],范围【-0.5，0.5】。
       *label:类型为int32，形状为【batch_size],范围【0，mnist.NUM_CLASSES]
       注意tf.train.QueueRunner必须用tf.train.start_queue_runners()来启动线程
    '''
    # 获取文件路径，即./MNIST_data/train.tfrecords,./MNIST_data/validation.records
    filename = os.path.join('tfrecord',file)
    with tf.name_scope('input'):
        # tf.train.string_input_producer返回一个QueueRunner,里面有一个FIFOQueue
        filename_queue = tf.train.string_input_producer(  # 如果样本量很大，可以分成若干文件，把文件名列表传入
            [filename], num_epochs=num_epochs)
        image, label = read_and_decode(filename_queue)
        # 随机化example,并把它们整合成batch_size大小
        # tf.train.shuffle_batch生成了RandomShuffleQueue,并开启两个线程
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            min_after_dequeue=1000)  # 留下一部分队列，来保证每次有足够的数据做随机打乱
        return images, sparse_labels


mnist = input_data.read_data_sets("MNIST_data/", reshape=False)

_convert_to_tfrecord(mnist.train,'train')
_convert_to_tfrecord(mnist.validation,'validation')
_convert_to_tfrecord(mnist.test,'test')
images,lables=read_from_tfrecord(TRAIN_FILE,50,3)
print(images.shape,lables.shape)