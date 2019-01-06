import tensorflow as tf
from ImageCoder import ImageCoder
import os
import mobilenet_v1

slim = tf.contrib.slim

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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

coder = ImageCoder()

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

image_buffer = coder.process_image('1.jpg')

writer = tf.python_io.TFRecordWriter('test.tfrecord')
example = _convert_to_example('1.jpg', image_buffer, int(4),
                                          224, 224)
writer.write(example.SerializeToString())

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['test.tfrecord'],)
_, example_serialized = reader.read(filename_queue)
image_buffer, label_index, fname = parse_example_proto(example_serialized)
# image = image_preprocessing(image_buffer, 224, True)
image = tf.image.decode_jpeg(image_buffer, channels=3)

images = tf.placeholder(tf.float32, [1, 224, 224, 3])
scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=False, weight_decay=0.0)
with slim.arg_scope(scope):
    logits, _ = mobilenet_v1.mobilenet_v1(
        images,
        is_training=False,
        depth_multiplier=1.0,
        num_classes=9)
final_result = tf.nn.softmax(logits,name='final_result')


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# print(sess.run(image_buffer))
# print(sess.run(image))
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./log/')
saver.restore(sess, 'log/model.ckpt-6700')
# print(sess.run(tf.expand_dims(image,0)))
print(sess.run(logits,feed_dict={images:sess.run(tf.expand_dims(image,0))}))
print(sess.run(final_result,feed_dict={images:sess.run(tf.expand_dims(image,0))}))

coord.request_stop()
# Wait for threads to finish.
coord.join(threads)
sess.close()







