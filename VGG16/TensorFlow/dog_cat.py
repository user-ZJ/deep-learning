import argparse
import random

import numpy as np
import sys
import tensorflow as tf
import collections
import re
import hashlib
import os.path
import VGG16


MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
labels = ['cat','dog']

def create_image_lists(image_dir, testing_percentage,validation_percentage):
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None

    # 定义有序字典存储image读取结果
    result = collections.OrderedDict()

    file_glob = os.path.join(image_dir,'train', '*.jpg')
    file_list = tf.gfile.Glob(file_glob)
    if not file_list:
        tf.logging.warning('No files found')
    for label in labels:
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            if label in base_name:
                # 当决定将图像放入哪个数据集时，我们想要忽略文件名里_nohash_之后的所有
                # 数据集的创建者，有办法将有密切变化的图片分组
                hash_name = re.sub(r'_nohash_.*$', '', file_name)
                # 对文件名生成hash值，并根据hash值排序，使得每次运行训练集划分时，分到的数据更稳定
                hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
                percentage_hash = ((int(hash_name_hashed, 16) %
                                    (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                                   (100.0 / MAX_NUM_IMAGES_PER_CLASS))
                if percentage_hash < validation_percentage:
                    validation_images.append(base_name)
                elif percentage_hash < (testing_percentage + validation_percentage):
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)
        result[label] = {
                'training': training_images,
                'testing': testing_images,
                'validation': validation_images,
            }
    return result

def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)



def add_jpeg_decoding(modleInputShape=[None,224,224,3]):
    input_height = modleInputShape[1]
    input_width = modleInputShape[2]
    input_channel = modleInputShape[3]
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    #jpeg_data = tf.gfile.FastGFile(file_name, 'rb').read()
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_channel)
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    # print(decoded_image_4d.eval().shape)
    return jpeg_data,resized_image


def get_image_path(image_lists, label_name, index, cache_dir, category):
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    #sub_dir = label_lists['dir']
    sub_dir='train'
    full_path = os.path.join(cache_dir, sub_dir, base_name)
    return full_path


def get_cache_path(image_lists, label_name, index, cache_dir, category):
    return get_image_path(image_lists, label_name, index, cache_dir,
                          category)+".npy"


def run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor):
    resized_input_image = sess.run(decoded_image_tensor,
                                    {jpeg_data_tensor: image_data})
    return resized_input_image


def create_bottleneck_file(cache_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor):
    tf.logging.info('Creating cache at ' + cache_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    try:
        resized_input_image = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    np.save(cache_path,resized_input_image)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, cache_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor):
    ensure_dir_exists('cache/train')
    cache_path = get_cache_path(image_lists, label_name, index,
                                          cache_dir, category)
    if not os.path.exists(cache_path):
        create_bottleneck_file(cache_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor)
    return np.load(cache_path)



def cache_bottleneck(sess, image_lists, image_dir, cache_dir, jpeg_data_tensor, decoded_image_tensor,
                     resized_input_tensor):
    ensure_dir_exists(cache_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training',  'testing','validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index, image_dir, category,
                    cache_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    # prediction：每个batch中图片预测值
    # evaluation_step：准确率
    return evaluation_step, prediction


def get_random_cached_bottlenecks(sess, image_lists, batch_size, category, cache_dir, image_dir,
                                  jpeg_data_tensor, decoded_image_tensor, resized_input_tensor):
    class_count = len(image_lists.keys())
    caches = []
    ground_truths = []
    filenames = []
    if batch_size >= 0:
        #随机选取batch_size张图片
        for unused_i in range(batch_size):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)
            cache = get_or_create_bottleneck(
                sess, image_lists, label_name, image_index, image_dir, category,
                cache_dir, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor)
            caches.append(cache.reshape((224,224,3)))
            ground_truths.append(label_index)
            filenames.append(image_name)
    else:
        #将所有图片放在一个batch里训练
        pass
    caches = np.array(caches,dtype=np.float32)
    ground_truths = tf.one_hot(ground_truths,class_count,1.0,0.0).eval()
    return caches, ground_truths, filenames


def caculateloss(logits, labels):
    return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits), reduction_indices=[1]))


def accuracy(logits, labels):
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct) * 100.0
    return accuracy


def optimize(loss, learning_rate,my_global_step):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss,global_step=my_global_step)
    return train_step


def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         FLAGS.image_dir +
                         ' - multiple classes are needed for classification.')
        return -1

    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, class_count])
    logits = VGG16.vgg16_net(x,class_count)
    loss = caculateloss(logits, y_)
    acc = accuracy(logits, y_)
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = optimize(loss, FLAGS.learning_rate,my_global_step)

    #with graph.as_default():
    #     (train_step, cross_entropy,bottleneck_input,input_groud_truth) = train(output_tensor)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        #print(tf.trainable_variables())
        load_with_skip('vgg16.npy', sess, ['fc6', 'fc7', 'fc8'])
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding()
        cache_bottleneck(sess, image_lists, FLAGS.image_dir,
                           FLAGS.cache_dir, jpeg_data_tensor,
                           decoded_image_tensor, x)


        # 评估预测准确率
        #evaluation_step, _ = add_evaluation_step(output_tensor, ground_truth_input)

        saver = tf.train.Saver(tf.global_variables())
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)
        validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/validation')
        train_saver = tf.train.Saver()

        for i in range(FLAGS.how_many_training_steps):
            (train_cached_tensor,
             train_ground_truth, _) = get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.train_batch_size, 'training',
                FLAGS.cache_dir, FLAGS.image_dir, jpeg_data_tensor,
                decoded_image_tensor, x)
            _, tra_loss, tra_acc = sess.run([train_step,loss,acc],feed_dict={x:train_cached_tensor, y_:train_ground_truth})
            print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (i, tra_loss, tra_acc))
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            # 训练完成或每完成eval_step_interval各batch训练，打印准确率和交叉熵
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                (validation_cached_tensor,
                 validation_ground_truth, _) = get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.validation_batch_size, 'validation',
                    FLAGS.cache_dir, FLAGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, x)
                val_loss, val_acc = sess.run([loss, acc],
                                             feed_dict={x: validation_cached_tensor, y_: validation_ground_truth})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (i, val_loss, val_acc))
            if i % 2000 == 0 or is_last_step:
                checkpoint_path = os.path.join('model', 'model.ckpt')
                saver.save(sess, checkpoint_path,global_step=i)
















if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='dog_cat',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--cache_dir',
      type=str,
      default='cache',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=10,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=50,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=5,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
        How many images to use in an evaluation batch. This validation set is
        used much more often than the test set, and is an early indicator of how
        accurate the model is during training.
        A value of -1 causes the entire validation set to be used, which leads to
        more stable results across training iterations, but may be slower on large
        training sets.\
        """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

