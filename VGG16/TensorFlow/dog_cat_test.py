import argparse
import random

from gevent import os

import VGG16
import dog_cat
import numpy as np
import sys
import tensorflow as tf


MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


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
                          category) + ".npy"


def get_bottleneck(sess, image_lists, label_name, index, image_dir, category, cache_dir):
    cache_path = get_cache_path(image_lists, label_name, index,
                                cache_dir, category)
    return np.load(cache_path)


def get_random_cached_bottlenecks(sess, image_lists, batch_size, category, cache_dir, image_dir):
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
            cache = get_bottleneck(
                sess, image_lists, label_name, image_index, image_dir, category,
                cache_dir)
            caches.append(cache.reshape((224,224,3)))
            ground_truths.append(label_index)
            filenames.append(image_name)
    else:
        #将所有图片放在一个batch里训练
        pass
    caches = np.array(caches,dtype=np.float32)
    ground_truths = tf.one_hot(ground_truths,class_count,1.0,0.0).eval()
    return caches, ground_truths, filenames


def num_correct_prediction(logits, labels):
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    image_lists = dog_cat.create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
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
    with tf.Session() as sess:
        (test_cached_tensor,
         test_ground_truth, _) = get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.test_batch_size, 'testing',
            FLAGS.cache_dir, FLAGS.image_dir)
        logits = VGG16.vgg16_net(tf.convert_to_tensor(test_cached_tensor), class_count)
        correct = num_correct_prediction(logits, test_ground_truth)
        saver = tf.train.Saver(tf.global_variables())
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state('model')
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            return
        print('\nEvaluating......')
        batch_correct = sess.run(correct)
        print('Total testing samples: %d' % len(test_ground_truth))
        print('Total correct predictions: %d' % batch_correct)
        print('Average accuracy: %.2f%%' % (100 * batch_correct / len(test_ground_truth)))



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
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=100,
      help="""\
        How many images to test on. This test set is only used once, to evaluate
        the final accuracy of the model after training completes.
        A value of -1 causes the entire test set to be used, which leads to more
        stable results across runs.\
        """
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
