import tensorflow as tf
import mobilenet
from ImageCoder import ImageCoder
import os
from data import distorted_inputs

RESIZE_FINAL = 224
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('class_type', 'age','Classification type (age|gender)')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
flags.DEFINE_integer('num_classes', 9, 'Number of classes to distinguish')
flags.DEFINE_string('filename', '1.jpg','File (Image) or File list (Text/No header TSV) to process')
flags.DEFINE_string('train_dir', './AgeGenderDeepLearning/Folds/tf/test_fold_is_0','Training directory')
flags.DEFINE_integer('num_preprocess_threads', 4,'Number of preprocessing threads')
flags.DEFINE_integer('batch_size', 1,'Batch size')
flags.DEFINE_integer('image_size', 224,'Image size')

FLAGS = flags.FLAGS


def build_model():
    g = tf.Graph()
    with g.as_default():
        label_list = AGE_LIST if FLAGS.class_type == 'age' else GENDER_LIST
        #nlabels = len(label_list)
        images = tf.placeholder(tf.float32, [1, RESIZE_FINAL, RESIZE_FINAL, 3])
        scope = mobilenet.mobilenet_v1_arg_scope(is_training=False, weight_decay=0.0)
        with slim.arg_scope(scope):
            logits, _ = mobilenet.mobilenet_v1(
                images,
                is_training=False,
                depth_multiplier=FLAGS.depth_multiplier,
                num_classes=FLAGS.num_classes)
        if FLAGS.quantize:
            tf.contrib.quantize.create_eval_graph()


        return g,logits,images



def main(_):
    g,logits,images =  build_model()
    with g.as_default():
        final_result = tf.nn.softmax(logits,name='final_result')

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config,graph=g) as sess:

            coder = ImageCoder()
            image = coder.process_image(FLAGS.filename)
            image = coder.preprocess_image(image,FLAGS.image_size,False)
            pp_image = tf.expand_dims(image,0)
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            saver.restore(sess, 'log/model.ckpt-40800')

            graph_def = g.as_graph_def()
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                          output_node_names=['final_result'])
            tf.train.write_graph(constant_graph, 'result/', 'model.pbtxt')
            with tf.gfile.GFile(os.path.join('result/', 'model.pb'), "wb") as f:
                f.write(constant_graph.SerializeToString())

            print(sess.run(logits,feed_dict={images:sess.run(pp_image)}))
            predection = sess.run(final_result,feed_dict={images:sess.run(pp_image)})
            print(predection)

if __name__ == '__main__':
    tf.app.run()