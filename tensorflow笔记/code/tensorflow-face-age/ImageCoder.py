import tensorflow as tf
from preprocessing import preprocessing_factory

RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224

input_mean = 127.5
input_std = 127.5


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string,name='DecodeJPGInput')
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        cropped = tf.image.resize_images(self._decode_jpeg, [RESIZE_HEIGHT, RESIZE_WIDTH])
        cropped = tf.cast(cropped, tf.uint8)
        # decoded_image_as_float = tf.cast(self._decode_jpeg, dtype=tf.float32)
        # decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        # resize_shape = tf.stack([RESIZE_HEIGHT, RESIZE_WIDTH])
        # resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        # resized_image = tf.image.resize_bilinear(decoded_image_4d,resize_shape_as_int)
        #offset_image = tf.subtract(resized_image, input_mean)
        #mul_image = tf.multiply(offset_image, 1.0 / input_std)
        # cropped = tf.image.resize_images(self._decode_jpeg, [RESIZE_HEIGHT, RESIZE_WIDTH])
        # cropped = tf.cast(cropped, tf.uint8)
        # cropped = tf.cast(resized_image,tf.uint8)
        # cropped = tf.squeeze(cropped,squeeze_dims=0)
        self._recoded = tf.image.encode_jpeg(cropped, format='rgb', quality=100)

    def _is_png(self,filename):
        return '.png' in filename

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def resample_jpeg(self, image_data):
        image = self._sess.run(self._recoded,  # self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})

        return image

    def preprocess_image(self,image_data,image_size,is_training=False,scope=None):
        image = tf.image.decode_jpeg(image_data, channels=3)
        image_preprocessing_fn = preprocessing_factory.get_preprocessing('mobilenet_v1',is_training)
        processed_image = image_preprocessing_fn(image,image_size,image_size)
        return processed_image

    def process_image(self,filename):
        if not tf.gfile.Exists(filename):
            raise ValueError('File does not exist %s', filename)
        with tf.gfile.FastGFile(filename, 'rb') as f:
            image_data = f.read()
        if self._is_png(filename):
            print('Converting PNG to JPEG for %s' % filename)
            image_data = self.png_to_jpeg(image_data)
        image_data = self.resample_jpeg(image_data)
        return image_data

