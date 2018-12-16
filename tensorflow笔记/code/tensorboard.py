import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 定义超参数
max_steps = 1000  # 最大迭代次数
learning_rate = 0.001   # 学习率
dropout = 0.9   # dropout时随机保留神经元的比例
data_dir = 'MNIST_data'   # 样本数据存储的路径
log_dir = 'logs'    # 输出日志保存的路径
# 获取手写字体训练集
mnist = input_data.read_data_sets(data_dir, one_hot=True)
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10) # 在日志中存储图片

with tf.variable_scope('lay1',reuse=tf.AUTO_REUSE):
    w1 = tf.get_variable('w1',initializer=tf.random_normal(shape=[784,500]))
    tf.summary.histogram('w1',w1) # 权重数据存入到
    b1 = tf.get_variable('b1',initializer=tf.zeros(shape=[500]))
    tf.summary.histogram('b1',b1)
    model = tf.add(tf.matmul(x,w1),b1)
    model = tf.nn.relu(model)
with tf.variable_scope('lay2',reuse=tf.AUTO_REUSE):
    w2 = tf.get_variable('w2',initializer=tf.random_normal(shape=[500,10]))
    tf.summary.histogram('w2',w2)
    b2 = tf.get_variable('b2',initializer=tf.random_normal(shape=[10]))
    tf.summary.histogram('b2',b2)
    logits = tf.add(tf.matmul(model,w2),b2)
with tf.name_scope('loss'):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits)
    loss = tf.reduce_mean(loss,name='cross_entropy')
    tf.summary.scalar('loss',loss)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)),tf.float32))
    tf.summary.scalar('accuracy',accuracy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# summaries合并
merged = tf.summary.merge_all()
# 定义日志文件
train_writer = tf.summary.FileWriter(log_dir + '/train', tf.get_default_graph())
test_writer = tf.summary.FileWriter(log_dir + '/test')

def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = mnist.train.next_batch(100)
    else:
      xs, ys = mnist.test.images, mnist.test.labels
    return {x: xs, y_: ys}

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(max_steps):
    if i % 10 == 0:  # 记录测试集的summary与accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)  #将输出写入到日志文件
        print('Accuracy at step %s: %s' % (i, acc))
    else:  # 记录训练集的summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()
