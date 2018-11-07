import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义权重函数
def init_weights(shape):
  return tf.Variable(tf.random_normal(shape, stddev=0.01))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,mnist.test.labels

X = tf.placeholder("float", [None, 784],name="input_holder")
Y = tf.placeholder("float", [None, 10])
# 初始化权重参数
w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

# 定义模型
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
  # 第一个全连接层
  X = tf.nn.dropout(X, p_keep_input)
  h = tf.nn.relu(tf.matmul(X, w_h))

  h = tf.nn.dropout(h, p_keep_hidden)
  # 第二个全连接层
  h2 = tf.nn.relu(tf.matmul(h, w_h2))
  h2 = tf.nn.dropout(h2, p_keep_hidden)

  return tf.matmul(h2, w_o,name="prediction") #输出预测值

# 生成网络模型，得到预测值
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)

# 计数器变量，设置它的 trainable=False，不需要被训练
global_step = tf.Variable(0, name='global_step', trainable=False)

# 在声明完所有变量后，调用 tf.train.Saver
saver = tf.train.Saver()
# 位于 tf.train.Saver 之后的变量将不会被存储
non_storable_variable = tf.Variable(777)

with tf.Session() as sess:
  tf.initialize_all_variables().run()

  start = global_step.eval() # 得到 global_step 的初始值
  print("Start from:", start)

  #加载模型
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print(ckpt.model_checkpoint_path)
    #加载模型参数
    saver.restore(sess, ckpt.model_checkpoint_path)
    # 从这里开始就可以直接使用模型进行预测，或者接着继续训练了

  for i in range(start, 10):
    # 以128作为batch_size
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
      sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
               p_keep_input: 0.8, p_keep_hidden: 0.5})

    global_step.assign(i).eval() # 更新计数器
    # 存储ckpt模型
    saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
  # 2.通过 get_default_graph().as_graph_def() 得到当前图的计算节点信息
  graph_def = tf.get_default_graph().as_graph_def()
  # 保存graph和参数
  # 3.通过 graph_util.convert_variables_to_constants 将相关节点的values固定
  output_graph_def = tf.graph_util.convert_variables_to_constants(sess,graph_def,["prediction"])
  # 4.通过tf.train.write_graph进行持久化
  tf.train.write_graph(output_graph_def, 'tfmodel', 'train.pbtxt')
  # 4.通过 tf.gfile.GFile 进行模型持久化
  with tf.gfile.GFile('tfmodel/train.pb', "wb") as f:
    f.write(output_graph_def.SerializeToString())
  #只保存graph
  tf.train.write_graph(graph_def, 'tfmodel', 'train_graph.pbtxt')
  with tf.gfile.GFile('tfmodel/train_graph.pb', "wb") as f:
    f.write(graph_def.SerializeToString())

  for op in tf.get_default_graph().get_operations():
    print(op.name, op.values())
  for v in tf.get_collection(tf.GraphKeys.VARIABLES):
    print(v)
