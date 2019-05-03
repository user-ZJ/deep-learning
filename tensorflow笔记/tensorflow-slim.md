# tensorflow slim
TF-Slim 是用于定义、训练和评估复杂模型的tensorflow轻量级库。tf-slim的组件能轻易地与原生tensorflow框架还有其他的框架（例如tf.contrib.learn）进行整合  

## 使用方法
	import tensorflow as tf
	slim = tf.contrib.slim

## TF-sim组成
* arg_scope：提供了一个名为arg_scope的作用域，允许用户为该作用域内的特定操作定义默认参数。  
* data：包括TF-slim的数据集定义dataset，数据提供器（data providers），并行读入器（parallel_reader）和解码器（decoding）等。  
* evaluation：评估模型程序  
* layers：使用tensorflow构建模型的high level layers    
* learning：模型训练程序   
* losses：常用的loss函数  
* metrics：常用评估矩阵  
* nets：常用网络如VGG和AlexNet模型  
* queues：提供一个上下文管理器来方便安全的启动或关闭QueueRunners  
* regularizers：权值正则项   
* variables：提供便捷封装器来创建和操作变量  

## slim中变量定义
TensorFlow中变量分为规则变量（regulra variables）和局部变量（transient variables），在slim中又将规则变量分为模型变量（model variables）和非模型变量：  
**规则变量**：可以通过saver保存在磁盘上    
**局部变量**：只能在一个session期间存在，不能保存在磁盘上  
**模型变量**：在学习阶段被训练或微调，在评估或预测阶段可以从checkpoint中加载。例如使用slim.fully_connected或者 slim.conv2d创建变量。  
**非模型变量**: 在学习或评估过程中使用，在预测阶段并不会起作用。例如global_step是一个在学习和预测阶段使用的变量，但是并不能真正算是模型的一部分。    

模型变量和规则变量创建：  

	# Model Variables
	weights = slim.model_variable('weights',
	                              shape=[10, 10, 3 , 3],
	                              initializer=tf.truncated_normal_initializer(stddev=0.1),
	                              regularizer=slim.l2_regularizer(0.05),
	                              device='/CPU:0')
	model_variables = slim.get_model_variables()
	
	# Regular variables
	my_var = slim.variable('my_var',
	                       shape=[20, 1],
	                       initializer=tf.zeros_initializer())
	regular_variables_and_model_variables = slim.get_variables()  

将自定义网络或变量加入到模型变量管理：  

	my_model_variable = CreateViaCustomCode()

	# Letting TF-Slim know about the additional variable.
	slim.add_model_variable(my_model_variable)  

## 在slim中定义层
### 在tensorflow中定义层
在TensorFlow中定义一个层（如：卷积层、全连接层、BatchNorm层）往往涉及很多操作，如创建一个卷积层，通常由多个底层操作组成：
1.	创建权值（weight）和偏置（bias）变量
2.	将来自上一层的数据和权值进行卷积
3.	将偏置加到卷积结果上
4.	应用激活函数
如果使用tensorflow实现，代码很多，修改起来比较麻烦：


	input = ...
	with tf.name_scope('conv1_1') as scope:
	     kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
	                                           stddev=1e-1), name='weights')
	     conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
	     biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
	                       trainable=True, name='biases')
	     bias = tf.nn.bias_add(conv, biases)
	     conv1 = tf.nn.relu(bias, name=scope)  

### 在slim中定义层
为了降低重复劳动以及减少重复代码，TF-Slim在更抽象的层面上提供了一系列边界操作定义神经网络。例如，对应于上述代码的TF-Slim代码如下：  

	input = ...
	net = slim.conv2d(input, 128, [3, 3], scope='conv1_1')  
TF-slim提供了标准接口用来组建神经网络，包括：  

Layer | TF-Slim
------- | --------
BiasAdd  | [slim.bias_add](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
BatchNorm  | [slim.batch_norm](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
Conv2d | [slim.conv2d](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
Conv2dInPlane | [slim.conv2d_in_plane](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
Conv2dTranspose (Deconv) | [slim.conv2d_transpose](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
FullyConnected | [slim.fully_connected](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
AvgPool2D | [slim.avg_pool2d](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
Dropout| [slim.dropout](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
Flatten | [slim.flatten](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
MaxPool2D | [slim.max_pool2d](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
OneHotEncoding | [slim.one_hot_encoding](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
SeparableConv2 | [slim.separable_conv2d](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
UnitNorm | [slim.unit_norm](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)  

## slim中repeat 、stack和arg_scope
### repeat：提供使用重复或相同操作，以减少代码
普通网络定义：  

	net = ...
	net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
	net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
	net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
	net = slim.max_pool2d(net, [2, 2], scope='pool2')  

使用repeat定义：

	net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
	net = slim.max_pool2d(net, [2, 2], scope='pool2')
slim.repeat不但可以在一行中使用相同的参数，而且还能智能地展开scopes，即每个后续的slim.conv2d调用所对应的scope都会追加下划线及迭代数字。更具体地讲，上面代码的scope分别为’conv3/conv3_1’, ‘conv3/conv3_2’ 和 ‘conv3/conv3_3’  

### stack：提供使用重复或相同操作，以减少代码
普通网络定义：  

	x = slim.conv2d(x, 32, [3, 3], scope='core/core_1')
	x = slim.conv2d(x, 32, [1, 1], scope='core/core_2')
	x = slim.conv2d(x, 64, [3, 3], scope='core/core_3')
	x = slim.conv2d(x, 64, [1, 1], scope='core/core_4')  
使用stack定义：  

	slim.stack(x, slim.conv2d, [(32, [3, 3]), (32, [1, 1]), (64, [3, 3]), (64, [1, 1])], scope='core')
TF-Slim的slim.stack操作允许调用者用不同的参数重复使用相同的操作符。  

### arg_scope: 用于处理网络中存在大量相同的参数
arg_scrope处理前：

	net = slim.conv2d(inputs, 64, [11, 11], 4, padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
	net = slim.conv2d(net, 128, [11, 11], padding='VALID',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
	net = slim.conv2d(net, 256, [11, 11], padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')  
arg_scope处理后：

	with slim.arg_scope([slim.conv2d], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                      weights_regularizer=slim.l2_regularizer(0.0005)):
       net = slim.conv2d(inputs, 64, [11, 11], scope='conv1')
       net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='conv2')
       net = slim.conv2d(net, 256, [11, 11], scope='conv3')  
arg_scope的作用范围内，是定义了指定层的默认参数，若想特别指定某些层的参数，可以重新赋值  
另外，如果除了卷积层，还有其他层，如全连接层，需要按照如下定义，写两个arg_scope即可  

	with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
	   with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
	       net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
	       net = slim.conv2d(net, 256, [5, 5],
	                      weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
	                      scope='conv2')
	       net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc')

### 使用以上方法定义VGG16网络
	def vgg16(inputs):
	    with slim.arg_scope([slim.conv2d, slim.fully_connected],
	                      activation_fn=tf.nn.relu,
	                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
	                      weights_regularizer=slim.l2_regularizer(0.0005)):
	        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
	        net = slim.max_pool2d(net, [2, 2], scope='pool1')
	        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
	        net = slim.max_pool2d(net, [2, 2], scope='pool2')
	        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
	        net = slim.max_pool2d(net, [2, 2], scope='pool3')
	        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
	        net = slim.max_pool2d(net, [2, 2], scope='pool4')
	        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
	        net = slim.max_pool2d(net, [2, 2], scope='pool5')
	        net = slim.fully_connected(net, 4096, scope='fc6')
	        net = slim.dropout(net, 0.5, scope='dropout6')
	        net = slim.fully_connected(net, 4096, scope='fc7')
	        net = slim.dropout(net, 0.5, scope='dropout7')
	        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
	  return net


# 参考
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/README.md


