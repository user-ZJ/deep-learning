# tf.nn.conv2d
	最底层的函数，其他各种库可以说都是基于这个底层库来进行扩展的

	tf.nn.conv2d(
	    input,
	    filter,
	    strides,
	    padding,
	    use_cudnn_on_gpu=True,
	    data_format='NHWC',
	    dilations=[1, 1, 1, 1],
	    name=None
	)

	filter: A Tensor. Must have the same type as input. A 4-D tensor of shape
			[filter_height, filter_width, in_channels, out_channels]
	strides为卷积时在图像每一维的步长，这是一个一维的向量，长度为4，对应的是在input的4个维度上的步长
	use_cudnn_on_gpu指定是否使用cudnn加速，默认为true
	data_format是用于指定输入的input的格式，默认为NHWC格式
	dilations:每个输入维度的扩张因子

# tf.layers.conv2d
	比tf.nn更高级的库，对tf.nn进行了多方位功能的扩展。用程序员的话来说，就是用tf.nn造的轮子
	这个功能已被弃用。 它将在以后的版本中删除。 更新说明：请改用keras.layers.conv2d。

	tf.layers.conv2d(inputs, filters, kernel_size, strides=(1,1),
                      padding='valid', data_format='channels_last',
                　　　 dilation_rate=(1,1), activation=None, 
                　　　 use_bias=True, kernel_initializer=None, 
                　　　 bias_initializer=init_ops.zeros_initializer(), 
                　　　 kernel_regularizer=None, 
                　　　 bias_regularizer=None, 
                　　　 activity_regularizer=None, trainable=True, 
                　　　 name=None, reuse=None)

	filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution)

# tf.contrib.slim.conv2d
	convolution(inputs,
          num_outputs,
          kernel_size,
          stride=1,
          padding='SAME',
          data_format=None,
          rate=1,
          activation_fn=nn.relu,
          normalizer_fn=None,
          normalizer_params=None,
          weights_initializer=initializers.xavier_initializer(),
          weights_regularizer=None,
          biases_initializer=init_ops.zeros_initializer(),
          biases_regularizer=None,
          reuse=None,
          variables_collections=None,
          outputs_collections=None,
          trainable=True,
          scope=None)

	num_outputs指定卷积核的个数（就是filter的个数）
	kernel_size用于指定卷积核的维度（卷积核的宽度，卷积核的高度）
	activation_fn用于激活函数的指定，默认的为ReLU函数
	normalizer_fn用于指定正则化函数
	normalizer_params用于指定正则化函数的参数
	weights_initializer用于指定权重的初始化程序
	weights_regularizer为权重可选的正则化程序
	biases_initializer用于指定biase的初始化程序
	biases_regularizer: biases可选的正则化程序
	reuse指定是否共享层或者和变量
	variable_collections指定所有变量的集合列表或者字典
	outputs_collections指定输出被添加的集合
	trainable:卷积层的参数是否可被训练
	scope:共享变量所指的variable_scope
	

# tf.keras.conv2d
	如果说tf.layers是轮子，那么keras可以说是汽车。tf.keras是基于tf.layers和tf.nn的高度封装。

# tf.nn.depthwise_conv2d和tf.nn.depthwise_conv2d区别
https://www.cnblogs.com/ranjiewen/p/9278631.html  
