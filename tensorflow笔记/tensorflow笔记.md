# tensorflow    
    
## TensorFlow基础知识    
TensorFlow 的设计理念主要体现在以下两个方面：      
1. 将图的定义和图的运行完全分开。因此，TensorFlow 被认为是一个“符号主义”的库      
   符号式计算一般是先定义各种变量，然后建立一个数据流图，在数据流图中规定各个变量之间的计算关系，最后需要对数据流图进行编译，但此时的数据流图还是一个空壳儿，里面没有任何实际数据，只有把需要运算的输入放进去后，才能在整个模型中形成数据流，从而形成输出值。      
2. TensorFlow 中涉及的运算都要放在图中，而图的运行只发生在会话（session）中。开启会话后，就可以用数据去填充节点，进行运算；关闭会话后，就不能进行计算了。因此，会话提供了操作运行和 Tensor 求值的环境。       
    
### 数据流图中的各个要素：      
图中包含输入（input）、塑形（reshape）、Relu 层（Relu layer）、Logit 层（Logit layer）、Softmax、交叉熵（cross entropy）、梯度（gradient）、SGD 训练（SGD Trainer）等      
TensorFlow 的数据流图是由节点（node）和边（edge）组成的有向无环图（directed acycline graph，DAG）      
还有一种特殊边，一般画为虚线边，称为**控制依赖**（control dependency），可以用于控制操作的运行，这被用来确保 happens-before 关系，这类边上没有数据流过，但源节点必须在目的节点开始执行前完成执行。常用代码如下：      
tf.Graph.control_dependencies(control_inputs)    
    
### 数据类型：     
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/dtypes.py    
#### 常量    
constant是TensorFlow的常量节点，通过constant方法创建，其是计算图中的起始节点，是传入数据。      
tf.constant(value=[1,2],dtype=tf.float32,shape=(1,2),name='testconst', verify_shape=False)       
tf.constant() 返回的 tensor 是一个常量 tensor，这个 tensor 的值不会变    
value：初始值，必填，必须是一个张量    
dtype：数据类型，选填，默认为value的数据类型    
shape：数据形状，选填，默认为value的shape，设置时不得比value小，可以比value阶数、维度更高，超过部分按value提供最后一个数字填充    
name：常量名，选填，默认值不重复，根据创建顺序为（Const，Const_1，Const_2.......）    
verify_shape:是否验证value的shape和指定shape相符，若设为True则进行验证，不相符时会抛出异常    
     
#### 变量    
Vatiable是tensorflow的变量节点，通过tf.Variable, tf.get_variable方法创建，并且需要传递初始值。在使用前需要通过tensorflow的初始化方法进行初始化      
tf.Variable 存在于单个 session.run 调用的上下文之外,可以持久性存储。具体 op 允许读取和修改变脸的值。这些修改在多个 tf.Session 之间是可见的，因此对于一个 tf.Variable，多个Session可以看到相同的值      
tf.Variable      
创建一个变量      
tf.get_variable()      
通过变量名获取变量，如果变量不存在则创建该变量，要求指定变量的名称,其他副本将使用此名称访问同一变量，以及在对模型设置检查点和导出模型时指定此变量的值      
name:已创建变量名或新变量名      
shape:数据形状      
dtype:数据类型      
initializer：变量初始化函数      
regularizer：正则化函数，结果会添加到tf.GraphKeys.REGULARIZATION_LOSSES用来计算正则化loss    
trainable:为TRUE时，变量会被添加到GraphKeys.TRAINABLE_VARIABLES中      
变量初始化函数      
1. tf.constant_initializer()也可以简写为tf.Constant()：初始化为常量常量      
2. tf.zeros_initializer()也可以简写为tf.Zeros()：初始化为0      
3. tf.ones_initializer()也可以简写为tf.Ones()：初始化为1      
4. tf.truncated_normal_initializer()或者简写为tf.TruncatedNormal()：生成截断正态分布的随机数      
5. tf.random_normal_initializer()可简写为 tf.RandomNormal()：生成标准正态分布的随机数      
6. tf. random_uniform_initializer()可简写为tf.RandomUniform():生成均匀分布的随机数      
7. tf.uniform_unit_scaling_initializer()可简写为tf.UniformUnitScaling():和均匀分布差不多，只是这个初始化方法不需要指定最小最大值，是通过计算出来的      
8. tf.variance_scaling_initializer()可简写为tf.VarianceScaling():生成截断正态分布或均匀分布的随机数      
9. tf.orthogonal_initializer()简写为tf.Orthogonal():生成正交矩阵的随机数      
10. tf.glorot_uniform_initializer():也称之为Xavier uniform initializer，由一个均匀分布（uniform distribution)来初始化数据      
11. glorot_normal_initializer（）: 也称之为 Xavier normal initializer. 由一个 truncated normal distribution来初始化数据.      
    
在使用变量之前要使用 session.run(tf.global_variables_initializer()) 函数来初始化graph中的变量,变量的状态存储在session里.      
另外还可以通过tf.train.Saver.restore()来初始化变量      
    
#### placeholder    
placeholder是TensorFlow的占位符节点，由placeholder方法创建，其也是一种常量，但是由用户在调用run方法是传递的，也可以将placeholder理解为一种形参。即其不像constant那样直接可以使用，需要用户传递常数值。      
tf.placeholder(dtype=tf.float32, shape=[144, 10], name='X')      
dtype：数据类型，必填，默认为value的数据类型      
shape：数据形状，选填，不填则随传入数据的形状自行变动，可以在多次调用中传入不同形状的数据       
name：常量名，选填，默认值不重复，根据创建顺序为（Placeholder，Placeholder_1，Placeholder_2.......）      
    
### 节点:      
图中的节点又称为算子，它代表一个操作（operation，OP），一般用来表示施加的数学运算，也可以表示数据输入（feed in）的起点以及输出（push out）的终点，或者是读取/写入持久变量（persistent variable）的终点。     
     
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_ops.py      
数学运算操作	Add、Subtract、Multiply、Div、Exp、Log、Greater、Less、Equal      
数组运算操作	Concat、Slice、Split、Constant、Rank、Shape、Shuffle      
矩阵运算操作	MatMul、MatrixInverse、MatrixDeterminant      
有状态的操作	Variable、Assign、AssignAdd      
神经网络构建操作	SoftMax、Sigmoid、ReLU、Convolution2D、MaxPool      
检查点操作	Save、Restore      
队列和同步操作	Enqueue、Dequeue、MutexAcquire、MutexRelease      
控制张量流动的操作	Merge、Switch、Enter、Leave、NextIteration      
误差函数  softmax_cross_entropy_with_logits、sparse_softmax_cross_entropy_with_logits、sigmoid_cross_entropy_with_logits、weighted_cross_entropy_with_logits      
option操作  tf.Operation.name、tf.Operation.type    

tf.identity   将一个tensor赋值给另一个tensor，并增加一个op，eg:y=tf.identity(x) 
    
> tf.stack,tf.concat:将两个张量合并，tf.unstack:是将一个高阶数的张量在某个axis上分解为低阶数的张量      
> tf.expand_dims:增加一个维度；tf.squeeze:从tensor中删除所有大小是1的维度；tf.squeeze(cropped,squeeze_dims=0):删除指定尺寸为1的维度      
> tf.cast：张量数据类型转换      
    
### 图像预处理    
https://blog.csdn.net/lovelyaiq/article/details/78716325    
    
    
### 其他概念    
1. 图：把操作任务描述成有向无环图        
tensorflow中所有的operation都必须定义在graph中。在tensorflow程序中存在一个默认graph，如果没有指定graph，所有operation都保存在默认图中。      
tf.get_default_graph()：获取默认graph      
tf.Graph.init()：创建一个空图      
Graph.as_default()：将某图设置为默认图      
tf.get_default_graph().as_graph_def().node：获取图中所有节点      
tf.reset_default_graph ：移除之前的权重和偏置项      
Graph. get_operation_by_name: 对于给定的名称，返回操作（Operation）      
Graph. get_tensor_by_name: 返回给定名称的tensor      
2. 会话：      
会话（session）提供在图中执行操作的一些方法。一般的模式是，建立会话，此时会生成一张空图，在会话中添加节点和边，形成一张图，然后执行。在调用 Session 对象的 run()方法来执行图时，传入一些 Tensor，这个过程叫填充（feed）；返回的结果类型根据输入的类型而定，这个过程叫取回（fetch）      
会话是图交互的一个桥梁，一个会话可以有多个图，会话可以修改图的结构，也可以往图中注入数据进行计算。因此，会话主要有两个 API 接口：Extend 和 Run。Extend 操作是在 Graph 中添加节点和边，Run 操作是输入计算的节点和填充必要的数据后，进行运算，并输出运算结果。      
**InteractiveSession()创建交互式上下文的 TensorFlow 会话，与常规会话不同的是，交互式会话会成为默认会话，方法（如 tf.Tensor.eval 和 tf.Operation.run）都可以使用该会话来运行操作（OP）**      
详见代码： session.py      
3. 设备：      
设备（device）是指一块可以用来运算并且拥有自己的地址空间的硬件，如 GPU 和 CPU。TensorFlow 为了实现分布式执行操作，充分利用计算资源，可以明确指定操作在哪个设备上执行      
如：with tf.device("/gpu:1")      
4. 变量：     
变量（variable）是一种特殊的数据，它在图中有固定的位置，不像普通张量那样可以流动。使用 tf.Variable()构造函数，这个构造函数需要一个初始值，初始值的形状和类型决定了这个变量的形状和类型      
5. 常量：      
input1 = tf.constant(3.0)      
6. placeholder      
TensorFlow 还提供了填充机制，可以在构建图时使用 tf.placeholder()临时替代任意操作的张量，在调用 Session 对象的 run()方法去执行图时，使用填充数据作为调用的参数，调用结束后，填充数据就消失。     
	input1 = tf.placeholder(tf.float32)    
	input2 = tf.placeholder(tf.float32)    
	output = tf.multiply(input1, input2)    
	with tf.Session() as sess:    
	  print sess.run([output], feed_dict={input1:[7.], input2:[2.]})    
	输出 [array([ 14.], dtype=float32)]      
7. 内核：      
我们知道操作（operation）是对抽象操作（如 matmul 或者 add）的一个统称，而内核（kernel）则是能够运行在特定设备（如 CPU、GPU）上的一种对操作的实现。因此，同一个操作可能会对应多个内核。      
当自定义一个操作时，需要把新操作和内核通过注册的方式添加到系统中。      
    
### 常用 API      
1. 图、操作和张量：      
	tf.Graph 类中包含一系列表示计算的操作对象      
	tf.Graph.init()	创建一个空图      
	tf.Graph.as_default()	将某图设置为默认图，并返回一个上下文管理器。如果不显式添加一个默认图，系统会自动设置一个全局的默认图。所设置的默认图，在模块范围内定义的节点都将默认加入默认图中      
	tf.Graph.device(device_name_or_function)	定义运行图所使用的设备，并返回一个上下文管理器      
	tf.Graph.name_scope(name)	为节点创建层次化的名称，并返回一个上下文管理器      
    tf.reset_default_graph  移除之前的权重和偏置项      
	**tf.get_default_graph().as_graph_def().node：获取图中所有节点**    
     
	tf.Operation 类代表图中的一个节点，用于计算张量数据，该类型由节点构造器（如 tf.matmul()或者 Graph.create_op()）产生      
	tf.Operation.name	操作的名称      
	tf.Operation.type	操作的类型，如MatMul      
	tf.Operation.inputstf.Operation.outputs	操作的输入与输出      
	tf.Operation.control_inputs	操作的依赖      
	tf.Operation.run(feed_dict=None, session=None)	在会话中运行该操作      
	tf.Operation.get_attr(name)	获取操作的属性值       
     
	tf.Tensor 类是操作输出的符号句柄，它不包含操作输出的值，而是提供了一种在 tf.Session 中计算这些值的方法。这样就可以在操作之间构建一个数据流连接，使 TensorFlow 能够执行一个表示大量多步计算的图形。      
	tf.Tensor.dtype	张量的数据类型      
	tf.Tensor.name	张量的名称      
	tf.Tensor.value_index	张量在操作输出中的索引      
	tf.Tensor.graph	张量所在的图      
	tf.Tensor.op	产生该张量的操作      
	tf.Tensor.consumers()	返回使用该张量的操作列表      
	tf.Tensor.eval(feed_dict=None, session=None)	**在会话中求张量的值**，需要使用sess.as_default()或者eval(session=sess)      
	tf.Tensor.get_shape()	返回用于表示张量的形状（维度）的类TensorShape      
	tf.Tensor.set_shape(shape)	更新张量的形状      
	tf.Tensor.device	设置计算该张量的设备      
    
2. 可视化      
	可视化时，需要在程序中给必要的节点添加摘要（summary），摘要会收集该节点的数据，并标记上第几步、时间戳等标识，写入事件文件（event file）中。tf.summary.FileWriter 类用于在目录中创建事件文件，并且向文件中添加摘要和事件，用来在 TensorBoard 中展示。        
	tf.summary.FileWriter(logdir, graph=None, max_queue= 10, flush_secs=120, graph_def=None)	创建 FileWriter 和事件文件，会在 logdir 中创建一个新的事件文件,调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中        
	tf.summary.FileWriter.add_summary(summary, global_step=None)	将摘要添加到事件文件      
	tf.summary.FileWriter.add_event(event)	向事件文件中添加一个事件      
	tf.summary.FileWriter.add_graph(graph, global_step=None, graph_def=None)	向事件文件中添加一个图      
	tf.summary.FileWriter.get_logdir()	获取事件文件的路径      
	tf.summary.FileWriter.flush()	将所有事件都写入磁盘      
	tf.summary.FileWriter.close()	将事件写入磁盘，并关闭文件操作符      
	tf.summary.scalar(name, tensor, collections=None)	输出包含单个标量值的摘要,一般在画loss,accuary时会用到这个函数        
	tf.summary.histogram(name, values, collections=None)	输出包含直方图的摘要,一般用来显示训练过程中变量的分布情况      
	tf.summary.distribution  分布图，一般用于显示weights分布      
	tf.summary.text  可以将文本类型的数据转换为tensor写入summary        
	tf.summary.audio(name, tensor, sample_rate, max_outputs=3, collections=None)	输出包含音频的摘要      
	tf.summary.image(name, tensor, max_outputs=3, collections= None)	输出包含图片的摘要      
	tf.summary.merge(inputs, collections=None, name=None)	选择要保存的信息还需要用到tf.get_collection()函数        
	tf.summary.merge_all()  将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。      
	https://www.cnblogs.com/lyc-seu/p/8647792.html    
	    
    
### 变量作用域    
在 TensorFlow 中有两个作用域（scope），一个是 name_scope，另一个是 variable_scope。variable_scope 主要是给 variable_name 加前缀，也可以给 op_name 加前缀；name_scope 是给 op_name 加前缀。      
示例代码：scope.py      
    
#### variable_scope    
variable_scope 变量作用域机制在 TensorFlow 中主要由两部分组成：      
v = tf.get_variable(name, shape, dtype, initializer) # 通过所给的名字创建或是返回一个变量      
tf.variable_scope(<scope_name>) # 为变量指定命名空间      
    
1.获取变量作用域


	可以直接通过 tf.variable_scope()来获取变量作用域：
	with tf.variable_scope("foo") as foo_scope:
		v = tf.get_variable("v", [1])
		with tf.variable_scope(foo_scope)
			w = tf.get_variable("w", [1])
	如果在开启的一个变量作用域里使用之前预先定义的一个作用域，则会跳过当前变量的作用域，保持预先存在的作用域不变。  
	with tf.variable_scope("foo") as foo_scope:
		assert foo_scope.name == "foo"
		with tf.variable_scope("bar")
			with tf.variable_scope("baz") as other_scope:
				assert other_scope.name == "bar/baz"
			with tf.variable_scope(foo_scope) as foo_scope2:
				assert foo_scope2.name == "foo"  # 保持不变    
    
2．变量作用域的初始化    
变量作用域可以默认携带一个初始化器，在这个作用域中的子作用域或变量都可以继承或者重写父作用域初始化器中的值。      
    
    with tf.variable_scope("foo", initializer=tf.constant_initializer(0.4)):    
    	v = tf.get_variable("v", [1])    
    	assert v.eval() == 0.4  # 被作用域初始化    
    	w = tf.get_variable("w", [1], initializer=tf.constant_initializer(0.3)):    
    	assert w.eval() == 0.3  # 重写初始化器的值    
    	with tf.variable_scope("bar"):    
    		v = tf.get_variable("v", [1])    
    		assert v.eval() == 0.4  # 继承默认的初始化器    
    	with tf.variable_scope("baz", initializer=tf.constant_initializer(0.2)):    
    		v = tf.get_variable("v", [1])    
    		assert v.eval() == 0.2  # 重写父作用域的初始化器的值    
    
对于 opname,在 variable_scope 作用域下的操作，也会被加上前缀：      
    
    with tf.variable_scope("foo"):    
    	x = 1.0 + tf.get_variable("v", [1])    
    assert x.op.name == "foo/add"     
      
variable_scope 主要用在循环神经网络（RNN）的操作中，其中需要大量的共享变量。      
    
#### name_scope    
TensorFlow 中常常会有数以千计的节点，在可视化的过程中很难一下子展示出来，因此用 name_scope 为变量划分范围，在可视化中，这表示在计算图中的一个层级。name_scope 会影响 op_name，不会影响用 get_variable()创建的变量，而会影响通过 Variable()创建的变量      
    
    with tf.variable_scope("foo"):    
    	with tf.name_scope("bar"):    
    		v = tf.get_variable("v", [1])    
    		b = tf.Variable(tf.zeros([1]), name='b')    
    		x = 1.0 + v    
    assert v.name == "foo/v:0"    
    assert b.name == "foo/bar/b:0"    
    assert x.op.name == "foo/bar/add"    
    
### 批标准化    
https://github.com/user-ZJ/deep-learning/blob/master/batch%20normal.md      
    
### 神经元函数及优化方法    
1. 激活函数和dropout      
tf.nn.relu()      
tf.nn.sigmoid()      
tf.nn.tanh()      
tf.nn.elu()      
tf.nn.bias_add()      
tf.nn.crelu()      
tf.nn.relu6()      
tf.nn.softplus()      
tf.nn.softsign()      
tf.nn.dropout()      
2. 卷积函数      
tf.nn.convolution 计算 N 维卷积的和      
tf.nn.conv2d  对一个四维的输入数据input和四维的卷积核filter进行操作，然后对输入数据进行一个二维的卷积操作，最后得到卷积之后的结果      
tf.nn.depthwise_conv2d      
tf.nn.separable_conv2d  利用几个分离的卷积核去做卷积，应用一个二维的卷积核，在每个通道上，以深度 channel_multiplier 进行卷积      
tf.nn.atrous_conv2d  计算 Atrous 卷积，又称孔卷积或者扩张卷积。      
tf.nn.conv2d_transpose  解卷积网络（deconvolutional network）中有时称为“反卷积”，但实际上是 conv2d 的转置，而不是实际的反卷积。      
tf.nn.conv1d  输入是三维，卷积核的维度也是三维，少了一维filter_height      
tf.nn.conv3d  函数用来计算给定五维的输入和过滤器的情况下的三维卷积      
tf.nn.conv3d_transpose      
3. 池化函数      
tf.nn.avg_pool      
tf.nn.max_pool      
tf.nn.max_pool_with_argmax  这个函数的作用是计算池化区域中元素的最大值和该最大值所在的位置，计算位置 argmax 的时候，我们将 input 铺平了进行计算，所以，如果 input = [b, y, x, c]，那么索引位置是(( b * height + y) * width + x) * channels + c；该函数只能在 GPU 下运行，在 CPU 下没有对应的函数实现；返回结果是一个张量组成的元组（output, argmax）      
tf.nn.avg_pool3d  池化后的图片大小可以成非整数倍数缩小      
tf.nn.max_pool3d  池化后的图片大小可以成非整数倍数缩小      
tf.nn.fractional_avg_pool  在三维下的平均池化      
tf.nn.fractional_max_pool  在三维下的最大池化      
tf.nn.pool  ？？？      
4. 分类函数      
tf.nn.sigmoid_cross_entropy_with_logits  如果采用此函数作为损失函数，在神经网络的最后一层不需要进行 sigmoid 运算。        
tf.nn.softmax  Softmax激活  softmax = exp(logits) / reduce_sum(exp(logits), dim)。      
tf.nn.log_softmax  计算 log softmax 激活，也就是 logsoftmax = logits - log(reduce_sum(exp(logits), dim))。      
tf.nn.softmax_cross_entropy_with_logits  计算交叉熵，label需要是onehot之后形式      
tf.nn.sparse_softmax_cross_entropy_with_logits  计算交叉熵，label需要是onehot之前形式      
5. 优化方法      
class tf.train.GradientDescentOptimizer      
class tf.train.AdadeltaOptimizer      
class tf.train.AdagradOptimizer      
class tf.train.AdagradDAOptimizer      
class tf.train.MomentumOptimizer      
class tf.train.AdamOptimizer      
class tf.train.FtrlOptimizer      
class tf.train.RMSPropOptimizer      
    
### 模型的存储与加载      
    
save-restore-model.py      
model-convert.py    
    
**tensorflow模型文件**      
checkpoint：该文件是个文本文件，里面记录了保存的最新的checkpoint文件以及其它checkpoint文件列表。在inference时，可以通过修改这个文件，指定使用哪个model      
model.ckpt.meta：保存的是图结构，meta文件是pb（protocol buffer）格式文件，包含变量、op、集合等      
model.ckpt.index：string-string不可变表，每个键都是张量的名称，其值是序列化的BundleEntryProto。每个BundleEntryProto描述一个张量的metadata：张量的数据包含在哪个文件，该文件的offset，checksum和一些辅助数据等。      
model.ckpt.data-00000-of-00001：它是TensorBundle集合，保存所有变量的值。      
    
#### 保存为ckpt文件    
模型存储主要是建立一个 tf.train.Saver()来保存变量，并且指定保存的位置，一般模型的扩展名为.ckpt。      
saver = tf.train.Saver() #在声明完所有变量后，调用 tf.train.Saver      
non_storable_variable = tf.Variable(777)  # 位于 tf.train.Saver 之后的变量将不会被存储      
saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step) # 存储模型      
保存步骤为：      
1. 定义运算过程      
2. 声明并得到一个 Saver      
3. 通过 Saver.save 保存模型      
    
#### 保存pbtxt或pb文件    
协议缓冲区（Protocol Buffer/简写 Protobufs）是 TF 有效存储和传输数据的常用方式，可以把它当成一个更快的 JSON 格式，当你在存储/传输时需要节省空间/带宽，你可以压缩它。简而言之，可以使用 Protobufs 作为：      
* 一种未压缩的、人性化的文本格式，扩展名为 .pbtxt      
* 一种压缩的、机器友好的二进制格式，扩展名为 .pb 或根本没有扩展名      
    
保存步骤为：      
1. 定义运算过程    
2. 通过 get_default_graph().as_graph_def() 得到当前图的序列化图形表示    
3. 通过 graph_util.convert_variables_to_constants 将相关节点的values固定    
4. 通过 tf.gfile.GFile 进行模型持久化    
    
tf.train.write_graph(sess.graph_def, '/tmp/tfmodel', 'train.pbtxt')    
    
保存图表并保存变量参数:    
      
    方式1：    
    from tensorflow.python.framework import graph_util    
    var_list=tf.global_variables()    
    constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=[var_list[i].name for i in range(len(var_list))]) #将相关节点的value固定，包括output_node_names计算需要的所有节点      
    tf.train.write_graph(constant_graph, './output', 'expert-graph.pbtxt', as_text=False) #不会进行压缩        
	方式2：    
    from tensorflow.python.framework import graph_util    
    var_list=tf.global_variables()    
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=[var_list[i].name for i in range(len(var_list))])    
    with tf.gfile.FastGFile(logdir+'expert-graph.pb', mode='wb') as f:    
    	f.write(constant_graph.SerializeToString())  #序列化输出，压缩后的Protobufs文件    
    
只保留图表：    
    
    方式1：    
	graph_def = tf.get_default_graph().as_graph_def()    
    with gfile.GFile('./output/export.pb', 'wb') as f:    
    	f.write(graph_def.SerializeToString())    
    方式2：    
    tf.train.write_graph(graph_def, './output', 'expert-graph.pb', as_text=False)    
    
free_graph.py制作：      
tensorflow/python/tools/free_graph.py      
先加载模型文件，再从checkpoint文件读取权重数据初始化到模型里的权重变量，再将权重变量转换成权重常量，然后再通过指定的输出节点将没用于输出推理的Op节点从图中剥离掉，再重新保存到指定的文件里。      
例子：      
python tensorflow/python/tools/free_graph.py \    
--input_graph=some_graph_def.pb \ 注意：这里的pb文件是用tf.train.write_graph方法保存的，格式为pbtxt文件，未经过压缩    
--input_checkpoint=model.ckpt.1001 \ 注意：这里若是r12以上的版本，只需给.data-00000....前面的文件名，如：model.ckpt.1001.data-00000-of-00001，只需写model.ckpt.1001      
--output_graph=/tmp/frozen_graph.pb    
--output_node_names=softmax    
    
参数说明：    
input_graph：（必选）模型文件，可以是二进制的pb文件，或文本的meta文件，用input_binary来指定区分      
input_saver：（可选）Saver解析器。保存模型和权限时，Saver也可以自身序列化保存，以便在加载时应用合适的版本。主要用于版本不兼容时使用。可以为空，为空时用当前版本的Saver      
input_binary：（可选）配合input_graph用，为true时，input_graph为二进制，为false时，input_graph为文件。默认False      
input_checkpoint：（必选）检查点数据文件。训练时，给Saver用于保存权重、偏置等变量值。这时用于模型恢复变量值。      
output_node_names：（必选）输出节点的名字，有多个时用逗号分开。用于指定输出节点，将没有在输出线上的其它节点剔除。      
restore_op_name：（可选）从模型恢复节点的名字。升级版中已弃用。默认：save/restore_all      
filename_tensor_name：（可选）已弃用。默认：save/Const:0      
output_graph：（必选）用来保存整合后的模型输出文件。      
clear_devices：（可选），默认True。指定是否清除训练时节点指定的运算设备（如cpu、gpu、tpu。cpu是默认）      
initializer_nodes：（可选）默认空。权限加载后，可通过此参数来指定需要初始化的节点，用逗号分隔多个节点名字。      
variable_names_blacklist：（可先）默认空。变量黑名单，用于指定不用恢复值的变量，用逗号分隔多个变量名字。      
    
如果模型文件是.meta格式的，也就是说用saver.Save方法和checkpoint一起生成的元模型文件，free_graph.py不适用，但可以改造下：      
1、copy free_graph.py为free_graph_meta.py      
2、修改free_graph.py，导入meta_graph：from tensorflow.python.framework import meta_graph      
3、将91行到97行换成：input_graph_def = meta_graph.read_meta_graph_file(input_graph).graph_def       
    
#### 使用ckpt文件    
方法1：在使用模型的时候，必须把模型的结构重新定义一遍，然后载入对应名字的变量的值。    
    
	saver = tf.train.Saver()    
	with tf.Session() as sess:      
		ckpt = tf.train.get_checkpoint_state('./model/')    
		saver.restore(sess, 'model/model.ckpt-10') #加载指定模型    
		saver.restore(sess, ckpt.model_checkpoint_path) #加载checkpoint中记录的模型     
    
方法2：不需重新定义网络结构      
	    
	with tf.Session() as sess:    
		ckpt = tf.train.get_checkpoint_state('./model/')    
		if ckpt and ckpt.model_checkpoint_path:    
			print(ckpt.model_checkpoint_path)    
			#加载指定模型
			saver = tf.train.import_meta_graph('model/model.ckpt-10.meta')		    
			saver.restore(sess, 'model/model.ckpt-10') #加载指定模型    
			#从checkpoint文件中加载指定的模型    
			saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta',    
                                       clear_devices=True)    
			saver.restore(sess, ckpt.model_checkpoint_path) #加载checkpoint中记录的模型    
			#使用graph.get_tensor_by_name()方法来操纵这个保存的模型
			graph = tf.get_default_graph()
            inputs = graph.get_tensor_by_name('inputs:0')
            labels = graph.get_tensor_by_name('labels:0')
			outputs = graph.get_tensor_by_name('xxx/outputs:0')
			outputs = sess.run(outputs,feed_dict={inputs:test_data,labels:test_labels})
			print(outputs)
    
#### 使用pbtxt或pb文件    
    
	# 新建空白图    
	output_graph_def  = tf.Graph()    
	# 空白图列为默认图    
	with output_graph_def.as_default():    
    	# 二进制读取模型文件    
    	with tf.gfile.FastGFile(os.path.join(model_dir,model_name),'rb') as f:    
			# 新建GraphDef文件，用于临时载入模型中图的序列化图形表示    
			graph_def = tf.GraphDef()    
			# GraphDef加载模型中的图    
			graph_def.ParseFromString(f.read())    
			# 在空白图中加载GraphDef中的图    
			tf.import_graph_def(graph_def,name='')    
			    
		with tf.Session() as sess:    
        	# 在图中获取张量需要使用graph.get_tensor_by_name加张量名    
        	# 这里的张量可以直接用于session的run方法求值了    
        	# 补充一个基础知识，形如'conv1'是节点名称，而'conv1:0'是张量名称，表示节点的第一个输出张量    
			init = tf.global_variables_initializer()    
			sess.run(init)    
			image = cv.imread("1.jpg")    
			image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    
			image = cv.resize(image, (224, 224))    
			image = np.expand_dims(image, 0)    
			image = image.astype(np.float32)    
			input_tensor = sess.graph.get_tensor_by_name("input:0")    
			output_tensor = sess.graph.get_tensor_by_name('output:0')    
			print(sess.run(output_tensor,feed_dict={input_tensor:image}))    
			    
    
	#使用pbtxt    
	output_graph_def  = tf.GraphDef()      
	with open('tfmodel/train.pbtxt', 'r') as f:      
    	graph_str = f.read()      
	text_format.Parse(graph_str, output_graph_def)      
	tf.import_graph_def(output_graph_def)      
	    
    
#### 相互转换    
##### pb转pbtxt    
    
	def convert_pb_to_pbtxt(filename):      
		with gfile.FastGFile(filename,'rb') as f:      
    		graph_def = tf.GraphDef()      
    		graph_def.ParseFromString(f.read())      
    		tf.import_graph_def(graph_def, name='')      
    		tf.train.write_graph(graph_def, './', 'protobuf.pbtxt', as_text=True)      
    
##### pbtxt转pb    
    
	def convert_pbtxt_to_pb(filename):    
		"""Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.    
		Args:      
    	filename: The name of a file containing a GraphDef pbtxt (text-formatted    
	      `tf.GraphDef` protocol buffer data)."""      
		with tf.gfile.FastGFile(filename, 'r') as f:      
    	graph_def = tf.GraphDef()      
    	file_content = f.read()       
    	# Merges the human-readable string in `file_content` into `graph_def`.      
    	text_format.Merge(file_content, graph_def)      
    	tf.train.write_graph( graph_def , './' , 'protobuf.pb' , as_text = False )      
    
##### ckpt转pb    
    
	def freeze_graph(ckptmodel_folder):      
	    checkpoint = tf.train.get_checkpoint_state(ckptmodel_folder)  # 检查目录下ckpt文件状态是否可用      
	    input_checkpoint = checkpoint.model_checkpoint_path  # 得ckpt文件路径      
	    output_graph = 'model-convert/ckptmodel.pb'      
	    output_node_names = "prediction"  # 原模型输出操作节点的名字      
	    # 得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.      
	    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',      
	                                       clear_devices=True)      
	    graph = tf.get_default_graph()  # 获得默认的图      
	    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图      
	    with tf.Session() as sess:      
	        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据      
	        # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字      
	        #print("predictions : ", sess.run("prediction:0", feed_dict={"input_holder:0": [10.0]}))      
	        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定      
	            sess,      
	            input_graph_def,      
	            output_node_names.split(",")  # 如果有多个输出节点，以逗号隔开      
	        )      
	        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型      
	            f.write(output_graph_def.SerializeToString())  # 序列化输出      
        	print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点      
    
 	      for op in graph.get_operations():      
            print(op.name, op.values())      
    
##### 生成tflite    

###### tf 1.13之前版本 
	# module 'tensorflow.contrib' has no attribute 'lite'问题，可尝试安装tensorflow1.8以上版本，并且安装pip install --force-reinstall tensorflow-gpu==1.9.0rc1/pip install --force-reinstall tf_nightly_gpu      
	# tflite仅支持ADD, AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, DIV, FLOOR, MUL, RESHAPE, SOFTMAX运算，如果包含其他运算，模型会转换失败        
	import tensorflow as tf      
	filepath="model.pb"      
	inp=["Placeholder"]      
	opt=["MobilenetV1/logits/pool/AvgPool"]      
	converter = tf.contrib.lite.TocoConverter.from_frozen_graph(filepath, inp, opt,input_shapes=None)  #input_shapes参数，当输入存在None维度时，可以将None修改为指定数值，eg：{"foo" : [1, 16, 16, 3]}            
	tflite_model=converter.convert()      
	f = open("model.tflite", "wb")      
	f.write(tflite_model)        
    
	或者使用toco工具进行转换      
	bazel run --config=opt tensorflow/contrib/lite/toco:toco --input_file=$OUTPUT_DIR/tflite_graph.pb --output_file=$OUTPUT_DIR/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='raw_outputs/class_predictions','raw_outputs/box_encodings' --inference_type=FLOAT --allow_custom_ops --default_ranges_min=0 --default_ranges_max=6
	量化
	bazel run -c opt tensorflow/contrib/lite/toco:toco -- --input_file=/root/freeze_graph.pb  --output_file=/root/model_quant.tflite --input_shapes=1,60,60,1 --input_arrays=inputs --output_arrays=rescnn/ln --inference_type=QUANTIZED_UINT8  --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops        
	tflite api:      
	https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/register.cc       
	androidnn api:      
	https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/nnapi/NeuralNetworksTypes.h     

###### tf 1.13及之后版本 

	graph_def_file = args.graph_def_file#'freeze_pruning_graph.pb'
    input_arrays = args.input_arrays.replace(' ', '').split(',')#["inputs"]
    output_arrays = args.output_arrays.replace(' ', '').split(',')#["rescnn/ln"]
	input_shapes = [1,64,64,3]
    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_shapes)
    if quant:
		#方法1，对模型压缩，计算还是使用float32,压缩后模型精度变化不大
        #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
		#方法2，参数转换为int8
        converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
        converter.quantized_input_stats = {input_arrays[0]: (0., 1.)} #均值和方法
        converter.default_ranges_stats = (0, 6) #default_ranges_min和default_ranges_max
        tflite_model = converter.convert()
        f = open(args.tflite_filename, "wb")
    else:
        tflite_model = converter.convert()
        f = open(args.tflite_filename, "wb")
    f.write(tflite_model)
    f.close()

	命令行方式转换：
	ckpt转tflite
	bazel run //tensorflow/lite/python:tflite_convert -- --output_file=foo.tflite  --saved_model_dir=/tmp/saved_model
	pb转tflite
	bazel run //tensorflow/lite/python:tflite_convert -- --output_file=foo.tflite --graph_def_file=frozen_graph.pb --input_arrays=input --output_arrays=MobilenetV1/Predictions/Reshape_1
	keras转tflite
	bazel run //tensorflow/lite/python:tflite_convert -- --output_file=foo.tflite --keras_model_file=/keras_model.h5
	量化
	bazel run //tensorflow/lite/python:tflite_convert -- --output_file=foo.tflite --graph_def_file=some_quantized_graph.pb --inference_type=QUANTIZED_UINT8 --input_shapes=1,28,28,96 --input_arrays=input --output_arrays=outputs --mean_values=128 --std_dev_values=127 --default_ranges_min=0 --default_ranges_max=6   
	

  
    
| TensorFlow Version | Python API |
| ------------------ | ---------- |
| 1.7-1.8 | tf.contrib.lite.toco_convert |
| 1.9-1.11 | tf.contrib.lite.TocoConverter |
| 1.12 | tf.contrib.lite.TFLiteConverter |
| 1.13 | tf.lite.TFLiteConverter |

参考：https://www.tensorflow.org/lite/convert/python_api


    
##### tflite测试    
	import numpy as np      
	import tensorflow as tf      
	import cv2 as cv      
	    
	input_mean = 127.5      
	input_std = 127.5      
	    
	# Load TFLite model and allocate tensors.      
	tflite_model = tf.contrib.lite.Interpreter(model_path="model.tflite")      
	tflite_model.allocate_tensors()      
	    
	# Get input and output tensors.      
	input_details = tflite_model.get_input_details()      
	output_details = tflite_model.get_output_details()      
	    
	# Test model on random input data.      
	input_shape = input_details[0]['shape']      
	    
	image = cv.imread("1.jpg")      
	image = cv.cvtColor(image,cv.COLOR_BGR2RGB)      
	image = cv.resize(image,(224,224))      
	image = np.expand_dims(image,0)      
	image = image.astype(np.float32)      
	image = np.subtract(image, input_mean)      
	image = np.multiply(image, 1.0 / input_std)      
	print(image.shape,image.dtype)	     
	    
	input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # 输入随机数      
	    
	input_data = np.subtract(input_data, input_mean)      
	input_data = np.multiply(input_data, 1.0 / input_std)      
	print(input_data.shape)      
	    
	tflite_model.set_tensor(input_details[0]['index'], image)      
	# tflite_model.set_tensor(input_details[0]['index'], input_data)      
	    
	tflite_model.invoke()      
	output_data = tflite_model.get_tensor(output_details[0]['index'])      
	print("out_class")      
	print(output_data)      
    
    
### 队列和线程    
    
https://www.jianshu.com/p/d063804fb272    
    
#### 队列    
和 TensorFlow 中的其他组件一样，队列（queue）本身也是图中的一个节点，是一种有状态的节点，其他节点，如入队节点（enqueue）和出队节点（dequeue），可以修改它的内容。主要有两种队列：FIFOQueue 和 RandomShuffleQueue，源代码在tensorflow/tensorflow/python/ops/data_flow_ops.py中。      
    
**FIFOQueue**      
FIFOQueue 创建一个先入先出队列。在训练一些语音、文字样本时，使用循环神经网络的网络结构，希望读入的训练样本是有序的，就要用 FIFOQueue。       
    
	import tensorflow as tf       
    
	# 创建一个先入先出队列,初始化队列插入0.1、0.2、0.3三个数字       
	q = tf.FIFOQueue(3, "float")        
	init = q.enqueue_many(([0.1, 0.2, 0.3],))      
	# 定义出队、+1、入队操作      
	x = q.dequeue()      
	y = x + 1      
	q_inc = q.enqueue([y])      
	with tf.Session() as sess:      
		sess.run(init)      
		quelen =  sess.run(q.size())      
		for i in range(2):      
			sess.run(q_inc) # 执行2次操作，队列中的值变为0.3,1.1,1.2      
    
		quelen =  sess.run(q.size())      
		for i in range(quelen):        
    		print (sess.run(q.dequeue())) # 输出队列的值        
    
**RandomShuffleQueue**      
RandomShuffleQueue 创建一个随机队列，在出队列时，是以随机的顺序产生元素的。在训练一些图像样本时，使用CNN的网络结构，希望可以无序地读入训练样本，就要用 RandomShuffleQueue，每次随机产生一个训练样本。      
RandomShuffleQueue 在 TensorFlow 使用异步计算时非常重要。因为 TensorFlow 的会话是支持多线程的，我们可以在主线程里执行训练操作，使用 RandomShuffleQueue 作为训练输入，开多个线程来准备训练样本，将样本压入队列后，主线程会从队列中每次取出 mini-batch 的样本进行训练。      
    
	import tensorflow as tf      
    
	# 创建一个随机队列，队列最大长度为10，出队后最小长度为2      
	q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes="float")      
	sess = tf.Session()      
	for i in range(0, 10): #10次入队      
		sess.run(q.enqueue(i))      
    
	for i in range(0, 8): # 8次出队      
		print(sess.run(q.dequeue()))      
    
当：      
队列长度等于最小值，执行出队操作；    
队列长度等于最大值，执行入队操作。    
程序会发生阻断，卡住不动，如：修改入队次数为12次或修改出队次数为10次，程序会卡住，只有队列满足要求后，才能继续执行。可以通过设置会话在运行时的等待时间来解除阻断：      
    
	import tensorflow as tf      
    
	# 创建一个随机队列，队列最大长度为10，出队后最小长度为2      
	q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes="float")      
	run_options = tf.RunOptions(timeout_in_ms = 10000)  # 等待10秒      
	sess = tf.Session()      
	for i in range(0, 12): #12次入队,会产生阻塞      
	  try:      
	    sess.run(q.enqueue(i),options=run_options)      
	  except tf.errors.DeadlineExceededError:      
	    print('out of range')      
    
	for i in range(0, 8): # 8次出队      
	  print(sess.run(q.dequeue()))      
    
**队列管理器QueueRunner**      
    
	# 创建一个含有队列的图    
	q = tf.FIFOQueue(1000, "float")      
	counter = tf.Variable(0.0)    # 计数器      
	increment_op = tf.assign_add(counter, tf.constant(1.0))    # 操作：给计数器加1      
	enqueue_op = q.enqueue([counter]) # 操作：计数器值加入队列      
	# 创建一个队列管理器 QueueRunner，用这两个操作向队列 q 中添加元素。      
	qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)      
	#主线程      
	with tf.Session() as sess:      
	  sess.run(tf.global_variables_initializer())      
	  enqueue_threads = qr.create_threads(sess, start=True)  # 启动入队线程      
	  #主线程      
	  for i in range(10):      
    	print (sess.run(q.dequeue()))      
    
    
以上程序，输出不是连续的自然数，且线程被阻断（因为加1操作和入队操作不同步，可能加1操作执行了很多次之后，才会进行一次入队操作）。      
QueueRunner 有一个问题就是：入队线程自顾自地执行，在需要的出队操作完成之后，程序没法结束。      
使用 tf.train.Coordinator 来实现线程间的同步，终止其他线程，可以解决以上问题      
    
**线程和协调器(coordinator)**      
使用协调器（coordinator）来管理线程;所有队列管理器被默认加在图的 tf.GraphKeys.QUEUE_RUNNERS 集合中。        
    
	# 创建一个含有队列的图      
	q = tf.FIFOQueue(1000, "float")      
	counter = tf.Variable(0.0)    # 计数器      
	increment_op = tf.assign_add(counter, tf.constant(1.0))    # 操作：给计数器加1      
	enqueue_op = q.enqueue([counter]) # 操作：计数器值加入队列      
	# 创建一个队列管理器 QueueRunner，用这两个操作向队列 q 中添加元素。      
	qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)      
	    
	# 主线程      
	sess = tf.Session()      
	sess.run(tf.global_variables_initializer())      
	# Coordinator：协调器，协调线程间的关系可以视为一种信号量，用来做同步      
	coord = tf.train.Coordinator()      
	# 启动入队线程，协调器是线程的参数      
	enqueue_threads = qr.create_threads(sess, coord = coord,start=True)      
	    
	# 使用方式1    
	# 主线程    
	for i in range(0, 10):      
	  print(sess.run(q.dequeue()))      
	# coord.request_stop()# 通知其他线程关闭      
	# coord.join(enqueue_threads) # join操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回      
	# 方式1在第二次执行会报tf.errors.OutOfRange错误      
	    
	# 使用方式2(推荐使用)      
	coord.request_stop()# 通知其他线程关闭      
	# 主线程      
	for i in range(0, 10):      
	  try:      
	    print(sess.run(q.dequeue()))      
	  except tf.errors.OutOfRangeError:      
	    break      
	coord.join(enqueue_threads) # join操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回      
    
### 数据加载    
TensorFlow 作为符号编程框架，需要先构建数据流图，再读取数据，随后进行模型训练。      
TensorFlow 官方网站给出了以下读取数据3种方法：      
1. 预加载数据（preloaded data）：在 TensorFlow 图中定义常量或变量来保存所有数据。    
2. 填充数据（feeding）：Python 产生数据，再把数据填充后端。    
3. 从文件读取数据（reading from file）：从文件中直接读取，让队列管理器从文件中读取数据。    
    
**预加载数据**      
    
	x1 = tf.constant([2, 3, 4])        
	x2 = tf.constant([4, 0, 1])        
	y = tf.add(x1, x2)      
    
**填充数据**      
    
	import tensorflow as tf      
	# 设计图      
	a1 = tf.placeholder(tf.int16)      
	a2 = tf.placeholder(tf.int16)      
	b = tf.add(x1, x2)      
	# 用Python产生数据      
	li1 = [2, 3, 4]      
	li2 = [4, 0, 1]      
	# 打开一个会话，将数据填充给后端     
	with tf.Session() as sess:      
	  print sess.run(b, feed_dict={a1: li1, a2: li2})      
    
**文件读取数据**      
    
将图片写入tfrecord文件要以int的数值写入，float32可能会导致精度丢失    
    
实例代码：code/tfrecord.py    
写入tfrecord pipeline:tfrecord.md    
    
1. 把样本数据写入 TFRecords 二进制文件；      
* numpy数据通过toString转换为二进制，再通过_bytes_feature写入tfrecord      
* 图片数据通过gfile读取二进制，通过_bytes_feature写入tfrecord      
* 字符串通过str.encode编码，通过_bytes_feature写入tfrecord      
* int数据通过_int64_feature写入tfrecord      
TFRecords 是一种二进制文件，能更好地利用内存，更方便地复制和移动，并且不需要单独的标记文件。具体代码：tensorflow/tensorflow/examples/ how_tos/reading_data/convert_to_records.py      
将数据填入到 tf.train.Example 的协议缓冲区（protocol buffer）中，将协议缓冲区序列化为一个字符串，通过 tf.python_io.TFRecordWriter 写入 TFRecords 文件。      
	    writer = tf.python_io.TFRecordWriter(output_file)    
	    filename = filenames[i]    
	    label = int(labels[i])    
	    image_buffer, height, width = _process_image(filename, coder)    
	    example = _convert_to_example(filename, image_buffer, label,    
	      height, width)    
	    writer.write(example.SerializeToString())    
	        
	    def _convert_to_example(filename, image_buffer, label, height, width):    
	    	example = tf.train.Example(features=tf.train.Features(feature={    
	    		'image/class/label': _int64_feature(label),    
	    		'image/filename': _bytes_feature(str.encode(os.path.basename(filename))),    
	    		'image/encoded': _bytes_feature(image_buffer),    
	    		'image/height': _int64_feature(height),    
	    		'image/width': _int64_feature(width)    
	    	}))    
	    	return example    
	        
	    def _int64_feature(value):    
	      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))    
	        
	    def _bytes_feature(value):    
	      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))    
    
    
2. 从队列中读取      
（1）创建张量，从二进制文件读取一个样本；      
（2）创建张量，从二进制文件随机读取一个 mini-batch；      
（3）把每一批张量传入网络作为输入节点。      
具体代码：tensorflow/tensorflow/examples/ how_tos/reading_data/fully_connected_reader.py      
首先我们定义从文件中读取并解析一个样本;接下来使用 tf.train.shuffle_batch 将前面生成的样本随机化，获得一个最小批次的张量;最后，我们把生成的 batch 张量作为网络的输入，进行训练。      
* 对于以string的方式写入的numpy数据，通过tf.decode_raw将原来编码为字符串类型的变量重新变回来      
      
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
	    reader = tf.TFRecordReader()    
	    filename_queue = tf.train.string_input_producer(['test.tfrecord'],)    
	    _, example_serialized = reader.read(filename_queue)    
	    image_buffer, label_index, fname = parse_example_proto(example_serialized)    
    
### Tensorboard    
示例代码：tensorboard.py      
Tensorboard是Google提供给tensorflow开发者用于在web端可视化训练过程中数据的工具，开发者在训练模型过程中，将需要可视化数据存入自定义日志文件中，然后在指定的web端可视化地展现这些信息。      
Tensorboard可以记录与展示以下数据形式有：      
* GRAPHS：整个模型计算图结构        
* SCALARS：各标量在训练过程中的变化趋势，如accuracy、cross entropy、learning_rate、网络各层的bias和weights等标量      
* IMAGES：输入数据中图片、视频      
* AUDIO：输入数据中音频      
* HISTOGRAM：各变量（如：activations、gradients，weights 等变量）随着训练轮数的数值分布直方图，横轴上越靠前就是越新的轮数的结果      
* DISTRIBUTIONS：数据分布      
* PROJECTOR 模型权重分析，默认使用PCA分析方法，将高维数据投影到3D空间，从而显示数据之间的关系      
Tensorboard的可视化过程，可以分为以下几个步骤：      
1.	创建graph，确定需要获取哪些数据      
2.	在需要获取数据部分，放置summary operations以记录信息      
3.	定义merged = tf.summary.merge_all()，用于执行所有summary节点；      
summary本质是operation，需要在session中run，因此，我们需要特地去运行所有的summary节点。但是呢，一份程序下来可能有超多这样的summary 节点，要手动一个一个去启动自然是及其繁琐的，因此我们可以使用tf.summary.merge_all去将所有summary节点合并成一个节点，只要运行这个节点，就能产生所有我们之前设置的summary data      
4.	使用tf.summary.FileWriter创建本地日志文件，存放summary输出的数据      
5.	通过summary=sess.run(merged)执行summary，获取输出数据      
6.	使用writer.add_summary(summary, i)将输出数据写入到文件      
7.	训练完成后，在命令行输入运行tensorboard的指令，之后打开web端可查看可视化的结果      
    
    
## tensorflow源码    
    
### contrib    
contrib 目录中保存的是将常用的功能封装成的高级 API。但是这个目录并不是官方支持的，很有可能在高级 API 完善后被官方迁移到核心的 TensorFlow 目录中或去掉，现在有一部分包（package）在https://github.com/tensorflow/models 有了更完整的体现。      
**framework：**      
很多函数（如 get_variables、get_global_step）都在这里定义，还有一些废弃或者不推荐（deprecated）的函数      
**layers：**      
这个包主要有 initializers.py、layers.py、optimizers.py、regularizers.py、summaries.py 等文件。      
initializers.py 中主要是做变量初始化的函数。      
layers.py 中有关于层操作和权重偏置变量的函数。      
optimizers.py 中包含损失函数和global_step 张量的优化器操作。      
regularizers.py 中包含带有权重的正则化函数。      
summaries.py 中包含将摘要操作添加到 tf.GraphKeys.SUMMARIES 集合中的函数。      
**learn：**      
这个包是使用 TensorFlow 进行深度学习的高级 API，包括完成训练模型和评估模型、读取批处理数据和队列功能的API封装。      
**rnn：**      
这个包提供了额外的 RNN Cell，也就是对 RNN 隐藏层的各种改进，如 LSTMBlockCell、GRUBlockCell、FusedRNNCell、GridLSTMCell、AttentionCellWrapper 等。      
**seq2seq：**      
这个包提供了建立神经网络 seq2seq 层和损失函数的操作。      
**slim：**       
TensorFlow-Slim （TF-Slim）是一个用于定义、训练和评估 TensorFlow 中的复杂模型的轻量级库。在使用中可以将 TF-Slim 与 TensorFlow 的原生函数和 tf.contrib 中的其他包进行自由组合。TF-Slim 现在已经被逐渐迁移到 TensorFlow 开源的 Models中        
    
### core    
TensorFlow 的原始实现       
    
	├── BUILD    
	├── common_runtime # 公共运行库    
	├── debug    
	├── distributed_runtime # 分布式执行模块，含有 grpc session、grpc worker、 grpc master 等    
	├── example    
	├── framework # 基础功能模块    
	├── graph    
	├── kernels # 一些核心操作在 CPU、CUDA 内核上的实现    
	├── lib # 公共基础库    
	├── ops    
	├── platform # 操作系统实现相关文件    
	├── protobuf # .proto 文件，用于传输时的结构序列化    
	├── public # API 的头文件目录    
	├── user_ops    
	└── util      
    
### examples      
examples 目录中给出了深度学习的一些例子，包括 MNIST、Word2vec、Deepdream、Iris、HDF5 的一些例子，对入门非常有帮助。此外，这个目录中还有 TensorFlow 在 Android 系统上的移动端实现，以及一些扩展为 .ipynb 的文档教程，可以用 jupyter 打开      
    
### g3doc      
TensorFlow 的离线手册      
g3doc/api_docs 目录中的任何内容都是从代码中的注释生成的，不应该直接编辑。脚本 tools/docs/gen_docs.sh 是用来生成 API 文档的。如果无参数调用，它只重新生成 Python API 文档（即操作的文档，包括用 Python 和 C++ 定义的）。如果传递了 -a，运行脚本时还会重新生成 C++ API 的文档。这个脚本必须从 tools/docs 目录调用，如果使用参数 -a，需要安装 doxygen      
    
### python    
激活函数、卷积函数、池化函数、损失函数等实现      
    
### tensorboard      
tensorboard 目录中是实现 TensorFlow 图表可视化工具的代码，代码是基于 Tornado 来实现网页端可视化的。      
绘制出的图形在 HISTOGRAMS 面板中：      
tf.summary.histogram('activations', activations)      
绘制出的图形在 SCALARS 面板中：      
tf.summary.scalar('accuracy', accuracy)      
    
## TensorFlow 程序    
    
code/first_tensorflow.py      
    
TensorFlow 的运行方式分如下4步：    
    
（1）加载数据及定义超参数；（生成及加载数据）    
    
（2）构建网络；（定义网络模型，误差函数，优化器）    
    
（3）训练模型；（使用sess.run执行优化器）    
    
（4）评估模型和进行预测。    
    
**超参数**      
1. 学习率（learning rate）：权重更新步长，学习率设置得越大，训练时间越短，速度越快；而学习率设置得越小，训练得准确度越高。      
2. mini-batch 大小：每批大小决定了权重的更新规则。例如，大小为32时，就是把32个样本的梯度全部计算完，然后求平均值，去更新权重。批次越大训练的速度越快      
3. 正则项系数（regularization parameter，λ）：在较复杂的网络发现出现了明显的过拟合（在训练数据准确率很高但测试数据准确率反而下降），可以考虑增加此项    
    
    
### MNIST    
已经有各种方法被应用在 MNIST 这个训练集上,代码目录在tensorflow/examples/tutorials/mnist/       
    
	mnist_softmax.py：MNIST 采用 Softmax 回归训练。      
	fully_connected_feed.py：MNIST 采用 Feed 数据方式训练。      
	mnist_with_summaries.py：MNIST 使用卷积神经网络（CNN），并且训练过程可视化。      
	mnist_softmax_xla.py.py：MNIST 使用 XLA 框架      
    
### tf.train.batch_join和tf.train.batch    
	batch(tensors,batch_size,num_threads=1,capacity=32,enqueue_many=False,    
		shapes=None,dynamic_pad=False,allow_smaller_final_batch=False,shared_name=None,name=None)    
创建在参数tensors里的张量的batch      
参数tensors可以是一个张量的列表或者字典。函数的返回值将会和参数tensors的类型一致      
这个函数使用队列来实现。一个用于队列的QueueRunner对象被添加当前图的QUEUE_RUNNER的集合中。    
    
	batch_join(tensors_list,batch_size,capacity=32,enqueue_many=False,shapes=None,    
    	dynamic_pad=False,allow_smaller_final_batch=False,shared_name=None,name=None)      
运行张量列表来填充队列，以创建样本的批次。      
tensors_list参数是一个张量元组的列表，或者张量字典的列表。在列表中的每个元素被类似于tf.train.batch()函数中的tensors一样对待。      
这个函数是非确定性的，因为它为每个张量启动了独立的线程      
在不同的线程中入队不同的张量列表。用队列实现——队列的QueueRunner被添加到当前图的QUEUE_RUNNER集合中。      
len(tensors_list)个线程被启动，第i个线程入队来自tensors_list[i]中的张量。tensors_list[i1][j]比如在类型和形状上与tensors_list[i2][j]相匹配，除了当enqueue_many参数为True的时候的第一维。      
    
### tf.ConfigProto()    
tf.ConfigProto()配置Session运行参数和指定GPU设备：      
log_device_placement：tf.ConfigProto()中参数log_device_placement = True ,可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打印出各项操作是在哪个设备上运行的      
allow_soft_placement：如果手动设置的设备不存在或者不可用，允许tf自动选择一个存在并且可用的设备来运行操作      
限制GPU资源使用：      
a. 动态申请显存      
	config = tf.ConfigProto()      
	config.gpu_options.allow_growth = True      
	session = tf.Session(config=config)      
b. 限制GPU使用率      
	config = tf.ConfigProto()      
	config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存      
	session = tf.Session(config=config)      
	或者      
	gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)      
	config=tf.ConfigProto(gpu_options=gpu_options)      
	session = tf.Session(config=config)      
    
    
    
## tensorflow开发流程    
1. 数据预处理，将images->decode->resize->encode->tfrecord，一般1000张图片存放到一个tfrecord文件中    
2. 对tfrecord文件名列表进行混淆，获取tfrecord文件内容，对内容进行混淆，获取batch size images和labels，需要调用tf.train.start_queue_runners(sess=sess)启动线程，获取数据。      
https://blog.csdn.net/dcrmg/article/details/79780331      
3. 定义loss函数，可以添加正则项，获取total loss，用于计算梯度    
4. 定义学习率衰减，部分优化器需要手动设置学习率衰减，tf.train.exponential_decay    
5. 定义优化器，在训练过程中优化权重等参数，tf.train.GradientDescentOptimizer    
6. 创建训练器    
slim.learning.create_train_op：a.计算loss，b.根据梯度更新权重，c.返回loss的值；和slim.learning.train配合使用       
tf.contrib.layers.optimize_loss：Given loss and parameters for optimizer, returns a training op；    
tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)   
7. 训练模型     
slim.learning.train():运行slim.learning.create_train_op创建的对象      
_, loss_value = sess.run([train_op, total_loss]) :运行tf.contrib.layers.optimize_loss创建的对象      
    
**脚本说明**      
save-restore-model.py：模型保存和加载文件      
model-convert.py：模型相互转换文件      
mqueue.py：队列文件    
scope.py：作用域说明文件      
session.py：session使用示例文件      
tensorboard.py：可视化示例文件      
tfrecord.py：tfrecord保存和读取文件      
variables.py：变量使用说明文件      
    
# F&Q    
**1.tf.image.convert_image_dtype**     
图片经过tf.image.decode_jpeg解码后为[0,255]的uint8类型，如果使用image = tf.image.convert_image_dtype(image, dtype=tf.float32)转换float32类型，会被缩放到[0,1]之间，不建议使用。在tf中建议使用preprocessing对数据进行预处理      
    
**2.Session、Graph、op、global_variables_initializer**    
* graph定义了计算方式，是一些加减乘除等运算的组合，类似于一个函数。它本身不会进行任何计算，也不保存任何中间计算结果      
* session用来运行一个graph，或者运行graph的一部分，它类似于一个执行者，给graph灌入输入数据，得到输出，并保存中间的计算结果。同时它也给graph分配计算资源（如内存、显卡等）      
* 一个graph可以供多个session使用，而一个session不一定需要使用graph的全部，可以只使用其中的一部分      
* 运行多个graph需要多个session，而每个session会试图耗尽所有的计算资源，开销太大，不建议这样使用    
* graph之间没有数据通道，要人为通过python/numpy传数据      
* graph是由一系列op构成的      
* global_variables_initializer用来初始化计算图中所有global variable的op    
* global_variables_initializer本身就是一个op，不要加入到graph中才能使用session运行    
如下实例：    

 
	#运行报错    
	g1 = tf.Graph()    
	with g1.as_default():    
		x = tf.Variable(tf.random_normal([1]))    
    	y = tf.Variable(tf.random_normal([1]))    
    	z = tf.add(x, y)    
	sess = tf.Session(graph=g1)    
	sess.run(tf.global_variables_initializer())    
	print(sess.run(z))    
    
	#正确写法为    
	g1 = tf.Graph()    
	with g1.as_default():    
		x = tf.Variable(tf.random_normal([1]))    
    	y = tf.Variable(tf.random_normal([1]))    
    	z = tf.add(x, y)    
		init = tf.global_variables_initializer()    
	sess = tf.Session(graph=g1)    
	sess.run(init)    
	print(sess.run(z))    
    
**3.tensorboard提示无法访问网站**    
使用tensorboard --logdir . 提示TensorBoard 1.11.0 at http://DESKTOP-7AE4QOG:6006 (Press CTRL+C to quit)，但将http://DESKTOP-7AE4QOG:6006输入到谷歌浏览器提示无法访问网站    
解决方法：使用http://localhost:6006访问即可    

**4.sess.run(summary_op)报错**  
> error：You must feed a value for placeholder tensor 'labels' with dtype float and shape  

因为模型的输入为tf.placeholder,sess.run(summary_op)功能是运行所有的summary节点，summary节点中的数据为模型运行过程中的数据，所以会依赖模型的输入数据，而输入数据是placeholder，需要使用feed_dict提供数据。  

解决方法：不建议单独运行sess.run(summary_op)，建议在训练或评估的同时运行sess.run(summary_op)；如：  
train_summary_str,_, train_loss = sess.run([summary_op,train_opt, loss],feed_dict={inputs:train_data,labels:train_labels})

**5.KeyError: "The name 'inputs:0' refers to a Tensor which does not exist. The operation, 'inputs', does not exist in the graph."**  
>bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=tensorflow_model.pb
查看input实际名称为InceptionV1/Logits/Predictions/Reshape_1

  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
