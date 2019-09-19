tf.transpose：获得矩阵的转置，即x.T
tf.matmul:矩阵乘法
tf.diag_part：返回张量的对角线部分
tf.expand_dims：在第axis位置增加一个维度
tf.equal：对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的  
tf.identity：属于tensorflow中的一个ops，跟x = x + 0.0的性质一样，返回一个tensor，受到tf.control_dependencies的约束，所以生效。
tf.squeeze(logits, axis=[1, 2], name='Squeeze')：从张量的形状中移除尺寸为1的维数，所有维度的size已知，axis为可选参数，当维度中有None时，必须指定axis

tf.losses.softmax_cross_entropy：先对网络输出进行softmax,在计算交叉熵，要求label是onehot后的  
tf.losses.sparse_softmax_cross_entropy：先对网络输出进行softmax,在计算交叉熵，要求label是onehot前的