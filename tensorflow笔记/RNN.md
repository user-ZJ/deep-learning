# RNN

## API

tf.nn.rnn_cell.LSTMCell：创建单个lstm单元，使用cell.zero_state(batch_size, dtype=tf.float32)初始化（h,c)  num_units隐藏层数  num_proj：投射（projection）操作之后输出的维度，要是为None的话，表示不进行投射操作
tf.nn.rnn_cell.MultiRNNCell：把单层LSTM结合为多层的LSTM
tf.nn.rnn_cell.LSTMStateTuple：
tf.nn.rnn_cell.ResidualWrapper()
tf.nn.rnn_cell.DropoutWrapper
tf.nn.dynamic_rnn():通过指定的RNN Cell来展开计算神经网络.dynamic_rnn是动态生成graph的
tf.nn.bidirectional_dynamic_rnn()：计算双向LSTM网络
tf.sequence_mask()：返回一个mask tensor表示每个序列的前N个位置
tf.boolean_mask()：把boolean类型的mask值应用到tensor上面,可以和numpy里面的tensor[mask] 类比
tf.nn.embedding_lookup()：把一个字或者词映射到对应维度的词向量上面去