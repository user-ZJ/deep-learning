#dataset 
dataset是tensorflow操作数据的高级api，可以轻松处理大量数据、不同的数据格式以及复杂的转换。  
Dataset：可以轻松处理大量数据、不同的数据格式以及复杂的转换
FixedLengthRecordDataset：来自一个或多个二进制文件的固定长度记录的数据集。  
Iterator：表示迭代数据集的状态  
Options：表示tf.data.Dataset的选项   
TFRecordDataset：数据集，包含来自一个或多个TFRecord文件的记录  
TextLineDataset：数据集，包含来自一个或多个文本文件的行。  

make_initializable_iterator():创建一个tf.data.Iterator来迭代数据集的元素    
make_one_shot_iterator():创建一个tf.data.Iterator来迭代数据集的元素   