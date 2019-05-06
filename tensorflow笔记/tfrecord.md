# 将原始音频/图像数据写入tfrecord
1. 创建tfrecord目录
2. 创建日志
3. 获取原始数据目录下的所有文件列表和所有标签列表
4. 选取部分标签，在数据量比较大时，选取部分标签的数据方便调试
5. 获取选取标签的所有文件
6. 混淆文件，给文件编号，统计每个标签的文件数
7. 将每个标签的文件按照一定比例划分为训练集、验证集、测试集
8. 提取音频/图像特征，将特征、label等写入examples
9. 创建tfrecord文件，没1000个文件写入一个tfrecord文件
10. 将example写入tfrecord文件

源码：code\tf-pipeline\data_generator\create_tfrecords.py