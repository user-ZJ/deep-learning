from google.protobuf import text_format
from tensorflow.python.platform import gfile
import tensorflow as tf
import os


def convert_pb_to_pbtxt(pbfilename,pbtxtfilename):
    with gfile.FastGFile(pbfilename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', pbtxtfilename, as_text=True)


def convert_pbtxt_to_pb(pbtxtfilename,pbfilename):
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
      filename: The name of a file containing a GraphDef pbtxt (text-formatted
        `tf.GraphDef` protocol buffer data).
    """
    with tf.gfile.FastGFile(pbtxtfilename, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()

        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, './', pbfilename, as_text=False)

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

        # for op in graph.get_operations():
        #     print(op.name, op.values())

def convert_pb_to_tflite(pbfilename,tflitefilename):
    input_node_names = ["input_holder"]
    output_node_names = ["prediction"]
    converter = tf.contrib.lite.TocoConverter.from_frozen_graph(pbfilename, input_node_names, output_node_names)
    tflite_model = converter.convert()
    f = open(tflitefilename, "wb")
    f.write(tflite_model)

freeze_graph('ckpt_dir')
convert_pb_to_pbtxt('tfmodel/train.pb','model-convert/model.pbtxt')
convert_pb_to_pbtxt('tfmodel/train_graph.pb','model-convert/model_graph.pbtxt')
convert_pbtxt_to_pb('tfmodel/train.pbtxt','model-convert/model.pb')
convert_pbtxt_to_pb('tfmodel/train_graph.pbtxt','model-convert/model_graph.pb')
convert_pb_to_tflite('tfmodel/train.pb','tflite/train.tflite')