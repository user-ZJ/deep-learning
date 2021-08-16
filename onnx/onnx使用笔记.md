# onnx使用笔记

## tensorflow转onnx

1. keras模型转onnx

```python 
import tensorflow as tf
import tf2onnx
import onnx
input_signature = [tf.TensorSpec([1, 32], tf.float32, name='input1'),
                   tf.TensorSpec([1, 32], tf.float32, name='input2')]
onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature, opset=12)
onnx.save(onnx_model, "model_sign.onnx")
```

2. savedModel模型转onnx

   ```shell
   python -m tf2onnx.convert --saved-model path/to/savedmodel --output dst/path/model.onnx --opset 13
   ```

3. tflite转onnx

   ```shell
   python -m tf2onnx.convert --saved-model path/to/savedmodel --output dst/path/model.onnx --opset 13
   ```

## onnxsim

onnxsim是简化onnx图的工具

```shell
#!/usr/bin/env python
# coding=utf-8
import onnx
output_path = "encoder.onnx"
from onnxsim import simplify
onnx_model = onnx.load(output_path)  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
print('finished exporting onnx')
```

或使用shell

```shell
# 静态输入优化：
python3 -m onnxsim encoder.onnx encoder1.onnx 
# 动态输入优化
python3 -m onnxsim encoder.onnx encoder1.onnx --dynamic-input-shape  --input-shape 1,120
```

## onnxruntime

1. 使用实例

   ```python 
   #!/usr/bin/env python
   # coding=utf-8
   import onnx
   import onnxruntime
   import numpy as np
   
   sess = onnxruntime.InferenceSession("XXX.onnx")
   
   input_name = sess.get_inputs()[0].name
   input_shape = sess.get_inputs()[0].shape
   input_type = sess.get_inputs()[0].type
   print(input_name)
   output_name = sess.get_outputs()[0].name
   output_shape = sess.get_outputs()[0].shape
   output_type = sess.get_outputs()[0].type
   
   
   x = np.random.random((1,120))
   x = x.astype(np.int64)
   
   #输入为节点名：数据的字典，输出为节点名
   res = sess.run([output_name], {input_name: x})   
   print(res)
   ```

2. 输出onnx中所有节点的运行结果

   ```python 
   #!/usr/bin/env python
   # coding=utf-8
   import onnx
   import onnxruntime as ort
   import numpy as np
   
   model = onnx.load("xxx.onnx")
   
   del model.graph.output[:]  # clear old output
   for node in graph.node:
       for output in node.output:
           model.graph.output.extend([onnx.ValueInfoProto(name=output)])
   
   ort_session = ort.InferenceSession(model.SerializeToString())
   outputs_node = [x.name for x in ort_session.get_outputs()]
   
   x = np.random.random((1,120))
   x = x.astype(np.int64)
   
   outputs = ort_session.run(outputs_node, {input_name: x})  
   print(res)
   ```

3. 对onnx进行裁剪（适合调试运行出错的onnx，将onnx运行出错部分裁剪掉，查看前半部分运行结果查找运行出错原因）

   ```python 
   #!/usr/bin/env python
   # coding=utf-8
   import onnx
   import onnxruntime as ort
   import numpy as np
   
   model = onnx.load("xxx.onnx")
   
   oldnodes = [n for n in model.graph.node]
   newnodes = oldnodes[0:103] # or whatever
   del model.graph.node[:] # clear old nodes
   model.graph.node.extend(newnodes)
   # 裁剪之后需要重新指定输入节点，要不会运行失败。输出节点可以指定为调试的观察节点
   del model.graph.output[:]
   graph.output.extend([onnx.ValueInfoProto(name="70")])
   onnx.save(model, "XXX_split.onnx")
   
   ort_session = ort.InferenceSession(model.SerializeToString())
   outputs_node = [x.name for x in ort_session.get_outputs()]
   
   x = np.random.random((1,120))
   x = x.astype(np.int64)
   
   outputs = ort_session.run(outputs_node, {input_name: x})  
   print(res)
   ```

   