import tensorflow as tf
print(tf.__version__)
filepath="result/model.pb"
inp=["Placeholder"]
opt=["final_result"]
converter = tf.contrib.lite.TocoConverter.from_frozen_graph(filepath, inp, opt)
tflite_model=converter.convert()
f = open("result/model.tflite", "wb")
f.write(tflite_model)