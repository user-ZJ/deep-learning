# Keras

## Keras 的模型
Keras 的核心数据结构是模型。模型是用来组织网络层的方式。模型有两种:
1. 一种叫 Sequential 模型，Sequential 模型是一系列网络层按顺序构成的栈，是单输入和单输出的，层与层之间只有相邻关系，是最简单的一种模型。  
2. 另一种叫 Model 模型,Model 模型是用来建立更复杂的模型的。  

### Sequential模型
	from keras.models import Sequential
	from keras.layers import Dense, Activation
	model = Sequential()
	model.add(Dense(output_dim=64, input_dim=100))
	model.add(Activation("relu"))
	model.add(Dense(output_dim=10))
	model.add(Activation("softmax"))
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
	loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

### Model模型
	input= Input(shape=(224, 224, 3))
	x = model.add(Conv2D(filters=16,kernel_size=2, activation='relu'))(input)
	x = model.add(MaxPooling2D(pool_size=2, strides=2))(x)
	x = model.add(Conv2D(filters=32,kernel_size=2, activation='relu'))(x)
	x = model.add(MaxPooling2D(pool_size=2, strides=2))(x)
	x = model.add(Conv2D(filters=64,kernel_size=2, activation='relu'))(x)
	x = model.add(MaxPooling2D(pool_size=2, strides=2))(x)
	x = model.add(GlobalAveragePooling2D(dim_ordering='default'))(x)
	x = model.add(Dense(133, activation='softmax'))(x)
	model = Model(input=input, output=x)
	                 
	model.summary()

## 模型的加载及保存
Keras 的 save_model 和 load_model 方法可以将 Keras 模型和权重保存在一个 HDF5 文件中，这里面包括模型的结构、权重、训练的配置（损失函数、优化器）等。如果训练因为某种原因中止，就用这个 HDF5 文件从上次训练的地方重新开始训练。  


如果只是希望保存模型的结构，而不包含其权重及训练的配置（损失函数、优化器），可以使用下面的代码将模型序列化成 json 或者 yaml 文件：  

	json_string = model.to_json()
	yaml_string = model.to_yaml()

保存完成后，还可以手动编辑，并且使用如下语句进行加载：  

	from keras.models import model_from_json
	model = model_from_json(json_string)
	model = model_from_yaml(yaml_string)

如果仅需要保存模型的权重，而不包含模型的结构，可以使用 save_weights 和 load_weights 语句来保存和加载：  

	model.save_weights('my_model_weights.h5')
	model.load_weights('my_model_weights.h5')
