# 包引用
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim

# 和numpy之间数据转换
	torch.from_numpy()
	data.numpy()  

# 数据类型转换
	images.type(torch.FloatTensor)
	images.float()

# Reshape
	# swap color axis because  
    # numpy image: H x W x C  
    # torch image: C X H X W  
    image = image.transpose((2, 0, 1))  

# 模型定义
卷积层：nn.Conv2d  
全连接层：nn.Linear  
池化层：nn.MaxPool2d  
flatten操作：x = x.view(x.size(0), -1)  
dropout：self.fc_drop = nn.Dropout(p=0.4) 
LSTM:nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers)  
损失函数：criterion = nn.MSELoss()  
优化器：optimizer = optim.Adam(net.parameters(), lr=0.01)  

参考代码： model.py

# 训练网络
训练函数在遍历训练数据集时执行的步骤：  
1. 为训练准备所有输入图像和标签数据  
2. 将输入数据传入网络中（前向传递）  outputs = net(inputs)  
3. 计算损失（预测类别与正确标签差别多大）  loss = criterion(outputs, labels)  
4. 将梯度反向传播到网络参数中（反向传递）  loss.backward()    
5. 更新权重（参数更新）  optimizer.step()     
重复这一流程，直到平均损失足够降低。  

# 模型保存和加载
	#保存模型
	model_dir = 'saved_models/'
	model_name = 'keypoints_model_1.pt'

	# after training, save your model parameters in the dir 'saved_models'
	torch.save(net.state_dict(), model_dir+model_name)

	#加载模型
	# instantiate your Net
	net = Net()
	# load the net parameters by name
	net.load_state_dict(torch.load('saved_models/fashion_net_ex.pt'))
	print(net)


# 可视化
	# 可视化filter
	# Get the weights in the first conv layer
	weights = net.conv1.weight.data
	w = weights.numpy()
	
	# for 10 filters
	fig=plt.figure(figsize=(20, 8))
	columns = 5
	rows = 2
	for i in range(0, columns*rows):
	    fig.add_subplot(rows, columns, i+1)
	    plt.imshow(w[i][0], cmap='gray')
	    
	print('First convolutional layer')
	plt.show()


	# 可视化feature map
	# obtain one batch of testing images
	dataiter = iter(test_loader)
	images, labels = dataiter.next()
	images = images.numpy()
	
	# select an image by index
	idx = 3
	img = np.squeeze(images[idx])
	
	# Use OpenCV's filter2D function 
	# apply a specific set of filter weights (like the one's displayed above) to the test image
	
	import cv2
	plt.imshow(img, cmap='gray')
	
	weights = net.conv1.weight.data
	w = weights.numpy()
	
	# 1. first conv layer
	# for 10 filters
	fig=plt.figure(figsize=(30, 10))
	columns = 5*2
	rows = 2
	for i in range(0, columns*rows):
	    fig.add_subplot(rows, columns, i+1)
	    if ((i%2)==0):
	        plt.imshow(w[int(i/2)][0], cmap='gray')
	    else:
	        c = cv2.filter2D(img, -1, w[int((i-1)/2)][0])
	        plt.imshow(c, cmap='gray')
	plt.show()