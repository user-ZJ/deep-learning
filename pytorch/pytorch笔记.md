# pytorch安装

https://pytorch.org/get-started/locally/

根据需要选择安装

# 包引用

	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim

# 数据初始化

```python 
x = torch.empty(5, 3)  #创建未初始化的矩阵，矩阵内容随机
x = torch.rand(5, 3)  #随机初始化矩阵
x = torch.rand(5, 3)  #使用高斯分布初始化矩阵
x = torch.zeros(5, 3, dtype=torch.long) #初始化0矩阵
x = torch.tensor([5.5, 3])  #使用数组中的数据初始化矩阵
x = x.new_ones(5, 3, dtype=torch.double)  #初始化1矩阵，并指定为double类型
x = torch.ones(5)
x = torch.randn_like(x, dtype=torch.float)  #初始化同x形状相同的矩阵，并指定type
x.size()  #获取矩阵的shape
```

# 运算

```python
print(x + y)
torch.add(x, y)
result = torch.empty(5, 3)
torch.add(x, y, out=result)  # 指定output
y.add_(x)   #自加，x+y赋值给y
```

# 索引

索引方式和numpy一致

```python
print(x[:, 1])
```



# 和numpy之间数据转换

pytroch

	torch.from_numpy()
	data.numpy()
	
	x = torch.randn(1)
	print(x)
	print(x.item())  # 只能访问单个数据的tensor
	
	a = torch.ones(5)
	b = a.numpy()
	
	a = np.ones(5)
	b = torch.from_numpy(a)

# 数据类型转换
	images.type(torch.FloatTensor)
	images.float()

# DataSet

```python
from torch.utils.data import TensorDataset
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs,shuffle=False, num_workers=2)
for xb,yb in train_dl:
    pred = model(xb)
```



# Reshape

	# swap color axis because  
	# numpy image: H x W x C  
	# torch image: C X H X W  
	image = image.transpose((2, 0, 1))  
	
	x = torch.randn(4, 4)
	y = x.view(16)
	z = x.view(-1, 8)

# 数据预处理

```python
class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(device), y.to(device)
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```



# 计算梯度

```python
x = torch.ones(2, 2, requires_grad=True)
# 或者使用x.requires_grad_(True)
y = x + 2
z = y * y * 3
out = z.mean()
print(z, out)
out.backward()
print(x.grad)

# 去掉梯度计算
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)
    
或者
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
```



# 模型定义

卷积层：nn.Conv2d  
全连接层：nn.Linear  
池化层：nn.MaxPool2d  
embed层：embeds = nn.Embedding(2, 5)   
flatten操作：x = x.view(x.size(0), -1)  
dropout：self.fc_drop = nn.Dropout(p=0.4) 
LSTM:nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers)  
损失函数：criterion = nn.MSELoss()  
优化器：optimizer = optim.Adam(net.parameters(), lr=0.01)  

参考代码： model.py

# [自定义网络层](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html)

# [自定义网络](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html)

# Sequential

```python
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def preprocess(x):
    return x.view(-1, 1, 28, 28)

model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```



# 训练网络

训练函数在遍历训练数据集时执行的步骤：  
1. 为训练准备所有输入图像和标签数据  
2. 将输入数据传入网络中（前向传递）  outputs = net(inputs)  
3. 计算损失（预测类别与正确标签差别多大）  loss = criterion(outputs, labels)  
4. 将梯度反向传播到网络参数中（反向传递）  loss.backward()    
5. 更新权重（参数更新）  optimizer.step()     
重复这一流程，直到平均损失足够降低。  

# 模型保存和加载
```python
#保存模型
model_dir = 'saved_models/'
model_name = 'keypoints_model_1.pt'

# after training, save your model parameters in the dir 'saved_models'
# 只保存模型参数
torch.save(net.state_dict(), model_dir+model_name) 

#加载模型
# instantiate your Net
net = Net()
# load the net parameters by name
net.load_state_dict(torch.load('saved_models/fashion_net_ex.pt'))
print(net)
# 保存整个模型
torch.save(model, path)
model = torch.load(path)

# 加载部分预训练参数
pretrained_dict=torch.load(model_weight)
model_dict=myNet.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
myNet.load_state_dict(model_dict)
```

# 设备选择  
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device)  
images = images.to(device)  
或者  
model.gpu()  
image.gpu()  
或者  
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
或者
y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
```

# [多GPU训练](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)
```



# 预训练模型加载  

	resnet = models.resnet50(pretrained=True)
	for param in resnet.parameters():
	param.requires_grad_(False)
	modules = list(resnet.children())[:-1]
	resnet = nn.Sequential(*modules)
	embed = nn.Linear(resnet.fc.in_features, embed_size)



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
# tensorboard

```python
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
writer.add_image('four_fashion_mnist_images', img_grid)  #image页
writer.add_graph(net, images)   #graph页
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))  #Projector页
writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i) #scalar页
```







Note that we always call `model.train()` before training, and `model.eval()` before inference, because these are used by layers such as `nn.BatchNorm2d` and `nn.Dropout` to ensure appropriate behaviour for these different phases

```py
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)
model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
```

# Reference

https://pytorch.org/docs/stable/index.html