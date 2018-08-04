
## 卷积神经网络（Convolutional Neural Network, CNN）

## 项目：实现一个狗品种识别算法App

在这个notebook文件中，有些模板代码已经提供给你，但你还需要实现更多的功能来完成这个项目。除非有明确要求，你无须修改任何已给出的代码。以**'(练习)'**开始的标题表示接下来的代码部分中有你需要实现的功能。这些部分都配有详细的指导，需要实现的部分也会在注释中以'TODO'标出。请仔细阅读所有的提示。

除了实现代码外，你还**需要**回答一些与项目及代码相关的问题。每个需要回答的问题都会以 **'问题 X'** 标记。请仔细阅读每个问题，并且在问题后的 **'回答'** 部分写出完整的答案。我们将根据 你对问题的回答 和 撰写代码实现的功能 来对你提交的项目进行评分。

>**提示：**Code 和 Markdown 区域可通过 **Shift + Enter** 快捷键运行。此外，Markdown可以通过双击进入编辑模式。

项目中显示为_选做_的部分可以帮助你的项目脱颖而出，而不是仅仅达到通过的最低要求。如果你决定追求更高的挑战，请在此 notebook 中完成_选做_部分的代码。

---

### 让我们开始吧
在这个notebook中，你将迈出第一步，来开发可以作为移动端或 Web应用程序一部分的算法。在这个项目的最后，你的程序将能够把用户提供的任何一个图像作为输入。如果可以从图像中检测到一只狗，它会输出对狗品种的预测。如果图像中是一个人脸，它会预测一个与其最相似的狗的种类。下面这张图展示了完成项目后可能的输出结果。（……实际上我们希望每个学生的输出结果不相同！）

![Sample Dog Output](images/sample_dog_output.png)

在现实世界中，你需要拼凑一系列的模型来完成不同的任务；举个例子，用来预测狗种类的算法会与预测人类的算法不同。在做项目的过程中，你可能会遇到不少失败的预测，因为并不存在完美的算法和模型。你最终提交的不完美的解决方案也一定会给你带来一个有趣的学习经验！

### 项目内容

我们将这个notebook分为不同的步骤，你可以使用下面的链接来浏览此notebook。

* [Step 0](#step0): 导入数据集
* [Step 1](#step1): 检测人脸
* [Step 2](#step2): 检测狗狗
* [Step 3](#step3): 从头创建一个CNN来分类狗品种
* [Step 4](#step4): 使用一个CNN来区分狗的品种(使用迁移学习)
* [Step 5](#step5): 建立一个CNN来分类狗的品种（使用迁移学习）
* [Step 6](#step6): 完成你的算法
* [Step 7](#step7): 测试你的算法

在该项目中包含了如下的问题：

* [问题 1](#question1)
* [问题 2](#question2)
* [问题 3](#question3)
* [问题 4](#question4)
* [问题 5](#question5)
* [问题 6](#question6)
* [问题 7](#question7)
* [问题 8](#question8)
* [问题 9](#question9)
* [问题 10](#question10)
* [问题 11](#question11)


---
<a id='step0'></a>
## 步骤 0: 导入数据集

### 导入狗数据集
在下方的代码单元（cell）中，我们导入了一个狗图像的数据集。我们使用 scikit-learn 库中的 `load_files` 函数来获取一些变量：
- `train_files`, `valid_files`, `test_files` - 包含图像的文件路径的numpy数组
- `train_targets`, `valid_targets`, `test_targets` - 包含独热编码分类标签的numpy数组
- `dog_names` - 由字符串构成的与标签相对应的狗的种类


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# 定义函数来加载train，test和validation数据集
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# 加载train，test和validation数据集
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# 加载狗品种列表
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# 打印数据统计描述
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    D:\anaconda\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.
    

### 导入人脸数据集

在下方的代码单元中，我们导入人脸图像数据集，文件所在路径存储在名为 `human_files` 的 numpy 数组。


```python
import random
random.seed(8675309)

# 加载打乱后的人脸数据集的文件名
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# 打印数据集的数据量
print('There are %d total human images.' % len(human_files))
```

    There are 13233 total human images.
    

---
<a id='step1'></a>
## 步骤1：检测人脸
 
我们将使用 OpenCV 中的 [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) 来检测图像中的人脸。OpenCV 提供了很多预训练的人脸检测模型，它们以XML文件保存在 [github](https://github.com/opencv/opencv/tree/master/data/haarcascades)。我们已经下载了其中一个检测模型，并且把它存储在 `haarcascades` 的目录中。

在如下代码单元中，我们将演示如何使用这个检测模型在样本图像中找到人脸。


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# 提取预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# 加载彩色（通道顺序为BGR）图像
img = cv2.imread(human_files[3])

# 将BGR图像进行灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 在图像中找出脸
faces = face_cascade.detectMultiScale(gray)

# 打印图像中检测到的脸的个数
print('Number of faces detected:', len(faces))

# 获取每一个所检测到的脸的识别框
for (x,y,w,h) in faces:
    # 在人脸图像中绘制出识别框
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# 将BGR图像转变为RGB图像以打印
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 展示含有识别框的图像
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1
    


![png](output_5_1.png)


在使用任何一个检测模型之前，将图像转换为灰度图是常用过程。`detectMultiScale` 函数使用储存在 `face_cascade` 中的的数据，对输入的灰度图像进行分类。

在上方的代码中，`faces` 以 numpy 数组的形式，保存了识别到的面部信息。它其中每一行表示一个被检测到的脸，该数据包括如下四个信息：前两个元素  `x`、`y` 代表识别框左上角的 x 和 y 坐标（参照上图，注意 y 坐标的方向和我们默认的方向不同）；后两个元素代表识别框在 x 和 y 轴两个方向延伸的长度 `w` 和 `d`。 

### 写一个人脸识别器

我们可以将这个程序封装为一个函数。该函数的输入为人脸图像的**路径**，当图像中包含人脸时，该函数返回 `True`，反之返回 `False`。该函数定义如下所示。


```python
# 如果img_path路径表示的图像检测到了脸，返回"True" 
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### **【练习】** 评估人脸检测模型


---

<a id='question1'></a>
### __问题 1:__ 

在下方的代码块中，使用 `face_detector` 函数，计算：

- `human_files` 的前100张图像中，能够检测到**人脸**的图像占比多少？
- `dog_files` 的前100张图像中，能够检测到**人脸**的图像占比多少？

理想情况下，人图像中检测到人脸的概率应当为100%，而狗图像中检测到人脸的概率应该为0%。你会发现我们的算法并非完美，但结果仍然是可以接受的。我们从每个数据集中提取前100个图像的文件路径，并将它们存储在`human_files_short`和`dog_files_short`中。


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
## 请不要修改上方代码


## TODO: 基于human_files_short和dog_files_short
## 中的图像测试face_detector的表现
def detect(detector, files):
    return sum([1 if detector(f) else 0 for f in files]) / len(files)

print('human: {:.2f}%'.format(detect(face_detector, human_files_short) * 100))
print('dog: {:.2f}%'.format(detect(face_detector, dog_files_short) * 100))
```

    human: 99.00%
    dog: 12.00%
    

---

<a id='question2'></a>

### __问题 2:__ 

就算法而言，该算法成功与否的关键在于，用户能否提供含有清晰面部特征的人脸图像。
那么你认为，这样的要求在实际使用中对用户合理吗？如果你觉得不合理，你能否想到一个方法，即使图像中并没有清晰的面部特征，也能够检测到人脸？

__回答:__
不合理，现实中，有人脸轮廓即可识别为人脸，可以根据人脸相对身体的相对位置，识别模糊的人脸。


---

<a id='Selection1'></a>
### 选做：

我们建议在你的算法中使用opencv的人脸检测模型去检测人类图像，不过你可以自由地探索其他的方法，尤其是尝试使用深度学习来解决它:)。请用下方的代码单元来设计和测试你的面部监测算法。如果你决定完成这个_选做_任务，你需要报告算法在每一个数据集上的表现。


```python
## (选做) TODO: 报告另一个面部检测算法在LFW数据集上的表现
### 你可以随意使用所需的代码单元数
```

---
<a id='step2'></a>

## 步骤 2: 检测狗狗

在这个部分中，我们使用预训练的 [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) 模型去检测图像中的狗。下方的第一行代码就是下载了 ResNet-50 模型的网络结构参数，以及基于 [ImageNet](http://www.image-net.org/) 数据集的预训练权重。

ImageNet 这目前一个非常流行的数据集，常被用来测试图像分类等计算机视觉任务相关的算法。它包含超过一千万个 URL，每一个都链接到 [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 中所对应的一个物体的图像。任给输入一个图像，该 ResNet-50 模型会返回一个对图像中物体的预测结果。


```python
from keras.applications.resnet50 import ResNet50

# 定义ResNet50模型
ResNet50_model = ResNet50(weights='imagenet')
```

### 数据预处理

- 在使用 TensorFlow 作为后端的时候，在 Keras 中，CNN 的输入是一个4维数组（也被称作4维张量），它的各维度尺寸为 `(nb_samples, rows, columns, channels)`。其中 `nb_samples` 表示图像（或者样本）的总数，`rows`, `columns`, 和 `channels` 分别表示图像的行数、列数和通道数。


- 下方的 `path_to_tensor` 函数实现如下将彩色图像的字符串型的文件路径作为输入，返回一个4维张量，作为 Keras CNN 输入。因为我们的输入图像是彩色图像，因此它们具有三个通道（ `channels` 为 `3`）。
    1. 该函数首先读取一张图像，然后将其缩放为 224×224 的图像。
    2. 随后，该图像被调整为具有4个维度的张量。
    3. 对于任一输入图像，最后返回的张量的维度是：`(1, 224, 224, 3)`。


- `paths_to_tensor` 函数将图像路径的字符串组成的 numpy 数组作为输入，并返回一个4维张量，各维度尺寸为 `(nb_samples, 224, 224, 3)`。 在这里，`nb_samples`是提供的图像路径的数据中的样本数量或图像数量。你也可以将 `nb_samples` 理解为数据集中3维张量的个数（每个3维张量表示一个不同的图像。


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### 基于 ResNet-50 架构进行预测

对于通过上述步骤得到的四维张量，在把它们输入到 ResNet-50 网络、或 Keras 中其他类似的预训练模型之前，还需要进行一些额外的处理：
1. 首先，这些图像的通道顺序为 RGB，我们需要重排他们的通道顺序为 BGR。
2. 其次，预训练模型的输入都进行了额外的归一化过程。因此我们在这里也要对这些张量进行归一化，即对所有图像所有像素都减去像素均值 `[103.939, 116.779, 123.68]`（以 RGB 模式表示，根据所有的 ImageNet 图像算出）。

导入的 `preprocess_input` 函数实现了这些功能。如果你对此很感兴趣，可以在 [这里](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py) 查看 `preprocess_input`的代码。


在实现了图像处理的部分之后，我们就可以使用模型来进行预测。这一步通过 `predict` 方法来实现，它返回一个向量，向量的第 i 个元素表示该图像属于第 i 个 ImageNet 类别的概率。这通过如下的 `ResNet50_predict_labels` 函数实现。

通过对预测出的向量取用 argmax 函数（找到有最大概率值的下标序号），我们可以得到一个整数，即模型预测到的物体的类别。进而根据这个 [清单](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)，我们能够知道这具体是哪个品种的狗狗。



```python
from keras.applications.resnet50 import preprocess_input, decode_predictions
def ResNet50_predict_labels(img_path):
    # 返回img_path路径的图像的预测向量
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### 完成狗检测模型


在研究该 [清单](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 的时候，你会注意到，狗类别对应的序号为151-268。因此，在检查预训练模型判断图像是否包含狗的时候，我们只需要检查如上的 `ResNet50_predict_labels` 函数是否返回一个介于151和268之间（包含区间端点）的值。

我们通过这些想法来完成下方的 `dog_detector` 函数，如果从图像中检测到狗就返回 `True`，否则返回 `False`。


```python
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### 【作业】评估狗狗检测模型

---

<a id='question3'></a>
### __问题 3:__ 

在下方的代码块中，使用 `dog_detector` 函数，计算：

- `human_files_short`中图像检测到狗狗的百分比？
- `dog_files_short`中图像检测到狗狗的百分比？


```python
### TODO: 测试dog_detector函数在human_files_short和dog_files_short的表现

print('human: {:.2f}%'.format(detect(dog_detector, human_files_short) * 100))
print('dog: {:.2f}%'.format(detect(dog_detector, dog_files_short) * 100))
```

    human: 1.00%
    dog: 100.00%
    

---

<a id='step3'></a>

## 步骤 3: 从头开始创建一个CNN来分类狗品种


现在我们已经实现了一个函数，能够在图像中识别人类及狗狗。但我们需要更进一步的方法，来对狗的类别进行识别。在这一步中，你需要实现一个卷积神经网络来对狗的品种进行分类。你需要__从头实现__你的卷积神经网络（在这一阶段，你还不能使用迁移学习），并且你需要达到超过1%的测试集准确率。在本项目的步骤五种，你还有机会使用迁移学习来实现一个准确率大大提高的模型。

在添加卷积层的时候，注意不要加上太多的（可训练的）层。更多的参数意味着更长的训练时间，也就是说你更可能需要一个 GPU 来加速训练过程。万幸的是，Keras 提供了能够轻松预测每次迭代（epoch）花费时间所需的函数。你可以据此推断你算法所需的训练时间。

值得注意的是，对狗的图像进行分类是一项极具挑战性的任务。因为即便是一个正常人，也很难区分布列塔尼犬和威尔士史宾格犬。


布列塔尼犬（Brittany） | 威尔士史宾格犬（Welsh Springer Spaniel）
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

不难发现其他的狗品种会有很小的类间差别（比如金毛寻回犬和美国水猎犬）。


金毛寻回犬（Curly-Coated Retriever） | 美国水猎犬（American Water Spaniel）
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">

同样，拉布拉多犬（labradors）有黄色、棕色和黑色这三种。那么你设计的基于视觉的算法将不得不克服这种较高的类间差别，以达到能够将这些不同颜色的同类狗分到同一个品种中。

黄色拉布拉多犬（Yellow Labrador） | 棕色拉布拉多犬（Chocolate Labrador） | 黑色拉布拉多犬（Black Labrador）
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

我们也提到了随机分类将得到一个非常低的结果：不考虑品种略有失衡的影响，随机猜测到正确品种的概率是1/133，相对应的准确率是低于1%的。

请记住，在深度学习领域，实践远远高于理论。大量尝试不同的框架吧，相信你的直觉！当然，玩得开心！


### 数据预处理


通过对每张图像的像素值除以255，我们对图像实现了归一化处理。


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|█████████████████████████████████████████████████████████████████████████████| 6680/6680 [00:47<00:00, 139.21it/s]
    100%|███████████████████████████████████████████████████████████████████████████████| 835/835 [00:06<00:00, 133.71it/s]
    100%|███████████████████████████████████████████████████████████████████████████████| 836/836 [00:05<00:00, 146.23it/s]
    

### 【练习】模型架构


创建一个卷积神经网络来对狗品种进行分类。在你代码块的最后，执行 `model.summary()` 来输出你模型的总结信息。
    
我们已经帮你导入了一些所需的 Python 库，如有需要你可以自行导入。如果你在过程中遇到了困难，如下是给你的一点小提示——该模型能够在5个 epoch 内取得超过1%的测试准确率，并且能在CPU上很快地训练。

![Sample CNN](images/sample_cnn.png)

---

<a id='question4'></a>  

### __问题 4:__ 

在下方的代码块中尝试使用 Keras 搭建卷积网络的架构，并回答相关的问题。

1. 你可以尝试自己搭建一个卷积网络的模型，那么你需要回答你搭建卷积网络的具体步骤（用了哪些层）以及为什么这样搭建。
2. 你也可以根据上图提示的步骤搭建卷积网络，那么请说明为何如上的架构能够在该问题上取得很好的表现。

__回答:__ 
问题2：
根据提示，搭建卷积神经网络，最后一层采用了全局平均池化，相对于采用全链接层，大大减低了参数个数，减小了过拟合风险；全局平均池化对每个特征图一整张图片进行全局均值池化，这样每张特征图都可以得到一个输出，对每个特征图一整张图片进行全局均值池化，这样每张特征图都可以得到一个输出，让卷积网络结构更简单，且能保证特征图与类别之间的一致性


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: 定义你的网络架构
model.add(Conv2D(filters=16,kernel_size=2, activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=32,kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(GlobalAveragePooling2D(dim_ordering='default'))
model.add(Dense(133, activation='softmax'))
                 
model.summary()
```

    D:\anaconda\lib\site-packages\ipykernel\__main__.py:14: UserWarning: Update your `GlobalAveragePooling2D` call to the Keras 2 API: `GlobalAveragePooling2D(data_format=None)`
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 223, 223, 16)      208       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 111, 111, 16)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 110, 110, 32)      2080      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 55, 55, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 54, 54, 64)        8256      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 27, 27, 64)        0         
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 64)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 133)               8645      
    =================================================================
    Total params: 19,189
    Trainable params: 19,189
    Non-trainable params: 0
    _________________________________________________________________
    


```python
## 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 【练习】训练模型


---

<a id='question5'></a>  

### __问题 5:__ 

在下方代码单元训练模型。使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。

可选题：你也可以对训练集进行 [数据增强](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)，来优化模型的表现。




```python
from keras.callbacks import ModelCheckpoint  

### TODO: 设置训练模型的epochs的数量

epochs = 50

### 不要修改下方代码

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/50
    6680/6680 [==============================] - 28s 4ms/step - loss: 4.8831 - acc: 0.0087 - val_loss: 4.8682 - val_acc: 0.0108
    
    Epoch 00001: val_loss improved from inf to 4.86819, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 2/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.8644 - acc: 0.0114 - val_loss: 4.8491 - val_acc: 0.0192
    
    Epoch 00002: val_loss improved from 4.86819 to 4.84913, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 3/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.8292 - acc: 0.0183 - val_loss: 4.8403 - val_acc: 0.0168
    
    Epoch 00003: val_loss improved from 4.84913 to 4.84032, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 4/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.7748 - acc: 0.0204 - val_loss: 4.7696 - val_acc: 0.0108
    
    Epoch 00004: val_loss improved from 4.84032 to 4.76960, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 5/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.7376 - acc: 0.0247 - val_loss: 4.7489 - val_acc: 0.0168
    
    Epoch 00005: val_loss improved from 4.76960 to 4.74888, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 6/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.7107 - acc: 0.0257 - val_loss: 4.7378 - val_acc: 0.0323
    
    Epoch 00006: val_loss improved from 4.74888 to 4.73780, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 7/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.6843 - acc: 0.0283 - val_loss: 4.7194 - val_acc: 0.0287
    
    Epoch 00007: val_loss improved from 4.73780 to 4.71936, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 8/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.6585 - acc: 0.0334 - val_loss: 4.7063 - val_acc: 0.0323
    
    Epoch 00008: val_loss improved from 4.71936 to 4.70633, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 9/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.6366 - acc: 0.0343 - val_loss: 4.7090 - val_acc: 0.0359
    
    Epoch 00009: val_loss did not improve from 4.70633
    Epoch 10/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.6137 - acc: 0.0383 - val_loss: 4.6843 - val_acc: 0.0275
    
    Epoch 00010: val_loss improved from 4.70633 to 4.68432, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 11/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.5928 - acc: 0.0392 - val_loss: 4.6548 - val_acc: 0.0311
    
    Epoch 00011: val_loss improved from 4.68432 to 4.65481, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 12/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.5721 - acc: 0.0424 - val_loss: 4.6428 - val_acc: 0.0299
    
    Epoch 00012: val_loss improved from 4.65481 to 4.64282, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 13/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.5481 - acc: 0.0461 - val_loss: 4.6379 - val_acc: 0.0216
    
    Epoch 00013: val_loss improved from 4.64282 to 4.63791, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 14/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.5248 - acc: 0.0499 - val_loss: 4.6089 - val_acc: 0.0311
    
    Epoch 00014: val_loss improved from 4.63791 to 4.60886, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 15/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.4993 - acc: 0.0521 - val_loss: 4.5720 - val_acc: 0.0371
    
    Epoch 00015: val_loss improved from 4.60886 to 4.57197, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 16/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.4731 - acc: 0.0552 - val_loss: 4.5699 - val_acc: 0.0455
    
    Epoch 00016: val_loss improved from 4.57197 to 4.56994, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 17/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.4432 - acc: 0.0603 - val_loss: 4.5200 - val_acc: 0.0383
    
    Epoch 00017: val_loss improved from 4.56994 to 4.51997, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 18/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.4128 - acc: 0.0597 - val_loss: 4.5118 - val_acc: 0.0419
    
    Epoch 00018: val_loss improved from 4.51997 to 4.51184, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 19/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.3861 - acc: 0.0639 - val_loss: 4.6340 - val_acc: 0.0443
    
    Epoch 00019: val_loss did not improve from 4.51184
    Epoch 20/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.3650 - acc: 0.0665 - val_loss: 4.4748 - val_acc: 0.0539
    
    Epoch 00020: val_loss improved from 4.51184 to 4.47484, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 21/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.3391 - acc: 0.0669 - val_loss: 4.4511 - val_acc: 0.0443
    
    Epoch 00021: val_loss improved from 4.47484 to 4.45109, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 22/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.3202 - acc: 0.0713 - val_loss: 4.4341 - val_acc: 0.0479
    
    Epoch 00022: val_loss improved from 4.45109 to 4.43412, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 23/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.2956 - acc: 0.0719 - val_loss: 4.4321 - val_acc: 0.0623
    
    Epoch 00023: val_loss improved from 4.43412 to 4.43207, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 24/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.2690 - acc: 0.0771 - val_loss: 4.4489 - val_acc: 0.0587
    
    Epoch 00024: val_loss did not improve from 4.43207
    Epoch 25/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.2496 - acc: 0.0759 - val_loss: 4.4082 - val_acc: 0.0479
    
    Epoch 00025: val_loss improved from 4.43207 to 4.40818, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 26/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.2313 - acc: 0.0810 - val_loss: 4.4243 - val_acc: 0.0467
    
    Epoch 00026: val_loss did not improve from 4.40818
    Epoch 27/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.2136 - acc: 0.0823 - val_loss: 4.4763 - val_acc: 0.0491
    
    Epoch 00027: val_loss did not improve from 4.40818
    Epoch 28/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.1976 - acc: 0.0816 - val_loss: 4.3337 - val_acc: 0.0790
    
    Epoch 00028: val_loss improved from 4.40818 to 4.33372, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 29/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.1770 - acc: 0.0870 - val_loss: 4.4365 - val_acc: 0.0599
    
    Epoch 00029: val_loss did not improve from 4.33372
    Epoch 30/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.1565 - acc: 0.0891 - val_loss: 4.3247 - val_acc: 0.0707
    
    Epoch 00030: val_loss improved from 4.33372 to 4.32474, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 31/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.1432 - acc: 0.0916 - val_loss: 4.3253 - val_acc: 0.0659
    
    Epoch 00031: val_loss did not improve from 4.32474
    Epoch 32/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.1306 - acc: 0.0871 - val_loss: 4.3174 - val_acc: 0.0754
    
    Epoch 00032: val_loss improved from 4.32474 to 4.31742, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 33/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.1168 - acc: 0.0870 - val_loss: 4.4529 - val_acc: 0.0671
    
    Epoch 00033: val_loss did not improve from 4.31742
    Epoch 34/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.1049 - acc: 0.0960 - val_loss: 4.2601 - val_acc: 0.0743
    
    Epoch 00034: val_loss improved from 4.31742 to 4.26013, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 35/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.0865 - acc: 0.0945 - val_loss: 4.2644 - val_acc: 0.0814
    
    Epoch 00035: val_loss did not improve from 4.26013
    Epoch 36/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.0714 - acc: 0.0961 - val_loss: 4.3671 - val_acc: 0.0743
    
    Epoch 00036: val_loss did not improve from 4.26013
    Epoch 37/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.0597 - acc: 0.0978 - val_loss: 4.2707 - val_acc: 0.0731
    
    Epoch 00037: val_loss did not improve from 4.26013
    Epoch 38/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.0465 - acc: 0.1004 - val_loss: 4.2335 - val_acc: 0.0874
    
    Epoch 00038: val_loss improved from 4.26013 to 4.23348, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 39/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.0319 - acc: 0.1030 - val_loss: 4.2569 - val_acc: 0.0731
    
    Epoch 00039: val_loss did not improve from 4.23348
    Epoch 40/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.0201 - acc: 0.1066 - val_loss: 4.2157 - val_acc: 0.0994
    
    Epoch 00040: val_loss improved from 4.23348 to 4.21573, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 41/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.0144 - acc: 0.1073 - val_loss: 4.2472 - val_acc: 0.0934
    
    Epoch 00041: val_loss did not improve from 4.21573
    Epoch 42/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 4.0007 - acc: 0.1066 - val_loss: 4.2476 - val_acc: 0.0778
    
    Epoch 00042: val_loss did not improve from 4.21573
    Epoch 43/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 3.9902 - acc: 0.1138 - val_loss: 4.2149 - val_acc: 0.0898
    
    Epoch 00043: val_loss improved from 4.21573 to 4.21485, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 44/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 3.9761 - acc: 0.1144 - val_loss: 4.2229 - val_acc: 0.0826
    
    Epoch 00044: val_loss did not improve from 4.21485
    Epoch 45/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 3.9647 - acc: 0.1117 - val_loss: 4.1899 - val_acc: 0.0946
    
    Epoch 00045: val_loss improved from 4.21485 to 4.18987, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 46/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 3.9496 - acc: 0.1135 - val_loss: 4.1783 - val_acc: 0.0886
    
    Epoch 00046: val_loss improved from 4.18987 to 4.17828, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 47/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 3.9501 - acc: 0.1189 - val_loss: 4.1809 - val_acc: 0.0862
    
    Epoch 00047: val_loss did not improve from 4.17828
    Epoch 48/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 3.9364 - acc: 0.1196 - val_loss: 4.1782 - val_acc: 0.0695
    
    Epoch 00048: val_loss improved from 4.17828 to 4.17821, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 49/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 3.9195 - acc: 0.1235 - val_loss: 4.2286 - val_acc: 0.0850
    
    Epoch 00049: val_loss did not improve from 4.17821
    Epoch 50/50
    6680/6680 [==============================] - 13s 2ms/step - loss: 3.9144 - acc: 0.1234 - val_loss: 4.3020 - val_acc: 0.1066
    
    Epoch 00050: val_loss did not improve from 4.17821
    




    <keras.callbacks.History at 0x211a6dabb38>




```python
## 加载具有最好验证loss的模型

model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### 测试模型

在狗图像的测试数据集上试用你的模型。确保测试准确率大于1%。


```python
# 获取测试数据集中每一个图像所预测的狗品种的index
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 8.0144%
    

---
<a id='step4'></a>
## 步骤 4: 使用一个CNN来区分狗的品种


使用 迁移学习（Transfer Learning）的方法，能帮助我们在不损失准确率的情况下大大减少训练时间。在以下步骤中，你可以尝试使用迁移学习来训练你自己的CNN。


### 得到从图像中提取的特征向量（Bottleneck Features）


```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### 模型架构

该模型使用预训练的 VGG-16 模型作为固定的图像特征提取器，其中 VGG-16 最后一层卷积层的输出被直接输入到我们的模型。我们只需要添加一个全局平均池化层以及一个全连接层，其中全连接层使用 softmax 激活函数，对每一个狗的种类都包含一个节点。


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_2 ( (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________
    


```python
## 编译模型

VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```


```python
## 训练模型
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6680/6680 [==============================] - 4s 601us/step - loss: 12.5971 - acc: 0.1160 - val_loss: 11.3566 - val_acc: 0.1988
    
    Epoch 00001: val_loss improved from inf to 11.35656, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 2/20
    6680/6680 [==============================] - 2s 318us/step - loss: 10.7210 - acc: 0.2644 - val_loss: 10.8617 - val_acc: 0.2539
    
    Epoch 00002: val_loss improved from 11.35656 to 10.86174, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 3/20
    6680/6680 [==============================] - 2s 335us/step - loss: 10.3522 - acc: 0.3070 - val_loss: 10.4884 - val_acc: 0.2838
    
    Epoch 00003: val_loss improved from 10.86174 to 10.48844, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 4/20
    6680/6680 [==============================] - 2s 335us/step - loss: 9.8673 - acc: 0.3368 - val_loss: 10.0370 - val_acc: 0.2994
    
    Epoch 00004: val_loss improved from 10.48844 to 10.03697, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 5/20
    6680/6680 [==============================] - 2s 295us/step - loss: 9.5684 - acc: 0.3645 - val_loss: 9.7211 - val_acc: 0.3269
    
    Epoch 00005: val_loss improved from 10.03697 to 9.72113, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 6/20
    6680/6680 [==============================] - 2s 307us/step - loss: 9.2589 - acc: 0.3891 - val_loss: 9.4897 - val_acc: 0.3485
    
    Epoch 00006: val_loss improved from 9.72113 to 9.48972, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 7/20
    6680/6680 [==============================] - 2s 298us/step - loss: 9.0863 - acc: 0.4085 - val_loss: 9.5174 - val_acc: 0.3461
    
    Epoch 00007: val_loss did not improve from 9.48972
    Epoch 8/20
    6680/6680 [==============================] - 2s 297us/step - loss: 8.9121 - acc: 0.4162 - val_loss: 9.2632 - val_acc: 0.3521
    
    Epoch 00008: val_loss improved from 9.48972 to 9.26316, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 9/20
    6680/6680 [==============================] - 2s 293us/step - loss: 8.5749 - acc: 0.4373 - val_loss: 9.0345 - val_acc: 0.3713
    
    Epoch 00009: val_loss improved from 9.26316 to 9.03447, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 10/20
    6680/6680 [==============================] - 2s 301us/step - loss: 8.4085 - acc: 0.4588 - val_loss: 9.0627 - val_acc: 0.3629
    
    Epoch 00010: val_loss did not improve from 9.03447
    Epoch 11/20
    6680/6680 [==============================] - 2s 293us/step - loss: 8.2167 - acc: 0.4636 - val_loss: 8.7051 - val_acc: 0.3808
    
    Epoch 00011: val_loss improved from 9.03447 to 8.70509, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 12/20
    6680/6680 [==============================] - 2s 309us/step - loss: 8.0120 - acc: 0.4846 - val_loss: 8.6207 - val_acc: 0.3928
    
    Epoch 00012: val_loss improved from 8.70509 to 8.62072, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 13/20
    6680/6680 [==============================] - 2s 292us/step - loss: 7.7706 - acc: 0.4969 - val_loss: 8.3674 - val_acc: 0.4072
    
    Epoch 00013: val_loss improved from 8.62072 to 8.36741, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 14/20
    6680/6680 [==============================] - 2s 304us/step - loss: 7.6885 - acc: 0.5094 - val_loss: 8.3158 - val_acc: 0.4144
    
    Epoch 00014: val_loss improved from 8.36741 to 8.31580, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 15/20
    6680/6680 [==============================] - 2s 300us/step - loss: 7.5739 - acc: 0.5154 - val_loss: 8.2186 - val_acc: 0.4228
    
    Epoch 00015: val_loss improved from 8.31580 to 8.21864, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 16/20
    6680/6680 [==============================] - 2s 297us/step - loss: 7.4138 - acc: 0.5159 - val_loss: 7.9882 - val_acc: 0.4263
    
    Epoch 00016: val_loss improved from 8.21864 to 7.98822, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 17/20
    6680/6680 [==============================] - 2s 298us/step - loss: 7.0000 - acc: 0.5371 - val_loss: 7.6365 - val_acc: 0.4491
    
    Epoch 00017: val_loss improved from 7.98822 to 7.63652, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 18/20
    6680/6680 [==============================] - 2s 298us/step - loss: 6.7299 - acc: 0.5582 - val_loss: 7.5253 - val_acc: 0.4503
    
    Epoch 00018: val_loss improved from 7.63652 to 7.52526, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 19/20
    6680/6680 [==============================] - 2s 300us/step - loss: 6.6167 - acc: 0.5710 - val_loss: 7.4486 - val_acc: 0.4623
    
    Epoch 00019: val_loss improved from 7.52526 to 7.44863, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 20/20
    6680/6680 [==============================] - 2s 326us/step - loss: 6.5226 - acc: 0.5793 - val_loss: 7.5104 - val_acc: 0.4563
    
    Epoch 00020: val_loss did not improve from 7.44863
    




    <keras.callbacks.History at 0x2a053703860>




```python
## 加载具有最好验证loss的模型

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### 测试模型
现在，我们可以测试此CNN在狗图像测试数据集中识别品种的效果如何。我们在下方打印出测试准确率。


```python
# 获取测试数据集中每一个图像所预测的狗品种的index
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 51.1962%
    

### 使用模型预测狗的品种


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # 提取bottleneck特征
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## 步骤 5: 建立一个CNN来分类狗的品种（使用迁移学习）

现在你将使用迁移学习来建立一个CNN，从而可以从图像中识别狗的品种。你的 CNN 在测试集上的准确率必须至少达到60%。

在步骤4中，我们使用了迁移学习来创建一个使用基于 VGG-16 提取的特征向量来搭建一个 CNN。在本部分内容中，你必须使用另一个预训练模型来搭建一个 CNN。为了让这个任务更易实现，我们已经预先对目前 keras 中可用的几种网络进行了预训练：

- [VGG-19](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogXceptionData.npz) bottleneck features

这些文件被命名为为：

    Dog{network}Data.npz

其中 `{network}` 可以是 `VGG19`、`Resnet50`、`InceptionV3` 或 `Xception` 中的一个。选择上方网络架构中的一个，下载相对应的bottleneck特征，并将所下载的文件保存在目录 `bottleneck_features/` 中。


### 【练习】获取模型的特征向量

在下方代码块中，通过运行下方代码提取训练、测试与验证集相对应的bottleneck特征。

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: 从另一个预训练的CNN获取bottleneck特征
bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
valid_VGG19 = bottleneck_features['valid']
test_VGG19 = bottleneck_features['test']


```

### 【练习】模型架构

建立一个CNN来分类狗品种。在你的代码单元块的最后，通过运行如下代码输出网络的结构：
    
        <your model's name>.summary()
   
---

<a id='question6'></a>  

### __问题 6:__ 


在下方的代码块中尝试使用 Keras 搭建最终的网络架构，并回答你实现最终 CNN 架构的步骤与每一步的作用，并描述你在迁移学习过程中，使用该网络架构的原因。


__回答:__ 

VGG19_model = Sequential()创建神经网络模型
VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))添加全局池化层
VGG19_model.add(Dense(133, activation='softmax'))添加全连接层，将输入图片转换为预测概率

使用VGG19原因：
通过构建VGG-19、ResNet-50 、Inception 、Xception模型，VGG19模型参数较少，训练速度快。
和自己构建模型相比，VGG-19采用了19层卷积/全连接层，特征获取更全面。



```python
### TODO: 定义你的框架
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
VGG19_model = Sequential()
VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
VGG19_model.add(Dense(133, activation='softmax'))
VGG19_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_2 ( (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________
    


```python
### TODO: 编译模型
VGG19_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

---

### 【练习】训练模型

<a id='question7'></a>  

### __问题 7:__ 

在下方代码单元中训练你的模型。使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。

当然，你也可以对训练集进行 [数据增强](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 以优化模型的表现，不过这不是必须的步骤。



```python
### TODO: 训练模型
from keras.callbacks import ModelCheckpoint 
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5', 
                               verbose=1, save_best_only=True)

VGG19_model.fit(train_VGG19, train_targets, 
          validation_data=(valid_VGG19, valid_targets),
          epochs=50, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/50
    6680/6680 [==============================] - 7s 1ms/step - loss: 7.6569 - acc: 0.5228 - val_loss: 8.2134 - val_acc: 0.4323
    
    Epoch 00001: val_loss improved from inf to 8.21343, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 2/50
    6680/6680 [==============================] - 5s 743us/step - loss: 7.4460 - acc: 0.5262 - val_loss: 8.0227 - val_acc: 0.4383
    
    Epoch 00002: val_loss improved from 8.21343 to 8.02271, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 3/50
    6680/6680 [==============================] - 5s 725us/step - loss: 7.0486 - acc: 0.5460 - val_loss: 7.7247 - val_acc: 0.4419
    
    Epoch 00003: val_loss improved from 8.02271 to 7.72466, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 4/50
    6680/6680 [==============================] - 5s 708us/step - loss: 6.8419 - acc: 0.5593 - val_loss: 7.5748 - val_acc: 0.4527
    
    Epoch 00004: val_loss improved from 7.72466 to 7.57484, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 5/50
    6680/6680 [==============================] - 5s 694us/step - loss: 6.6711 - acc: 0.5684 - val_loss: 7.4552 - val_acc: 0.4419
    
    Epoch 00005: val_loss improved from 7.57484 to 7.45524, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 6/50
    6680/6680 [==============================] - 5s 680us/step - loss: 6.5801 - acc: 0.5795 - val_loss: 7.4292 - val_acc: 0.4575
    
    Epoch 00006: val_loss improved from 7.45524 to 7.42919, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 7/50
    6680/6680 [==============================] - 5s 691us/step - loss: 6.5449 - acc: 0.5846 - val_loss: 7.3844 - val_acc: 0.4611
    
    Epoch 00007: val_loss improved from 7.42919 to 7.38444, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 8/50
    6680/6680 [==============================] - 5s 700us/step - loss: 6.5004 - acc: 0.5895 - val_loss: 7.4174 - val_acc: 0.4599
    
    Epoch 00008: val_loss did not improve from 7.38444
    Epoch 9/50
    6680/6680 [==============================] - 5s 712us/step - loss: 6.4679 - acc: 0.5913 - val_loss: 7.2943 - val_acc: 0.4743
    
    Epoch 00009: val_loss improved from 7.38444 to 7.29428, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 10/50
    6680/6680 [==============================] - 5s 708us/step - loss: 6.4425 - acc: 0.5930 - val_loss: 7.3142 - val_acc: 0.4719
    
    Epoch 00010: val_loss did not improve from 7.29428
    Epoch 11/50
    6680/6680 [==============================] - 5s 712us/step - loss: 6.4118 - acc: 0.5942 - val_loss: 7.4775 - val_acc: 0.4491
    
    Epoch 00011: val_loss did not improve from 7.29428
    Epoch 12/50
    6680/6680 [==============================] - 5s 709us/step - loss: 6.2767 - acc: 0.6010 - val_loss: 7.2500 - val_acc: 0.4802
    
    Epoch 00012: val_loss improved from 7.29428 to 7.25002, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 13/50
    6680/6680 [==============================] - 5s 685us/step - loss: 6.1630 - acc: 0.6084 - val_loss: 7.1906 - val_acc: 0.4707
    
    Epoch 00013: val_loss improved from 7.25002 to 7.19065, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 14/50
    6680/6680 [==============================] - 5s 683us/step - loss: 6.0062 - acc: 0.6142 - val_loss: 6.9892 - val_acc: 0.4862
    
    Epoch 00014: val_loss improved from 7.19065 to 6.98921, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 15/50
    6680/6680 [==============================] - 5s 679us/step - loss: 5.9082 - acc: 0.6265 - val_loss: 6.8632 - val_acc: 0.4910
    
    Epoch 00015: val_loss improved from 6.98921 to 6.86320, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 16/50
    6680/6680 [==============================] - 5s 678us/step - loss: 5.8973 - acc: 0.6284 - val_loss: 6.8426 - val_acc: 0.5006
    
    Epoch 00016: val_loss improved from 6.86320 to 6.84256, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 17/50
    6680/6680 [==============================] - 5s 675us/step - loss: 5.8841 - acc: 0.6313 - val_loss: 6.8513 - val_acc: 0.5066
    
    Epoch 00017: val_loss did not improve from 6.84256
    Epoch 18/50
    6680/6680 [==============================] - 5s 689us/step - loss: 5.8630 - acc: 0.6319 - val_loss: 6.7354 - val_acc: 0.4982
    
    Epoch 00018: val_loss improved from 6.84256 to 6.73544, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 19/50
    6680/6680 [==============================] - 5s 726us/step - loss: 5.6602 - acc: 0.6389 - val_loss: 6.5976 - val_acc: 0.5090
    
    Epoch 00019: val_loss improved from 6.73544 to 6.59756, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 20/50
    6680/6680 [==============================] - 5s 715us/step - loss: 5.5365 - acc: 0.6476 - val_loss: 6.5689 - val_acc: 0.5174
    
    Epoch 00020: val_loss improved from 6.59756 to 6.56888, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 21/50
    6680/6680 [==============================] - 5s 682us/step - loss: 5.3838 - acc: 0.6542 - val_loss: 6.4106 - val_acc: 0.5281
    
    Epoch 00021: val_loss improved from 6.56888 to 6.41056, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 22/50
    6680/6680 [==============================] - 5s 683us/step - loss: 5.3070 - acc: 0.6629 - val_loss: 6.3655 - val_acc: 0.5222
    
    Epoch 00022: val_loss improved from 6.41056 to 6.36551, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 23/50
    6680/6680 [==============================] - 5s 681us/step - loss: 5.2516 - acc: 0.6674 - val_loss: 6.2025 - val_acc: 0.5341
    
    Epoch 00023: val_loss improved from 6.36551 to 6.20253, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 24/50
    6680/6680 [==============================] - 5s 682us/step - loss: 5.1377 - acc: 0.6710 - val_loss: 6.0898 - val_acc: 0.5305
    
    Epoch 00024: val_loss improved from 6.20253 to 6.08976, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 25/50
    6680/6680 [==============================] - 5s 701us/step - loss: 5.0008 - acc: 0.6834 - val_loss: 6.0496 - val_acc: 0.5437
    
    Epoch 00025: val_loss improved from 6.08976 to 6.04960, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 26/50
    6680/6680 [==============================] - 5s 714us/step - loss: 4.9645 - acc: 0.6880 - val_loss: 6.0783 - val_acc: 0.5377
    
    Epoch 00026: val_loss did not improve from 6.04960
    Epoch 27/50
    6680/6680 [==============================] - 5s 684us/step - loss: 4.9044 - acc: 0.6891 - val_loss: 6.0441 - val_acc: 0.5389
    
    Epoch 00027: val_loss improved from 6.04960 to 6.04409, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 28/50
    6680/6680 [==============================] - 5s 679us/step - loss: 4.8513 - acc: 0.6949 - val_loss: 6.1204 - val_acc: 0.5413
    
    Epoch 00028: val_loss did not improve from 6.04409
    Epoch 29/50
    6680/6680 [==============================] - 5s 680us/step - loss: 4.8329 - acc: 0.6961 - val_loss: 5.9464 - val_acc: 0.5509
    
    Epoch 00029: val_loss improved from 6.04409 to 5.94640, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 30/50
    6680/6680 [==============================] - 5s 678us/step - loss: 4.7732 - acc: 0.6976 - val_loss: 5.9855 - val_acc: 0.5473
    
    Epoch 00030: val_loss did not improve from 5.94640
    Epoch 31/50
    6680/6680 [==============================] - 5s 678us/step - loss: 4.6765 - acc: 0.7015 - val_loss: 5.9395 - val_acc: 0.5425
    
    Epoch 00031: val_loss improved from 5.94640 to 5.93950, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 32/50
    6680/6680 [==============================] - 5s 674us/step - loss: 4.6122 - acc: 0.7070 - val_loss: 5.9055 - val_acc: 0.5473
    
    Epoch 00032: val_loss improved from 5.93950 to 5.90547, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 33/50
    6680/6680 [==============================] - 5s 692us/step - loss: 4.5196 - acc: 0.7126 - val_loss: 5.8516 - val_acc: 0.5413
    
    Epoch 00033: val_loss improved from 5.90547 to 5.85156, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 34/50
    6680/6680 [==============================] - 5s 709us/step - loss: 4.4609 - acc: 0.7132 - val_loss: 5.8312 - val_acc: 0.5413
    
    Epoch 00034: val_loss improved from 5.85156 to 5.83125, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 35/50
    6680/6680 [==============================] - 5s 762us/step - loss: 4.3463 - acc: 0.7193 - val_loss: 5.7528 - val_acc: 0.5593
    
    Epoch 00035: val_loss improved from 5.83125 to 5.75282, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 36/50
    6680/6680 [==============================] - 5s 751us/step - loss: 4.3053 - acc: 0.7240 - val_loss: 5.6541 - val_acc: 0.5665
    
    Epoch 00036: val_loss improved from 5.75282 to 5.65408, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 37/50
    6680/6680 [==============================] - 5s 699us/step - loss: 4.2983 - acc: 0.7269 - val_loss: 5.6611 - val_acc: 0.5677
    
    Epoch 00037: val_loss did not improve from 5.65408
    Epoch 38/50
    6680/6680 [==============================] - 5s 680us/step - loss: 4.2849 - acc: 0.7287 - val_loss: 5.6905 - val_acc: 0.5653
    
    Epoch 00038: val_loss did not improve from 5.65408
    Epoch 39/50
    6680/6680 [==============================] - 5s 687us/step - loss: 4.2570 - acc: 0.7307 - val_loss: 5.7094 - val_acc: 0.5653
    
    Epoch 00039: val_loss did not improve from 5.65408
    Epoch 40/50
    6680/6680 [==============================] - 5s 699us/step - loss: 4.2059 - acc: 0.7337 - val_loss: 5.6278 - val_acc: 0.5701
    
    Epoch 00040: val_loss improved from 5.65408 to 5.62779, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 41/50
    6680/6680 [==============================] - 5s 720us/step - loss: 4.1940 - acc: 0.7349 - val_loss: 5.6849 - val_acc: 0.5533
    
    Epoch 00041: val_loss did not improve from 5.62779
    Epoch 42/50
    6680/6680 [==============================] - 5s 685us/step - loss: 4.1813 - acc: 0.7370 - val_loss: 5.6894 - val_acc: 0.5665
    
    Epoch 00042: val_loss did not improve from 5.62779
    Epoch 43/50
    6680/6680 [==============================] - 5s 677us/step - loss: 4.1717 - acc: 0.7382 - val_loss: 5.6231 - val_acc: 0.5725
    
    Epoch 00043: val_loss improved from 5.62779 to 5.62307, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 44/50
    6680/6680 [==============================] - 5s 675us/step - loss: 4.1485 - acc: 0.7395 - val_loss: 5.5480 - val_acc: 0.5653
    
    Epoch 00044: val_loss improved from 5.62307 to 5.54796, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 45/50
    6680/6680 [==============================] - 5s 676us/step - loss: 4.0795 - acc: 0.7412 - val_loss: 5.8153 - val_acc: 0.5437
    
    Epoch 00045: val_loss did not improve from 5.54796
    Epoch 46/50
    6680/6680 [==============================] - 5s 683us/step - loss: 4.0604 - acc: 0.7440 - val_loss: 5.5534 - val_acc: 0.5641
    
    Epoch 00046: val_loss did not improve from 5.54796
    Epoch 47/50
    6680/6680 [==============================] - 5s 686us/step - loss: 4.0473 - acc: 0.7460 - val_loss: 5.5980 - val_acc: 0.5677
    
    Epoch 00047: val_loss did not improve from 5.54796
    Epoch 48/50
    6680/6680 [==============================] - 5s 695us/step - loss: 4.0457 - acc: 0.7461 - val_loss: 5.5348 - val_acc: 0.5725
    
    Epoch 00048: val_loss improved from 5.54796 to 5.53475, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 49/50
    6680/6680 [==============================] - 5s 683us/step - loss: 4.0415 - acc: 0.7475 - val_loss: 5.6146 - val_acc: 0.5665
    
    Epoch 00049: val_loss did not improve from 5.53475
    Epoch 50/50
    6680/6680 [==============================] - 5s 677us/step - loss: 4.0409 - acc: 0.7476 - val_loss: 5.5587 - val_acc: 0.5653
    
    Epoch 00050: val_loss did not improve from 5.53475
    




    <keras.callbacks.History at 0x21b28dad358>




```python
### TODO: 加载具有最佳验证loss的模型权重
VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')
```

---

### 【练习】测试模型

<a id='question8'></a>  

### __问题 8:__ 

在狗图像的测试数据集上试用你的模型。确保测试准确率大于60%。


```python
### TODO: 在测试集上计算分类准确率
VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 62.5598%
    

---

### 【练习】使用模型测试狗的品种


实现一个函数，它的输入为图像路径，功能为预测对应图像的类别，输出为你模型预测出的狗类别（`Affenpinscher`, `Afghan_hound` 等）。

与步骤5中的模拟函数类似，你的函数应当包含如下三个步骤：

1. 根据选定的模型载入图像特征（bottleneck features）
2. 将图像特征输输入到你的模型中，并返回预测向量。注意，在该向量上使用 argmax 函数可以返回狗种类的序号。
3. 使用在步骤0中定义的 `dog_names` 数组来返回对应的狗种类名称。

提取图像特征过程中使用到的函数可以在 `extract_bottleneck_features.py` 中找到。同时，他们应已在之前的代码块中被导入。根据你选定的 CNN 网络，你可以使用 `extract_{network}` 函数来获得对应的图像特征，其中 `{network}` 代表 `VGG19`, `Resnet50`, `InceptionV3`, 或 `Xception` 中的一个。
 
---

<a id='question9'></a>  

### __问题 9:__


```python
### TODO: 写一个函数，该函数将图像的路径作为输入
### 然后返回此模型所预测的狗的品种
from extract_bottleneck_features import *

def VGG19_predict_breed(img_path):
    # 提取bottleneck特征
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]

#VGG19_predict_breed("images/American_water_spaniel_00648.jpg")
```

---

<a id='step6'></a>
## 步骤 6: 完成你的算法



实现一个算法，它的输入为图像的路径，它能够区分图像是否包含一个人、狗或两者都不包含，然后：

- 如果从图像中检测到一只__狗__，返回被预测的品种。
- 如果从图像中检测到__人__，返回最相像的狗品种。
- 如果两者都不能在图像中检测到，输出错误提示。

我们非常欢迎你来自己编写检测图像中人类与狗的函数，你可以随意地使用上方完成的 `face_detector` 和 `dog_detector` 函数。你__需要__在步骤5使用你的CNN来预测狗品种。

下面提供了算法的示例输出，但你可以自由地设计自己的模型！

![Sample Human Output](images/sample_human_output.png)




<a id='question10'></a>  

### __问题 10:__

在下方代码块中完成你的代码。

---



```python
### TODO: 设计你的算法
### 自由地使用所需的代码单元数吧
def predict_labels(img_path):
    if dog_detector(img_path):
        return VGG19_predict_breed(img_path)
    elif face_detector(img_path):
        return "hello human! you look like a ... " + VGG19_predict_breed(img_path)
    else:
        return "I can't recognize it!!!"


```

---
<a id='step7'></a>
## 步骤 7: 测试你的算法

在这个部分中，你将尝试一下你的新算法！算法认为__你__看起来像什么类型的狗？如果你有一只狗，它可以准确地预测你的狗的品种吗？如果你有一只猫，它会将你的猫误判为一只狗吗？


<a id='question11'></a>  

### __问题 11:__

在下方编写代码，用至少6张现实中的图片来测试你的算法。你可以使用任意照片，不过请至少使用两张人类图片（要征得当事人同意哦）和两张狗的图片。
同时请回答如下问题：

1. 输出结果比你预想的要好吗 :) ？或者更糟 :( ？
2. 提出至少三点改进你的模型的想法。

1.算法在分辨是否包含人或狗的分类问题，测试准确率达到了100%，但在狗狗品种细分上，准确率低于50%
2.对训练数据进行数据增强，增加训练数据； 
在全局池化层后添加全连接层，获取更多数据特征； 
训练过程中增加dropout，防止过拟合




```python
## TODO: 在你的电脑上，在步骤6中，至少在6张图片上运行你的算法。
## 自由地使用所需的代码单元数吧
import os
file_names =[]
for root, dirs, files in os.walk('testimg'):
    file_names = files

#print(file_names)

for i in file_names:
    img = cv2.imread('testimg/%s' % i)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    print(predict_labels('testimg/%s' % i))
    plt.show()
```

    hello human! you look like a ... Greyhound
    


![png](output_63_1.png)


    Alaskan_malamute
    


![png](output_63_3.png)


    hello human! you look like a ... Kerry_blue_terrier
    


![png](output_63_5.png)


    I can't recognize it!!!
    


![png](output_63_7.png)


    Poodle
    


![png](output_63_9.png)


    American_eskimo_dog
    


![png](output_63_11.png)


    Golden_retriever
    


![png](output_63_13.png)


    hello human! you look like a ... Maltese
    


![png](output_63_15.png)


**注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出File -> Download as -> HTML (.html)把这个 HTML 和这个 iPython notebook 一起做为你的作业提交。**
