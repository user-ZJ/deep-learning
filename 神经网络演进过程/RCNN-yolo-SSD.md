# 检测网发展过程  
![](https://i.imgur.com/QdPFTAv.jpg)  

# RCNN-FastRCNN-FasterRCNN-yolo-SSD
本文主要梳理检测网络发展过程。  
检测网络基本思想为：  
1. 候选区域生成； 
2. 特征提取； 
3. 分类；  
4. 位置精修；  

数据库：   
一个较大的识别库（ImageNet ILSVC 2012）：标定每张图片中物体的类别。一千万图像，1000类。   
一个较小的检测库（PASCAL VOC 2007）：标定每张图片中，物体的类别和位置。一万图像，20类。  

名词：  
bounding boxes：预测的边界框  
Priorbox：先验框  
ground truth：标记框  
region proposal：区域生成/候选框  
ROI：Region of Interest的简写，一般是指图像上的区域框/候选框  
RoIs：其表示所有RoI的N*5的矩阵。其中N表示RoI的数量，第一列表示图像index，其余四列表示其余的左上角和右下角坐标，坐标的参考系不是针对feature map这张图的，而是针对原图的  

## 1.非极大值抑制
非极大值抑制顾名思义就是抑制不是极大值的元素，搜索局部的极大值  
非极大值抑制（NMS）先计算出每一个bounding box的面积，然后根据score进行排序，把score最大的bounding box作为选定的框，计算其余bounding box与当前最大score与box的IoU，去除IoU大于设定的阈值的bounding box，然后重复上面的过程，直至候选bounding box为空，然后再将score小于一定阈值的bounding box删除得到一类的结果。  

## 2.RCNN-Faster RCNN
从RCNN到fast RCNN，再到本文的faster RCNN，目标检测的四个基本步骤（候选区域生成，特征提取，分类，位置精修）被统一到一个深度网络框架之内。所有计算没有重复，完全在GPU中完成，大大提高了运行速度。   
![](https://i.imgur.com/y1OfrSr.png)  

## 3.RCNN
Region CNN(RCNN)可以说是利用深度学习进行目标检测的开山之作。  
整个架构如下图所示:  
![](https://i.imgur.com/FuHHVQR.png)  

### 候选区域生成
使用了Selective Search方法从一张图像生成约2000-3000个候选区域。基本思路如下：   
- 使用一种过分割手段，将图像分割成小区域   
- 查看现有小区域，合并可能性最高的两个区域。重复直到整张图像合并成一个区域位置   
- 输出所有曾经存在过的区域，所谓候选区域  

候选区域生成和后续步骤相对独立，实际可以使用任意算法进行。  


合并规则：  
优先合并以下四种区域：  
1.颜色（颜色直方图）相近的   
2.纹理（梯度直方图）相近的   
3.合并后总面积小的，保证合并操作的尺度较为均匀，避免一个大区域陆续“吃掉”其他小区域     
4.合并后，总面积在其BBOX中所占比例大的，保证合并后形状规则。  
![](https://i.imgur.com/Nd8waKT.png)    

多样化与后处理：  
为尽可能不遗漏候选区域，上述操作在多个颜色空间中同时进行（RGB,HSV,Lab等）。在一个颜色空间中，使用上述四条规则的不同组合进行合并。所有颜色空间与所有规则的全部结果，在去除重复后，都作为候选区域输出。  

### 特征提取  
预处理数据：  
使用深度网络提取特征之前，首先把候选区域归一化成同一尺寸227×227(padding=16(边缘扩张),对region proposal进行缩放)。   
此处有一些细节可做变化：外扩的尺寸大小，形变时是否保持原比例，对框外区域直接截取还是补灰。会轻微影响性能。  

#### 预训练网络
预网络结构：  
基本借鉴AlexNet的分类网络，略作简化   
![](https://i.imgur.com/kI81xkG.png)   
此网络提取的特征为4096维，之后送入一个4096->1000的全连接(fc)层进行分类。   
学习率0.01。  

预训练数据：  
使用ILVCR 2012的全部数据进行训练，输入一张图片，输出1000维的类别标号。   
![](https://i.imgur.com/RUhdr0t.png)   

#### finetune网络
网络结构：     
同样使用上述网络，最后一层换成4096->21的全连接网络。   
学习率0.001（保证训练只是对网络的微调而不是大幅度的变化），每一个batch包含32个正样本（属于20类）和96个背景。（这么做的主要原因还是正样本图片太少了）    

训练数据:  
使用PASCAL VOC 2007的训练集，输入一张图片，输出21维的类别标号，表示20类+背景。 
如果候选框和当前图像上所有标定框重叠面积最大的一个。如果重叠比例大于0.5，则认为此候选框为此标定的类别；否则认为此候选框为背景。  
如果当前region proposal（候选区域）与图像上所标定框的IOU大于0.5，把他标记为正样本（标定的类别），其余的是作为负样本（背景），去训练detection网络  
![](https://i.imgur.com/dCpTBmm.png)  

### 类别判断--分类  
分类器 
**对每一类，使用一个线性SVM二类分类器进行判别**。输入为深度网络输出的4096维特征，输出是否属于此类。 
由于负样本很多，使用hard negative mining方法。 
正样本 
考察每一个候选框，如果和本类标定框的IOU大于0.3，认定其为正样本  
负样本 
考察每一个候选框，如果和本类所有标定框的重叠都小于0.3，认定其为负样本  
![](https://i.imgur.com/165JX9T.png)  

### 位置精修  
目标检测问题的衡量标准是重叠面积：许多看似准确的检测结果，往往因为候选框不够准确，重叠面积很小。故需要一个位置精修步骤。 回归器对每一类目标，使用一个线性脊回归器进行精修。正则项。  
输入为深度网络pool5层的4096维特征，输出为xy方向的缩放和平移。 训练样本判定为本类的候选框中，和真值重叠面积大于0.6的候选框。
https://blog.csdn.net/zijin0802034/article/details/77685438/  

### 测试阶段
测试阶段，使用selective search的方法在测试图片上提取2000个region propasals ，将每个region proposals归一化到227x227，然后再CNN中正向传播，将最后一层得到的特征提取出来。然后对于**每一个类别，使用为这一类训练的SVM分类器对提取的特征向量进行打分**，得到测试图片中对于所有region proposals的对于这一类的分数，再使用贪心的**非极大值抑制**去除相交的多余的框。然后重复上面的过程，直至候选bounding box为空，然后再将score小于一定阈值的选定框删除得到一类的结果。  
再根据pool5 feature做了个bbox regression来decrease location error（位置精修）.  

> r-cnn需要两次进行跑cnn model，第一次得到classification的结果，第二次才能得到(nms+b-box regression)bounding-box。  
> 缺点：  
> 1.  训练时要经过多个阶段，首先要提取特征微调ConvNet，再用线性SVM处理proposal，计算得到的ConvNet特征，然后进行用bounding box回归  
> 2. 训练时间和空间开销大。要从每一张图像上提取大量proposal，还要从每个proposal中提取特征，并存到磁盘中  
> 3. 测试时间开销大，要从每个测试图像上提取大量proposal，再从每个proposal中提取特征来进行检测过程  

## 4. SPP-net（Spatial Pyramid Pooling-空间金字塔池化）
CNN网络需要固定尺寸的图像输入，**SPPNet将任意大小的图像池化生成固定长度的图像表示**，提升R-CNN检测的速度24-102倍。  
事实上，CNN的卷积层不需要固定尺寸的图像，全连接层是需要固定大小输入的，因此提出了SPP层放到卷积层的后面，改进后的网络如下图所示：  
![](https://i.imgur.com/eHTeSfy.png)  

### SPPNet结构  
改进：  
1. 生成的proposal（候选区域）不需要进行缩放即可进行特征提取  
2. 整张图片只进行一次特征提取，候选区域为提取特征中的部分区域   

通过对feature map进行相应尺度的pooling，使得能pooling出4×4, 2×2, 1×1的feature map，再将这些feature map concat成列向量与下一层全链接层相连。这样就消除了输入尺度不一致的影响。
例如：原图输入是224x224，对于conv5出来后的输出，是13x13x256，分成4x4 2x2 1x1三张子图，做max pooling后，出来的特征就是固定长度的(16+4+1)x256，如果原图的输入不是224x224，出来的特征依然是(16+4+1)x256  
![](https://i.imgur.com/lfukVVe.png)  

使用SPP进行检测，不像RCNN把每个候选区域给深度网络提特征，而是整张图提一次特征，再把候选框映射到conv5上，因为候选框的大小尺度不同，映射到conv5后仍不同，所以需要再通过SPP层提取到相同维度的特征，再进行分类和回归（比较耗时的卷积计算对整幅图像只进行一次）

> 缺点：  
> 1. 训练要经过多个阶段，特征也要存在磁盘中  
> 2. 在微调阶段SPP-net只能更新FC层,这是因为卷积特征是线下计算的,从而无法再微调阶段反向传播误差  


## 5.Fast-RCNN
改进：  
1. 比R-CNN更高的检测质量（mAP）；  
2. 把多个任务的损失函数写到一起，实现单级的训练过程；   
3. 在训练时可更新所有的层；   
4. 去掉了SVM这一步，所有的特征都暂存在显存中，不需要在磁盘中存储特征，可以实现反向传播，训练卷积层和全链接层。  

训练过程：  
![](https://i.imgur.com/D58IxOz.jpg)  
1. 用selective search在一张图片中生成约2000个object proposal，即RoI。  
2. 把整张图片输入到全卷积的网络中，在最后一个卷积层上对每个ROI求映射关系，并用一个RoI pooling layer（SPP layer）来统一到相同的大小(fc)feature vector 即提取一个固定维度的特征表示  
3. 继续经过两个全连接层（FC）得到特征向量。特征向量经由各自的FC层，得到两个输出向量：第一个是分类，使用softmax，第二个是每一类的bounding box回归  

### RoI pooling layer
是SPP pooling层的一个简化版，只有一级“金字塔”  

RoI pooling layer的作用主要有两个：  
1. 将image中的ROI定位到feature map中对应patch  
2. 用一个单层的SPP layer将这个feature map patch下采样为大小固定的feature再传入全连接层，即用RoI pooling layer来统一到相同的大小(fc)feature vector，即提取一个固定维度的特征表示  

RoI-centric sampling：从所有图片的所有RoI中均匀取样，这样每个SGD的mini-batch中包含了不同图像中的样本。  

FRCN想要解决微调的限制,就要反向传播到spp层之前的层->(reason)反向传播需要计算每一个RoI感受野的卷积层，通常会覆盖整个图像，如果一个一个用RoI-centric sampling的话就又慢又耗内存  

image-centric sampling：mini-batch采用层次取样，先对图像取样，再对RoI取样，同一图像的RoI共享计算和内存。 另外，FRCN在一次微调中联合优化softmax分类器和bbox回归  

### Multi-task loss（多任务损失）
ROI分类执行度：p = (p0, . . . , pK)  
bounding box回归的位移：![](https://i.imgur.com/GNicGeM.png)  
k表示类别的索引  
Multi-task loss表示为：  
![](https://i.imgur.com/kKHVMG0.png)  
其中u表示实际类别标签，v表示真实标记框坐标。  
背景标签为0，计算位置误差时，只计算目标位置误差，不计算背景位置误差  

位置误差使用平滑的L1计算
![](https://i.imgur.com/sOmn4XC.png)  

### Mini-batch sampling（小批量取样）
在微调时，每个SGD的mini-batch是随机找两个图片，R为128，因此每个图上取样64个RoI。从object proposal中选25%的RoI，就是和ground-truth交叠至少为0.5的。剩下的作为背景。  

每一个mini-batch中首先加入N张完整图片，而后加入从N张图片中选取的R个候选框。这R个候选框可以复用N张图片前5个阶段的网络特征。 

实际选择N=2， R=128－> 每一个mini-batch中首先加入2张完整图片，而后加入从2张图片中选取的128个候选框。这128个候选框可以复用2张图片前5个阶段的网络特征。   

N张完整图片以50%概率水平翻转。   
R个候选框的构成方式如下：  
前景占比25%：与某个真值重叠在[0.5,1]的候选框  
背景占比75%：与真值重叠的最大值在[0.1,0.5)的候选框  

### Backpropagation through RoI pooling layers（RoI pooling层的反向传播）
RoI pooling层计算损失函数对每个输入变量x的偏导数：  
![](https://i.imgur.com/ZmpOo39.png)  
y是pooling后的输出单元，x是pooling前的输入单元，如果y由x pooling而来，则将损失L对y的偏导计入累加值，最后累加完R个RoI中的所有输出单元。  

### Scale invariance（尺度不变性）
SPPnet用了两种实现尺度不变的方法：   
1. brute force （single scale），直接将image设置为某种scale，直接输入网络训练，期望网络自己适应这个scale。   
2. image pyramids （multi scale），生成一个图像金字塔，在multi-scale训练时，对于要用的RoI，在金字塔上找到一个最接近227x227的尺寸，然后用这个尺寸训练网络。   
虽然看起来2比较好，但是非常耗时，而且性能提高也不多，大约只有%1，所以论文在实现中还是用了1  

### Which layers to finetune?
1. 对于较深的网络，比如VGG，卷积层和全连接层是否一起tuning有很大的差别（66.9 vs 61.4）  
2. 没有必要tuning所有的卷积层，如果留着浅层的卷积层不tuning，可以减少训练时间，而且mAP基本没有差别  

### Truncated SVD for faster detection（奇异值分解加快检测速度）
在分类中，计算全连接层比卷积层快，而在检测中由于一个图中要提取2000个RoI，所以大部分时间都用在计算全连接层了。文中采用奇异值分解的方法来减少计算fc层的时间.  
具体来说，作者对全连接层的矩阵做了一个SVD分解，mAP几乎不怎么降（0.3%），但速度提速30%  

SVD中，（u × v）权重矩阵W参数化的层近似地分解为：  
![](https://i.imgur.com/hGYYIYX.png)  
U为（u × t）矩阵，Σt为（t × t）矩阵，V为（v x t）矩阵，SVD将计算量从u*v减小到t*(u+v),t小于min(u,v)  
在实现时，相当于把一个全连接层拆分成两个，中间以一个低维数据相连。  
![](https://i.imgur.com/2DBDZ9k.png)  

## 6. faster rcnn

Fast R-CNN依赖于外部候选区域方法，如选择性搜索。但这些算法在CPU上运行且速度很慢。在测试中，Fast R-CNN需要2.3秒来进行预测，其中2秒用于生成2000个ROI  

Faster RCNN网络结构：  
• 将RPN放在最后一个卷积层的后面   
• RPN直接训练得到候选区域    
![](https://i.imgur.com/fLVQRjE.png)  


faster RCNN可以简单地看做“区域生成网络(RPN)+fast RCNN“的系统，用区域生成网络代替fast RCNN中的Selective Search方法，  
Faster RCNN解决了一下三个问题：  
1. 如何设计区域生成网络 
2. 如何训练区域生成网络 
3. 如何让区域生成网络和fast RCNN网络共享特征提取网络


Faster RCNN网络结构：  
![](https://i.imgur.com/2ASnA6g.png)  
![](https://i.imgur.com/nxm55a9.png)  

原始特征提取（raw feature extraction）直接套用ImageNet上常见的分类网络即可。 额外添加一个conv+relu层，输出51*39*256维特征（feature）  

### RPN（Region Proposal Networks）
先通过SPP根据一一对应的点从conv5映射回原图，根据设计不同的固定初始尺度训练一个网络，就是给它大小不同（但设计固定）的region图，然后根据与ground truth的覆盖率给它正负标签，让它学习里面是否有object即可。  
由于检测网后面会做位置精修，为了降低模型复杂度，减少候选框数量，采用深度网络，固定尺度变化，固定scale ratio变化（长宽比），固定采样方式（在最后一层的feature map上采样）  
RPN是在CNN训练得到的用于分类任务的feature map基础上，对所有可能的候选框进行判别。由于后续还有位置精修步骤，所以候选框实际比较稀疏    
![](https://i.imgur.com/yYDmjE8.png)  
• 在feature map上滑动窗口。   
• 建一个神经网络用于物体分类（x_class）+框位置的回归(x_reg),对于特征图中的每一个位置，RPN会做k次预测。因此，RPN将输出4×k个坐标和每个位置上2×k个得分,faster RCNN中，使用3个不同宽高比的3个不同大小的锚点框（k=9个锚点框）       
• 滑动窗口的位置提供了物体的大体位置信息。   
• 框的回归修正框的位置，使其与对应的bbox位置更相近。  
![](https://i.imgur.com/9KIjH14.png)  

**原图尺度**：原始输入的大小。不受任何限制，不影响性能。  
**归一化尺度**：输入特征提取网络的大小，在测试时设置，源码中opts.test_scale=600。anchor在这个尺度上设定。这个参数和anchor的相对大小决定了想要检测的目标范围。  
**网络输入尺度**：输入特征检测网络的大小，在训练时设置，源码中为224\*224。  

## 7. [SSD](https://zhuanlan.zhihu.com/p/33544892)
全名是Single Shot MultiBox Detector  
### 网络结构
![](https://i.imgur.com/dzNRgUR.jpg)  
### SSD特点
1. 相比Yolo，SSD采用CNN来直接进行检测，而不是像Yolo那样在全连接层之后做检测  
2. SSD提取了不同尺度的特征图来做检测，大尺度特征图（较靠前的特征图）可以用来检测小物体，而小尺度特征图（较靠后的特征图）用来检测大物体  
3. SSD采用了不同尺度和长宽比的先验框（Prior boxes, Default boxes，在Faster R-CNN中叫做锚，Anchors） 
4. SSD将背景也当做了一个特殊的类别，如果检测目标共有 c 个类别，SSD其实需要预测 c+1 个置信度值，其中第一个置信度指的是不含目标或者属于背景的评分  
5. 真实预测值其实只是边界框相对于先验框的转换值  
### SSD中基础概念
先验框（Default boxes，在Faster R-CNN中叫做锚，Anchors）：feature map的每个小格(cell)上都有一系列固定大小的box，对于每个default box都需要预测c个类别score和4个offset  
Priorbox：类比于conv2d，为生成default box的option    
边界框（bounding boxes）/预测框：根据先验框偏移生成的boxs  
标记框（ground truth）：图片中实际标记的位置和label  
confidence：类别置信度，每个default box 生成21个类别confidence  
location：包含4个值 (cx, cy, w, h) ，分别表示box的中心坐标以及宽高  
feature map cell：指feature map中每一个小格子
先验框位置用 d=(d^{cx}, d^{cy}, d^w, d^h) 表示  
边界框用 b=(b^{cx}, b^{cy}, b^w, b^h)表示 

### 先验框生成方式
SSD网络中选取不同大小的feature map用来检测目标，在feature map上生成固定大小和长宽比的先验框用来检测目标，大的特征图来中先验框用来检测相对较小的目标，而小的特征图中先验框负责检测大目标  
![](https://i.imgur.com/1vR0I3O.jpg)  
不同特征图设置的先验框数目不同，同一个特征图上每个单元设置的先验框数目是相同的，这里的数目指的是一个单元的先验框数目

### 特征图选取
论文中使用的基础网络为VGG16，并对VGG16网络进行改造，将VGG16的全连接层fc6和fc7转换成 3\times3 卷积层 conv6和 1\times1 卷积层conv7，同时将池化层pool5由原来的stride=2的 2\times 2 变成stride=1的 3\times 3，然后移除dropout层和fc8层，并新增一系列卷积层，在检测数据集上做finetuing  
特征图选取了Conv4_3，Conv7，Conv8_2，Conv9_2，Conv10_2，Conv11_2作为检测所用的特征图，其大小分别是 (38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)  
其中conv4_3层特征图大小是 38\times38 ，但是该层比较靠前，其norm较大，所以在其后面增加了一个L2 Normalization层，以保证和后面的检测层差异不是很大。

### 先验框尺度确定
对于先验框的尺度，其遵守一个线性递增规则：随着特征图大小降低，先验框尺度线性增加  
![](https://i.imgur.com/clgFCu3.png)  
其中 m 指的特征图个数，但却是 5 ，因为第一层（Conv4_3层）是单独设置的，论文中其先验框的尺度比例一般设置为 s_{min}/2=0.1，尺度为 300\times 0.1=30    
s_k 表示先验框大小相对于图片的比例，而 s_{min} 和 s_{max} 表示比例的最小值与最大值，paper里面取0.2和0.9  
其他5个特征图尺度比例为：0.2，0.37，0.54，0.71，0.88，尺度为 60,111, 162,213,264  
那么各个特征图的先验框尺度为 30,60,111, 162,213,264  

### 先验框长宽比选取
对于长宽比，一般选取 a_r\in \{1,2,3,\frac{1}{2},\frac{1}{3}\} ，对于特定的长宽比，按如下公式计算先验框的宽度与高度：  
w^a_{k}=s_k\sqrt{a_r},\space h^a_{k}=s_k/\sqrt{a_r}    
**注意：这里的s_k指的是先验框实际尺度**
默认情况下，每个特征图会有一个 a_r=1 且尺度为 s_k 的先验框，除此之外，还会设置一个尺度为 s'_{k}=\sqrt{s_k s_{k+1}} 且 a_r=1 的先验框，这样每个特征图都设置了两个长宽比为1但大小不同的正方形先验框。因此，每个特征图一共有 6 个先验框 \{1,2,3,\frac{1}{2},\frac{1}{3},1'\}      
**注意：最后一个特征图需要参考一个虚拟 s_{m+1}=300\times105/100=315 来计算 s'_{m}**  
但在论文中，Conv4_3，Conv10_2和Conv11_2层仅使用4个先验框，它们不使用长宽比为 3,\frac{1}{3} 的先验框。  

### 先验框中心点计算
每个单元的先验框的中心点分布在各个单元的中心，即   
(\frac{i+0.5}{|f_k|},\frac{j+0.5}{|f_k|}),  
i,j\in[0, |f_k|) ，  
其中 |f_k| 为特征图的大小。 

### 预测过程
获得原始图片后，使用基础网络对图片特征进行提取，获取不同大小的feature；  
下面以一个 5\times5 大小的特征图展示检测过程：
![](https://i.imgur.com/Aj2Ox7t.jpg)  
检测分为3个步骤：   
1. 先验框生成：Priorbox是生成先验框option   
2. 计算预测框location：采用一次 3\times3 卷积来进行完成  
3. 计算类别置信度：采用一次 3\times3 卷积来进行完成  
令 n_k 为该特征图所采用的先验框数目，那么类别置信度需要的卷积核数量为 n_k\times c ，而边界框位置需要的卷积核数量为 n_k\times 4 。  
由于每个先验框都会预测一个边界框，所以SSD300一共可以预测 38\times38\times4+19\times19\times6+10\times10\times6+5\times5\times6+3\times3\times4+1\times1\times4=8732 个边界框，这是一个相当庞大的数字，所以说SSD本质上是密集采样。  

对于每个预测框，首先根据类别置信度确定其类别（置信度最大者）与置信度值，并过滤掉属于背景的预测框。然后根据置信度阈值（如0.5）过滤掉阈值较低的预测框。对于留下的预测框进行解码，根据先验框得到其真实的位置参数（解码后一般还需要做clip，防止预测框位置超出图片）。解码之后，一般需要根据置信度进行降序排列，然后仅保留top-k（如400）个预测框。最后就是进行NMS算法，过滤掉那些重叠度较大的预测框。最后剩余的预测框就是检测结果了。  

### location偏移量计算
SSD中ground truth和预测框都是计算相对prior box的相对偏移量，优化方向为减小预测框和ground truth偏移量的差值

先验框位置用 d=(d^{cx}, d^{cy}, d^w, d^h) 表示，边界框用 b=(b^{cx}, b^{cy}, b^w, b^h)表示  
误差计算公式为：  
l^{cx} = (b^{cx} - d^{cx})/d^w, \space l^{cy} = (b^{cy} - d^{cy})/d^h  

l^{w} = \log(b^{w}/d^w), \space l^{h} = \log(b^{h}/d^h)  
我们称上面这个过程为边界框的编码（encode）  

预测时，你需要反向这个过程，即进行解码（decode），从预测值偏差值 l 中得到边界框的真实位置 b：  
b^{cx}=d^w l^{cx} + d^{cx}, \space b^{cy}=d^y l^{cy} + d^{cy}  

b^{w}=d^w \exp(l^{w}), \space b^{h}=d^h \exp(l^{h})  

ground truth和default box之间的偏移误差同样使用上面公式计算。  

### 训练过程
1. 先验框匹配--确定正/负样本  
在训练过程中，首先要确定训练图片中的ground truth（真实目标）与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它。  
先验框与ground truth的匹配原则：
* a.一个先验框只能匹配一个ground truth，但一个ground truth可以匹配多个先验框  
* b.对于图片中每个ground truth，找到与其IOU最大的先验框，该先验框与其匹配，这样，可以保证每个ground truth一定与某个先验框匹配。  
* c.对于剩余的未匹配先验框，若某个ground truth的 \text{IOU} 大于某个阈值（一般是0.5），那么该先验框也与这个ground truth进行匹配  
* d.如果多个ground truth与某个先验框 \text{IOU} 大于阈值，那么先验框只与IOU最大的那个先验框进行匹配,该条原则要在遵循a、b两条原则之后执行，确保某个ground truth一定有一个先验框与之匹配      
* 与ground truth匹配的先验框为正样本，反之，若一个先验框没有与任何ground truth进行匹配，那么该先验框只能与背景匹配，就是负样本。
尽管一个ground truth可以与多个先验框匹配，但是ground truth相对先验框还是太少了，所以负样本相对正样本会很多。为了保证正负样本尽量平衡，SSD采用了hard negative mining，就是对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3  

2. 损失函数  
训练样本确定了，然后就是损失函数了。损失函数定义为位置误差（locatization loss， loc）与置信度误差（confidence loss, conf）的加权和  
L(x, c, l, g) = \frac{1}{N}(L_{conf}(x,c) + \alpha L_{loc}(x,l,g))  
\alpha 为权重系数 ，通常设置为1  
N 是先验框的正样本数量  
x^p_{ij}\in \{ 1,0 \} 为一个指示参数，当 x^p_{ij}= 1 时表示第 i 个先验框与第 j 个ground truth匹配，并且ground truth的类别为 p   
l 为先验框的所对应边界框的位置预测值  
g 是ground truth的位置参数  

* 对于位置误差，其采用Smooth L1 loss，定义如下：
![location loss](https://i.imgur.com/dTrC4PA.jpg)  
![](https://i.imgur.com/3BFH6Zu.jpg)   
相当于ground truth于default box偏差 和 predict box于default box偏差 的差值，再求smooth L1。  
由于 x^p_{ij} 的存在，所以位置误差仅针对正样本进行计算  

* 置信度误差,采用softmax loss  
![](https://i.imgur.com/14scf3S.jpg)  


## 8. yolo-v1
YOLO创造性的将物体检测任务直接当作回归问题（regression problem）来处理，将候选区和检测两个阶段合二为一。只需一眼就能知道每张图像中有哪些物体以及物体的位置  
事实上，YOLO也并没有真正的去掉候选区，而是直接将输入图片划分成7x7=49个网格，每个网格预测两个边界框，一共预测49x2=98个边界框。可以近似理解为在输入图片上粗略的选取98个候选区，这98个候选区覆盖了图片的整个区域，进而用回归预测这98个候选框对应的边界框  











