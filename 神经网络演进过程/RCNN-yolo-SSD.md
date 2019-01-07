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
RCNN-Faster RCNN是基于Region Proposal的算法，是two-stage的，需要先使用启发式方法（selective search）或者CNN网络（RPN）产生Region Proposal，然后再在Region Proposal上做分类与回归  
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
先验框位置用 ![](https://latex.codecogs.com/gif.latex?d=(d^{cx},&space;d^{cy},&space;d^w,&space;d^h)) 表示  
边界框用 ![](https://latex.codecogs.com/gif.latex?b=(b^{cx},&space;b^{cy},&space;b^w,&space;b^h))表示 

### 先验框生成方式
SSD网络中选取不同大小的feature map用来检测目标，在feature map上生成固定大小和长宽比的先验框用来检测目标，大的特征图来中先验框用来检测相对较小的目标，而小的特征图中先验框负责检测大目标  
![](https://i.imgur.com/1vR0I3O.jpg)  
不同特征图设置的先验框数目不同，同一个特征图上每个单元设置的先验框数目是相同的，这里的数目指的是一个单元的先验框数目

### 特征图选取
论文中使用的基础网络为VGG16，并对VGG16网络进行改造，将VGG16的全连接层fc6和fc7转换成 ![](https://latex.codecogs.com/gif.latex?3\times3) 卷积层 conv6和 ![](https://latex.codecogs.com/gif.latex?1\times1) 卷积层conv7，同时将池化层pool5由原来的stride=2的 ![](https://latex.codecogs.com/gif.latex?2\times2) 变成stride=1的 ![](https://latex.codecogs.com/gif.latex?3\times&space;3)，然后移除dropout层和fc8层，并新增一系列卷积层，在检测数据集上做finetuing  
特征图选取了Conv4_3，Conv7，Conv8_2，Conv9_2，Conv10_2，Conv11_2作为检测所用的特征图，其大小分别是 (38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)  
其中conv4_3层特征图大小是 ![](https://latex.codecogs.com/gif.latex?38\times38) ，但是该层比较靠前，其norm较大，所以在其后面增加了一个L2 Normalization层，以保证和后面的检测层差异不是很大。

### 先验框尺度确定
对于先验框的尺度，其遵守一个线性递增规则：随着特征图大小降低，先验框尺度线性增加  
![](https://i.imgur.com/clgFCu3.png)  
其中 m 指的特征图个数，但却是 5 ，因为第一层（Conv4_3层）是单独设置的，论文中其先验框的尺度比例一般设置为 ![](https://latex.codecogs.com/gif.latex?s_{min}/2=0.1)，尺度为 ![](https://latex.codecogs.com/gif.latex?300\times&space;0.1=30)    
![](https://latex.codecogs.com/gif.latex?s_k) 表示先验框大小相对于图片的比例，而 ![](https://latex.codecogs.com/gif.latex?s_{min}) 和 ![](https://latex.codecogs.com/gif.latex?s_{max}) 表示比例的最小值与最大值，paper里面取0.2和0.9  
其他5个特征图尺度比例为：0.2，0.37，0.54，0.71，0.88，尺度为 60,111, 162,213,264  
那么各个特征图的先验框尺度为 30,60,111, 162,213,264  

### 先验框长宽比选取
对于长宽比，一般选取 ![](https://latex.codecogs.com/gif.latex?a_r\in&space;\{1,2,3,\frac{1}{2},\frac{1}{3}\}) ，对于特定的长宽比，按如下公式计算先验框的宽度与高度：  
![](https://latex.codecogs.com/gif.latex?w^a_{k}=s_k\sqrt{a_r},\space&space;h^a_{k}=s_k/\sqrt{a_r})    
**注意：这里的s_k指的是先验框实际尺度**
默认情况下，每个特征图会有一个 ![](https://latex.codecogs.com/gif.latex?a_r=1) 且尺度为 ![](https://latex.codecogs.com/gif.latex?s_k) 的先验框，除此之外，还会设置一个尺度为 ![](https://latex.codecogs.com/gif.latex?s'_{k}=\sqrt{s_k&space;s_{k&plus;1}}) 且 ![](https://latex.codecogs.com/gif.latex?a_r=1) 的先验框，这样每个特征图都设置了两个长宽比为1但大小不同的正方形先验框。因此，每个特征图一共有 6 个先验框 ![](https://latex.codecogs.com/gif.latex?\{1,2,3,\frac{1}{2},\frac{1}{3},1'\})      
**注意：最后一个特征图需要参考一个虚拟 ![](https://latex.codecogs.com/gif.latex?s_{m&plus;1}=300\times105/100=315) 来计算 ![](https://latex.codecogs.com/gif.latex?s'_{m})**  
但在论文中，Conv4_3，Conv10_2和Conv11_2层仅使用4个先验框，它们不使用长宽比为 3,![](\frac{1}{3}) 的先验框。  

### 先验框中心点计算
每个单元的先验框的中心点分布在各个单元的中心，即   
![](https://latex.codecogs.com/gif.latex?(\frac{i&plus;0.5}{|f_k|},\frac{j&plus;0.5}{|f_k|}),)  
![](https://latex.codecogs.com/gif.latex?i,j\in[0,&space;|f_k|)&space;，)  
其中 ![](https://latex.codecogs.com/gif.latex?|f_k|) 为特征图的大小。 

### 预测过程
获得原始图片后，使用基础网络对图片特征进行提取，获取不同大小的feature；  
下面以一个 ![](https://latex.codecogs.com/gif.latex?5\times5) 大小的特征图展示检测过程：
![](https://i.imgur.com/Aj2Ox7t.jpg)  
检测分为3个步骤：   
1. 先验框生成：Priorbox是生成先验框option   
2. 计算预测框location：采用一次 ![](https://latex.codecogs.com/gif.latex?3\times3) 卷积来进行完成  
3. 计算类别置信度：采用一次 ![](https://latex.codecogs.com/gif.latex?3\times3) 卷积来进行完成  
令 ![](https://latex.codecogs.com/gif.latex?n_k) 为该特征图所采用的先验框数目，那么类别置信度需要的卷积核数量为 n_k\times c ，而边界框位置需要的卷积核数量为 ![](https://latex.codecogs.com/gif.latex?n_k\times&space;4) 。  
由于每个先验框都会预测一个边界框，所以SSD300一共可以预测 ![](https://latex.codecogs.com/gif.latex?38\times38\times4&plus;19\times19\times6&plus;10\times10\times6&plus;5\times5\times6&plus;3\times3\times4&plus;1\times1\times4=8732) 个边界框，这是一个相当庞大的数字，所以说SSD本质上是密集采样。  

对于每个预测框，首先根据类别置信度确定其类别（置信度最大者）与置信度值，并过滤掉属于背景的预测框。然后根据置信度阈值（如0.5）过滤掉阈值较低的预测框。对于留下的预测框进行解码，根据先验框得到其真实的位置参数（解码后一般还需要做clip，防止预测框位置超出图片）。解码之后，一般需要根据置信度进行降序排列，然后仅保留top-k（如400）个预测框。最后就是进行NMS算法，过滤掉那些重叠度较大的预测框。最后剩余的预测框就是检测结果了。  

### location偏移量计算
SSD中ground truth和预测框都是计算相对prior box的相对偏移量，优化方向为减小预测框和ground truth偏移量的差值

先验框位置用 ![](https://latex.codecogs.com/gif.latex?d=(d^{cx},&space;d^{cy},&space;d^w,&space;d^h)) 表示，边界框用 ![](https://latex.codecogs.com/gif.latex?b=(b^{cx},&space;b^{cy},&space;b^w,&space;b^h))表示  
误差计算公式为：  
![](https://latex.codecogs.com/gif.latex?l^{cx}&space;=&space;(b^{cx}&space;-&space;d^{cx})/d^w,&space;\space&space;l^{cy}&space;=&space;(b^{cy}&space;-&space;d^{cy})/d^h) 

![](https://latex.codecogs.com/gif.latex?l^{w}&space;=&space;\log(b^{w}/d^w),&space;\space&space;l^{h}&space;=&space;\log(b^{h}/d^h))  
我们称上面这个过程为边界框的编码（encode）  

预测时，你需要反向这个过程，即进行解码（decode），从预测值偏差值 l 中得到边界框的真实位置 b：  
![](https://latex.codecogs.com/gif.latex?b^{cx}=d^w&space;l^{cx}&space;&plus;&space;d^{cx},&space;\space&space;b^{cy}=d^y&space;l^{cy}&space;&plus;&space;d^{cy})  

![](https://latex.codecogs.com/gif.latex?b^{w}=d^w&space;\exp(l^{w}),&space;\space&space;b^{h}=d^h&space;\exp(l^{h}))  

ground truth和default box之间的偏移误差同样使用上面公式计算。  

### 训练过程
1. 先验框匹配--确定正/负样本  
在训练过程中，首先要确定训练图片中的ground truth（真实目标）与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它。  
先验框与ground truth的匹配原则：
* a.一个先验框只能匹配一个ground truth，但一个ground truth可以匹配多个先验框  
* b.对于图片中每个ground truth，找到与其IOU最大的先验框，该先验框与其匹配，这样，可以保证每个ground truth一定与某个先验框匹配。  
* c.对于剩余的未匹配先验框，若某个ground truth的 IOU 大于某个阈值（一般是0.5），那么该先验框也与这个ground truth进行匹配  
* d.如果多个ground truth与某个先验框 IOU 大于阈值，那么先验框只与IOU最大的那个先验框进行匹配,该条原则要在遵循a、b两条原则之后执行，确保某个ground truth一定有一个先验框与之匹配      
* 与ground truth匹配的先验框为正样本，反之，若一个先验框没有与任何ground truth进行匹配，那么该先验框只能与背景匹配，就是负样本。
尽管一个ground truth可以与多个先验框匹配，但是ground truth相对先验框还是太少了，所以负样本相对正样本会很多。为了保证正负样本尽量平衡，SSD采用了hard negative mining，就是对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3  

2. 损失函数  
训练样本确定了，然后就是损失函数了。损失函数定义为位置误差（locatization loss， loc）与置信度误差（confidence loss, conf）的加权和  
![](https://latex.codecogs.com/gif.latex?L(x,&space;c,&space;l,&space;g)&space;=&space;\frac{1}{N}(L_{conf}(x,c)&space;&plus;&space;\alpha&space;L_{loc}(x,l,g)))  
![](https://latex.codecogs.com/gif.latex?\alpha) 为权重系数 ，通常设置为1  
N 是先验框的正样本数量  
![](https://latex.codecogs.com/gif.latex?x^p_{ij}\in&space;\{&space;1,0&space;\}) 为一个指示参数，当 ![](https://latex.codecogs.com/gif.latex?x^p_{ij}=&space;1) 时表示第 i 个先验框与第 j 个ground truth匹配，并且ground truth的类别为 p   
l 为先验框的所对应边界框的位置预测值  
g 是ground truth的位置参数  

* 对于位置误差，其采用Smooth L1 loss，定义如下：
![location loss](https://i.imgur.com/dTrC4PA.jpg)  
![](https://i.imgur.com/3BFH6Zu.jpg)   
相当于ground truth于default box偏差 和 predict box于default box偏差 的差值，再求smooth L1。  
由于 ![](https://latex.codecogs.com/gif.latex?x^p_{ij}) 的存在，所以位置误差仅针对正样本进行计算  

* 置信度误差,采用softmax loss  
![](https://i.imgur.com/14scf3S.jpg)  


## 8. yolo-v1
YOLO-V1（You Only Look Once: Unified, Real-Time Object Detection）  

**网络结构**  
![](https://i.imgur.com/FceuWmv.jpg)   
网络架构受图像分类模型GoogLeNet的启发。网络有24个卷积层，后面是2个全连接层。我们只使用1×1降维层，后面是3×3卷积层，而不是GoogLeNet使用的Inception模块，（快速YOLO使用具有较少卷积层（9层而不是24层）的神经网络），对于卷积层和全连接层，采用Leaky ReLU激活函数： max(x, 0.1x)，但是最后一层却采用线性激活函数，最终输出是7×7×30的预测张量  
![](https://i.imgur.com/4KASkwC.jpg)  
网络的最后输出为 7\times 7\times 30 大小的张量，这个张量所代表的具体含义如图所示，对于每一个单元格，前20个元素是类别概率值，然后2个元素是边界框置信度，两者相乘可以得到类别置信度，最后8个元素是边界框的 (x, y,w,h)   


**设计理念**  
![](https://i.imgur.com/R0jPTT8.jpg)   
（1）把图像缩放到448X448，（2）在图上运行卷积网络，（3）根据模型的置信度对检测结果进行阈值处理 

S：Yolo的CNN网络将输入的图片分割成 S\times S 网格，然后每个单元格负责去检测那些中心点落在该格子内的目标  
B：每个单元格会预测 B 个边界框（bounding box）以及边界框的置信度（confidence score）  
c:置信度（confidence score）,所谓置信度其实包含两个方面，一是这个边界框含有目标的可能性大小，二是这个边界框的准确度。前者记为 Pr(object) (取值为0或者1)，当该边界框是背景时（即不包含目标），此时 Pr(object)=0 。而当该边界框包含目标时， Pr(object)=1 。边界框的准确度可以用预测框与实际框（ground truth）的IOU（intersection over union，交并比）来表征，记为![](https://latex.codecogs.com/gif.latex?\text{IOU}^{truth}_{pred}) 。因此置信度可以定义为 Pr(object) \* ![](https://latex.codecogs.com/gif.latex?\text{IOU}^{truth}_{pred}) 。  
(x, y,w,h) ：(x,y) 是边界框的中心坐标，而 w 和 h 是边界框的宽与高，中心坐标的预测值 (x,y) 是相对于每个单元格左上角坐标点的偏移值，并且单位是相对于单元格大小的；边界框的 w 和 h 预测值是相对于整个图片的宽与高的比例，(x, y,w,h)大小在 [0,1] 范围  
边界框：(x,y,w,h,c) ，其中前4个表征边界框的大小与位置，而最后一个值是置信度。  
C：分类数
![](https://latex.codecogs.com/gif.latex?Pr(class_{i}|object)) ：对于每一个单元格其还要给出预测出 C 个类别概率值，其表征的是由该单元格负责预测的边界框其目标属于各个类别的概率；但是这些概率值其实是在各个边界框置信度下的条件概率，即 Pr(class_{i}|object)；**不管一个单元格预测多少个边界框，其只预测一组类别概率值，这是Yolo算法的一个缺点，在后来的改进版本中，Yolo9000是把类别概率预测值与边界框是绑定在一起的**  
边界框类别置信度（class-specific confidence scores）: ![](https://latex.codecogs.com/gif.latex?\inline&space;Pr(class_{i}|object)*Pr(object)*\text{IOU}^{truth}_{pred}=Pr(class_{i})*\text{IOU}^{truth}_{pred})   
边界框类别置信度表征的是该边界框中目标属于各个类别的可能性大小以及边界框匹配目标的好坏，一般会根据类别置信度来过滤网络的预测框  

**训练**  
在训练之前，先在ImageNet上进行了预训练，其预训练的分类模型采用图8中前20个卷积层，然后添加一个average-pool层和全连接层。预训练之后，在预训练得到的20层卷积层之上加上随机初始化的4个卷积层和2个全连接层。由于检测任务一般需要更高清的图片，所以将网络的输入从224x224增加到了448x448。整个网络的流程如下图所示：  
![](https://i.imgur.com/v6mtb4N.jpg)  

> **误差函数计算:**  
> Yolo算法将目标检测看成回归问题，所以采用的是均方差损失函数。但是对不同的部分采用了不同的权重值。  
> 首先区分定位误差和分类误差。对于定位误差，即边界框坐标预测误差，采用较大的权重 ![](https://latex.codecogs.com/gif.latex?\inline&space;\lambda&space;_{coord}=5) 。然后其区分不包含目标的边界框与含有目标的边界框的置信度，对于前者，采用较小的权重值 ![](https://latex.codecogs.com/gif.latex?\inline&space;\lambda&space;_{noobj}=0.5) 。其它权重值均设为1。然后采用均方误差，其同等对待大小不同的边界框，但是实际上较小的边界框的坐标误差应该要比较大的边界框要更敏感。为了保证这一点，将网络的边界框的宽与高预测改为对其平方根的预测，即预测值变为 ![](https://latex.codecogs.com/gif.latex?\inline&space;(x,y,\sqrt{w},&space;\sqrt{h}))。  
>YOLO为每个网格单元预测多个边界框。在训练时，每个目标我们只需要一个边界框预测器来负责。若某预测器的预测值与目标的实际值的IOU值最高，则这个预测器被指定为“负责”预测该目标。这导致边界框预测器的专业化。每个预测器可以更好地预测特定大小，方向角，或目标的类别，从而改善整体召回率。但如果一个单元格内存在多个目标怎么办，其实这时候Yolo算法就只能选择其中一个来训练，这也是Yolo算法的缺点之一。  
> 对于不存在对应目标的边界框，其误差项就是只有置信度，坐标项误差是没法计算的。而只有当一个单元格内确实存在目标时，才计算分类误差项，否则该项也是无法计算的。  
> 综上讨论，最终的损失函数计算如下：  
> ![](https://i.imgur.com/9YRbUIv.jpg)  
> ![](https://latex.codecogs.com/gif.latex?\inline&space;1^{obj}_{ij}) 指的是第 i 个单元格存在目标，且该单元格中的第 j 个边界框负责预测该目标  
> ![](https://latex.codecogs.com/gif.latex?\inline&space;1^{obj}_{i}) 指的是第 i 个单元格存在目标  
> 置信度的target值 ![](https://latex.codecogs.com/gif.latex?\inline&space;C_i) ，如果是不存在目标，此时由于 Pr(object)=0，那么 ![](https://latex.codecogs.com/gif.latex?\inline&space;C_i=0) 。如果存在目标， Pr(object)=1 ，此时需要确定 ![](https://latex.codecogs.com/gif.latex?\inline&space;\text{IOU}^{truth}_{pred}) ，当然你希望最好的话，可以将IOU取1，这样 ![](https://latex.codecogs.com/gif.latex?\inline&space;C_i=1) ，但是在YOLO实现中，使用了一个控制参数rescore（默认为1），当其为1时，IOU不是设置为1，而就是计算truth和pred之间的真实IOU。不过很多复现YOLO的项目还是取 ![](https://latex.codecogs.com/gif.latex?\inline&space;C_i=1)   

**网络预测**  
对于一张输入图片。根据前面的分析，最终的网络输出是 ![](https://latex.codecogs.com/gif.latex?\inline&space;7\times&space;7&space;\times&space;30) ，但是我们可以将其分割成三个部分：类别概率部分为 [7, 7, 20] ，置信度部分为 [7,7,2] ，而边界框部分为 [7,7,2,4] （对于这部分不要忘记根据原始图片计算出其真实值）。然后将前两项相乘（矩阵 [7, 7, 20] 乘以 [7,7,2] 可以各补一个维度来完成 [7,7,1,20] x [7,7,2,1] ）可以得到类别置信度值为 [7, 7,2,20] ，这里总共预测了 7*7*2=98 个边界框  

网格设计强化了边界框预测中的空间多样性。通常一个目标落在哪一个网格单元中是很明显的，而网络只能为每个目标预测一个边界框。然而，一些大的目标或接近多个网格单元的边界的目标能被多个网格单元定位。非极大值抑制可以用来修正这些多重检测。

**优缺点**  
缺点：    
1）与fast RCNN相比，YOLO的误差分析显示YOLO产生大量的定位误差。2）与基于候选区域的方法相比，YOLO具有相对较低的召回率；3）如果两个小目标同时落入一个格子中，模型也只能预测一个；4）  
优点：  
1）YOLO简化了整个目标检测流程，速度的提升也很大（2）YOLO采用全图信息来进行预测。与滑动窗口方法和region proposal-based方法不同，YOLO在训练和预测过程中可以利用全图信息。Fast R-CNN检测方法会错误的将背景中的斑块检测为目标，原因在于Fast R-CNN在检测中无法看到全局图像。相对于Fast R-CNN，YOLO背景预测错误率低一半。（3）YOLO可以学习到目标的概括信息（generalizable representation），具有一定普适性。我们采用自然图片训练YOLO，然后采用艺术图像来预测。YOLO比其它目标检测方法（DPM和R-CNN）准确率高很多  


## [yolo v2](https://zhuanlan.zhihu.com/p/35325884)
相同的YOLOv2模型可以运行在不同的大小的图片上，提供速度和精度之间的轻松权衡  
YOLOv1在物体定位方面（localization）不够准确，并且召回率（recall）较低。YOLOv2共提出了几种改进策略来提升YOLO模型的定位准确度和召回率，从而提高mAP，YOLOv2在改进中遵循一个原则：保持检测速度，这也是YOLO模型的一大优势。YOLOv2的改进策略如图所示  
![](https://i.imgur.com/D5aOUGn.jpg)  

**Batch Normalization**  
Batch Normalization可以提升模型收敛速度，而且可以起到一定正则化效果，降低模型的过拟合。在YOLOv2中，每个卷积层后面都添加了Batch Normalization层，并且不再使用droput。使用Batch Normalization后，YOLOv2的mAP提升了2.4%。  

**高分辨率分类器 High Resolution Classifier**  
目前大部分的检测模型都会在先在ImageNet分类数据集上预训练模型的主体部分（CNN特征提取器），由于历史原因，ImageNet分类模型基本采用大小为 224 x 224 的图片作为输入，分辨率相对较低，不利于检测模型  
YOLOv1在采用 224x224 分类模型预训练后，将分辨率增加至 448x448 ，并使用这个高分辨率在检测数据集上finetune。但是直接切换分辨率，检测模型可能难以快速适应高分辨率。  
YOLOv2增加了在ImageNet数据集上使用 448x448 输入来finetune分类网络这一中间过程（10 epochs），这可以使得模型在检测数据集上finetune之前已经适用高分辨率输入。使用高分辨率分类器后，YOLOv2的mAP提升了约4%。  

**使用anchor boxs进行卷积(Convolutional With Anchor Boxes)**  
在YOLOv1中，输入图片最终被划分为 7\times7 网格，每个单元格预测2个边界框。YOLOv1最后采用的是全连接层直接对边界框进行预测，其中边界框的宽与高是相对整张图片大小的，而由于各个图片中存在不同尺度和长宽比（scales and ratios）的物体，YOLOv1在训练过程中学习适应不同物体的形状是比较困难的，这也导致YOLOv1在精确定位方面表现较差  
YOLOv2借鉴了Faster R-CNN中RPN网络的先验框（anchor boxes，prior boxes，SSD也采用了先验框）策略。RPN对CNN特征提取器得到的特征图（feature map）进行卷积来预测每个位置的边界框以及置信度（是否含有物体），并且各个位置设置不同尺度和比例的先验框，所以RPN预测的是边界框相对于先验框的offsets值，采用先验框使得模型更容易学习。所以YOLOv2移除了YOLOv1中的全连接层而采用了卷积和anchor boxes来预测边界框。  
为了使检测所用的特征图分辨率更高，移除其中的一个pool层。  
在检测模型中，YOLOv2不是采用 448x448 图片作为输入，而是采用 416x416 大小。因为YOLOv2模型下采样的总步长为32 ，对于 416x416 大小的图片，最终得到的特征图大小为 13x13 ，维度是奇数，这样特征图恰好只有一个中心位置。对于一些大物体，它们中心点往往落入图片中心位置，此时使用特征图的一个中心点去预测这些物体的边界框相对容易些。所以在YOLOv2设计中要保证最终的特征图有奇数个位置。  
对于YOLOv1，每个cell都预测2个boxes，每个boxes包含5个值： (x, y, w, h, c) ，前4个值是边界框位置与大小，最后一个值是置信度（confidence scores，包含两部分：含有物体的概率以及预测框与ground truth的IOU）。但是每个cell只预测一套分类概率值（class predictions，其实是置信度下的条件概率值）,供2个boxes共享。YOLOv2使用了anchor boxes之后，每个位置的各个anchor box都单独预测一套分类概率值，这和SSD比较类似（但SSD没有预测置信度，而是把background作为一个类别来处理）  
使用anchor boxes之后，YOLOv2的mAP有稍微下降（这里下降的原因，猜想是YOLOv2虽然使用了anchor boxes，但是依然采用YOLOv1的训练方法）。YOLOv1只能预测98个边界框（ 7x7x2 ），而YOLOv2使用anchor boxes之后可以预测上千个边界框（ 13x13xnum_anchors ）。所以使用anchor boxes之后，YOLOv2的召回率大大提升，由原来的81%升至88%。  

**尺度聚类（Dimension Clusters）**  
在Faster R-CNN和SSD中，先验框的维度（长和宽）都是手动设定的，带有一定的主观性。如果选取的先验框维度比较合适，那么模型更容易学习，从而做出更好的预测。因此，YOLOv2采用k-means聚类方法对训练集中的边界框做了聚类分析。因为设置先验框的主要目的是为了使得预测框与ground truth的IOU更好，所以聚类分析时选用box与聚类中心box之间的IOU值作为距离指标  
d(box, centroid) = 1 - IOU(box, centroid)  
下图为在VOC和COCO数据集上的聚类分析结果，随着聚类中心数目的增加，平均IOU值（各个边界框与聚类中心的IOU的平均值）是增加的，但是综合考虑模型复杂度和召回率，作者最终选取5个聚类中心作为先验框，其相对于图片的大小如右边图所示。对于两个数据集，5个先验框的width和height如下所示（来源：YOLO源码的cfg文件）：  
> COCO: (0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)  
VOC: (1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)   

这里先验框的大小是相对于预测的特征图大小（ 13x13 ）,对比两个数据集，也可以看到COCO数据集上的物体相对小点。   
![](https://i.imgur.com/eynWhEf.jpg)  

**New Network: Darknet-19**  
YOLOv2采用了一个新的基础模型（特征提取器），称为Darknet-19，包括19个卷积层和5个maxpooling层，如图所示。Darknet-19与VGG16模型设计原则是一致的，主要采用 3x3 卷积，采用 2x2 的maxpooling层之后，特征图维度降低2倍，而同时将特征图的channles增加两倍。与NIN(Network in Network)类似，Darknet-19最终采用global avgpooling做预测，并且在 3x3 卷积之间使用 1x1 卷积来压缩特征图channles以降低模型计算量和参数。Darknet-19每个卷积层后面同样使用了batch norm层以加快收敛速度，降低模型过拟合。在ImageNet分类数据集上，Darknet-19的top-1准确度为72.9%，top-5准确度为91.2%，但是模型参数相对小一些。使用Darknet-19之后，YOLOv2的mAP值没有显著提升，但是计算量却可以减少约33%。  
![](https://i.imgur.com/eR8zyqz.jpg)   

**直接位置预测(Direct location prediction)**  
YOLOv2借鉴RPN网络使用anchor boxes来预测边界框相对先验框的offsets。边界框的实际中心位置 (x,y) ，需要根据预测的坐标偏移值 (t_x, t_y) ，先验框的尺度 (w_a, h_a) 以及中心坐标 (x_a, y_a) （特征图每个位置的中心点）来计算：  
![](https://latex.codecogs.com/gif.latex?\inline&space;\\x&space;=&space;(t_x\times&space;w_a)-x_a)  
![](https://latex.codecogs.com/gif.latex?\inline&space;\\y=(t_y\times&space;h_a)&space;-&space;y_a)  
但是上面的公式是无约束的，预测的边界框很容易向任何方向偏移，如当 t_x=1 时边界框将向右偏移先验框的一个宽度大小，而当 t_x=-1 时边界框将向左偏移先验框的一个宽度大小，因此每个位置预测的边界框可以落在图片任何位置，这导致模型的不稳定性，在训练时需要很长时间来预测出正确的offsets。  
所以，YOLOv2弃用了这种预测方式，而是沿用YOLOv1的方法，就是预测边界框中心点相对于对应cell左上角位置的相对偏移值，为了将边界框中心点约束在当前cell中，使用sigmoid函数处理偏移值，这样预测的偏移值在(0,1)范围内（每个cell的尺度看做1）  
总结来看，根据边界框预测的4个offsets t_x, t_y, t_w, t_h ，可以按如下公式计算出边界框实际位置和大小：  
![](https://latex.codecogs.com/gif.latex?\inline&space;\\b_x&space;=&space;\sigma&space;(t_x)&plus;c_x) 
![](https://latex.codecogs.com/gif.latex?\inline&space;\\b_y&space;=&space;\sigma&space;(t_y)&space;&plus;&space;c_y)  
![](https://latex.codecogs.com/gif.latex?\inline&space;\\b_w&space;=&space;p_we^{t_w})  
![](https://latex.codecogs.com/gif.latex?\inline&space;\\b_h&space;=&space;p_he^{t_h})  
其中 (c_x, x_y) 为cell的左上角坐标，如下图所示，在计算时每个cell的尺度为1，所以当前cell的左上角坐标为 (1,1) 。由于sigmoid函数的处理，边界框的中心位置会约束在当前cell内部，防止偏移过多。p_w 和 p_h 是先验框的宽度与长度，它们的值也是相对于特征图大小的    
![](https://i.imgur.com/tNVTJHo.jpg)   
记特征图的大小为 (W, H) （在文中是 (13, 13) )，这样我们可以将边界框相对于整张图片的位置和大小计算出来（4个值均在0和1之间）：  
![](https://latex.codecogs.com/gif.latex?\inline&space;\\b_x&space;=&space;(\sigma&space;(t_x)&plus;c_x)/W)  
![](https://latex.codecogs.com/gif.latex?\inline&space;\\&space;b_y&space;=&space;(\sigma&space;(t_y)&space;&plus;&space;c_y)/H)  
![](https://latex.codecogs.com/gif.latex?\inline&space;\\b_w&space;=&space;p_we^{t_w}/W)  
![](https://latex.codecogs.com/gif.latex?\inline&space;\\b_h&space;=&space;p_he^{t_h}/H)  
将上面的4个值分别乘以图片的宽度和长度（像素点值）就可以得到边界框的最终位置和大小了。这就是YOLOv2边界框的整个解码过程。约束了边界框的位置预测值使得模型更容易稳定训练，结合聚类分析得到先验框与这种预测方法，YOLOv2的mAP值提升了约5%。  

**细粒度特征(Fine-Grained Features)**  
YOLOv2的输入图片大小为 416x416 ，经过5次maxpooling之后得到 13x13 大小的特征图，并以此特征图采用卷积做预测。 13x13 大小的特征图对检测大物体是足够了，但是对于小物体还需要更精细的特征图（Fine-Grained Features）。因此SSD使用了多尺度的特征图来分别检测不同大小的物体，前面更精细的特征图可以用来预测小物体。YOLOv2提出了一种passthrough层来利用更精细的特征图。YOLOv2所利用的Fine-Grained Features是 26x26 大小的特征图（最后一个maxpooling层的输入），对于Darknet-19模型来说就是大小为 26x26x512 的特征图。passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上。前面的特征图维度是后面的特征图的2倍，passthrough层抽取前面层的每个 2x2 的局部区域，然后将其转化为channel维度，对于 26x26x512 的特征图，经passthrough层处理之后就变成了 13x13x2048 的新特征图（特征图大小降低4倍，而channles增加4倍，图6为一个实例），这样就可以与后面的 13x13x1024 特征图连接在一起形成 13x13x3072 大小的特征图，然后在此特征图基础上卷积做预测。在YOLO的C源码中，passthrough层称为reorg layer。在TensorFlow中，可以使用tf.extract_image_patches或者tf.space_to_depth来实现passthrough层：  

	out = tf.extract_image_patches(in, [1, stride, stride, 1], [1, stride, stride, 1], [1,1,1,1], padding="VALID")  
	// or use tf.space_to_depth  
	out = tf.space_to_depth(in, 2)    
passthrough层实例：  
![](https://i.imgur.com/3w5lJJ0.jpg)  
另外，作者在后期的实现中借鉴了ResNet网络，不是直接对高分辨特征图处理，而是增加了一个中间卷积层，先采用64个 1x1 卷积核进行卷积，然后再进行passthrough处理，这样 26x26x512 的特征图得到 13x13x256 的特征图。这算是实现上的一个小细节。使用Fine-Grained Features之后YOLOv2的性能有1%的提升。  

**多尺度训练(Multi-Scale Training)**  
由于YOLOv2模型中只有卷积层和池化层，所以YOLOv2的输入可以不限于 416x416 大小的图片。为了增强模型的鲁棒性，YOLOv2采用了多尺度输入训练策略，具体来说就是在训练过程中每间隔一定的iterations之后改变模型的输入图片大小。由于YOLOv2的下采样总步长为32，输入图片大小选择一系列为32倍数的值： {320, 352,..., 608} ，输入图片最小为 320x320 ，此时对应的特征图大小为 10xs10 （不是奇数了，确实有点尴尬），而输入图片最大为 608x608 ，对应的特征图大小为 19x19 。在训练过程，每隔10个iterations随机选择一种输入图片大小，然后只需要修改对最后检测层的处理就可以重新训练。  
![](https://i.imgur.com/vr3bcxZ.jpg)  
采用Multi-Scale Training策略，YOLOv2可以适应不同大小的图片，并且预测出很好的结果。在测试时，YOLOv2可以采用不同大小的图片作为输入，在VOC 2007数据集上的效果如下图所示。可以看到采用较小分辨率时，YOLOv2的mAP值略低，但是速度更快，而采用高分辨输入时，mAP值更高，但是速度略有下降，对于 544x544 ，mAP高达78.6%。注意，这只是测试时输入图片大小不同，而实际上用的是同一个模型（采用Multi-Scale Training训练）。   


> **总结来看，虽然YOLOv2做了很多改进，但是大部分都是借鉴其它论文的一些技巧，如Faster R-CNN的anchor boxes，YOLOv2采用anchor boxes和卷积做预测，这基本上与SSD模型（单尺度特征图的SSD）非常类似了，而且SSD也是借鉴了Faster R-CNN的RPN网络。从某种意义上来说，YOLOv2和SSD这两个one-stage模型与RPN网络本质上无异，只不过RPN不做类别的预测，只是简单地区分物体与背景。在two-stage方法中，RPN起到的作用是给出region proposals，其实就是作出粗糙的检测，所以另外增加了一个stage，即采用R-CNN网络来进一步提升检测的准确度（包括给出类别预测）。而对于one-stage方法，它们想要一步到位，直接采用“RPN”网络作出精确的预测，要因此要在网络设计上做很多的tricks。YOLOv2的一大创新是采用Multi-Scale Training策略，这样同一个模型其实就可以适应多种大小的图片了。**  


**训练**  
YOLOv2的训练主要包括三个阶段:  
第一阶段就是先在ImageNet分类数据集上预训练Darknet-19，此时模型输入为 224x224 ，共训练160个epochs。  
第二阶段将网络的输入调整为 448x448 ，继续在ImageNet数据集上finetune分类模型，训练10个epochs，此时分类模型的top-1准确度为76.5%，而top-5准确度为93.3%。  
第三个阶段就是修改Darknet-19分类模型为检测模型，并在检测数据集上继续finetune网络。网络修改包括（网路结构可视化）：移除最后一个卷积层、global avgpooling层以及softmax层，并且新增了三个 3x3x2014卷积层，同时增加了一个passthrough层，最后使用 1x1 卷积层输出预测结果，输出的channels数为： num_anchors x (5 + num_classes) ，和训练采用的数据集有关系。由于anchors数为5，对于VOC数据集输出的channels数就是125，而对于COCO数据集则为425。这里以VOC数据集为例，最终的预测矩阵为 T （shape为 (batch_size, 13, 13, 125) ），可以先将其reshape为 (batch_size, 13, 13, 5, 25) ，其中 T[:, :, :, :, 0:4] 为边界框的位置和大小 ![](https://latex.codecogs.com/gif.latex?\inline&space;(t_x,&space;t_y,&space;t_w,&space;t_h)) ， T[:, :, :, :, 4] 为边界框的置信度，而 T[:, :, :, :, 5:] 为类别预测值。  
训练三个阶段图示：  
![](https://i.imgur.com/irDEhjr.jpg)  
yolov2网络结构示意图：  
![](https://i.imgur.com/1EzVLq5.jpg)  

> **先验框匹配（样本选择）**  
> 和YOLOv1一样，对于训练图片中的ground truth，若其中心点落在某个cell内，那么该cell内的5个先验框所对应的边界框负责预测它，具体是哪个边界框预测它，需要在训练中确定，即由那个与ground truth的IOU最大的边界框预测它，而剩余的4个边界框不与该ground truth匹配。YOLOv2同样需要假定每个cell至多含有一个grounth truth，而在实际上基本不会出现多于1个的情况。  
> YOLO中一个ground truth只会与一个先验框匹配（IOU值最好的），对于那些IOU值超过一定阈值的先验框，其预测结果就忽略了。这和SSD与RPN网络的处理方式有很大不同，因为它们可以将一个ground truth分配给多个先验框  
> **训练的损失函数**  
> 与ground truth匹配的先验框计算坐标误差、置信度误差（此时target为1）以及分类误差，而其它的边界框只计算置信度误差（此时target为0）。YOLOv2和YOLOv1的损失函数一样，为均方差函数。  
> loss计算公式:  
> ![](https://i.imgur.com/onmwokg.jpg)  
> W, H 分别指的是特征图（ 13x13 ）的宽与高，而 A 指的是先验框数目（这里是5）  
> 各个 ![](https://latex.codecogs.com/gif.latex?\inline&space;\lambda) 值是各个loss部分的权重系数  
> 第一项loss是计算background的置信度误差，但是哪些预测框来预测背景呢，需要先计算各个预测框和所有ground truth的IOU值，并且取最大值Max_IOU，如果该值小于一定的阈值（YOLOv2使用的是0.6），那么这个预测框就标记为background，需要计算noobj的置信度误差。   
> 第二项是计算先验框与预测框的坐标误差，但是只在前12800个iterations间计算，我觉得这项应该是在训练前期使预测框快速学习到先验框的形状  
> 第三大项计算与某个ground truth匹配的预测框各部分loss值，包括坐标误差、置信度误差以及分类误差  


## YOLO9000
YOLO9000是在YOLOv2的基础上提出的一种可以检测超过9000个类别的模型，其主要贡献点在于提出了一种分类和检测的联合训练策略。众多周知，检测数据集的标注要比分类数据集打标签繁琐的多，所以ImageNet分类数据集比VOC等检测数据集高出几个数量级。在YOLO中，边界框的预测其实并不依赖于物体的标签，所以YOLO可以实现在分类和检测数据集上的联合训练。对于检测数据集，可以用来学习预测物体的边界框、置信度以及为物体分类，而对于分类数据集可以仅用来学习分类，但是其可以大大扩充模型所能检测的物体种类。

作者选择在COCO和ImageNet数据集上进行联合训练，但是遇到的第一问题是两者的类别并不是完全互斥的，比如"Norfolk terrier"明显属于"dog"，所以作者提出了一种层级分类方法（Hierarchical classification），主要思路是根据各个类别之间的从属关系（根据WordNet）建立一种树结构WordTree，结合COCO和ImageNet建立的WordTree如下图所示：  
![](https://i.imgur.com/pQBnJEb.jpg)  
WordTree中的根节点为"physical object"，每个节点的子节点都属于同一子类，可以对它们进行softmax处理。在给出某个类别的预测概率时，需要找到其所在的位置，遍历这个path，然后计算path上各个节点的概率之积。  
![](https://i.imgur.com/3v8UvTs.jpg)  
在训练时，如果是检测样本，按照YOLOv2的loss计算误差，而对于分类样本，只计算分类误差。在预测时，YOLOv2给出的置信度就是 Pr(physical \space object) ，同时会给出边界框位置以及一个树状概率图。在这个概率图中找到概率最高的路径，当达到某一个阈值时停止，就用当前节点表示预测的类别。

通过联合训练策略，YOLO9000可以快速检测出超过9000个类别的物体，总体mAP值为19,7%。我觉得这是作者在这篇论文作出的最大的贡献，因为YOLOv2的改进策略亮点并不是很突出，但是YOLO9000算是开创之举。  

## yolov3

