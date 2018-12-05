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

## 7. SSD

## 8. yolo-v1
YOLO创造性的将物体检测任务直接当作回归问题（regression problem）来处理，将候选区和检测两个阶段合二为一。只需一眼就能知道每张图像中有哪些物体以及物体的位置  
事实上，YOLO也并没有真正的去掉候选区，而是直接将输入图片划分成7x7=49个网格，每个网格预测两个边界框，一共预测49x2=98个边界框。可以近似理解为在输入图片上粗略的选取98个候选区，这98个候选区覆盖了图片的整个区域，进而用回归预测这98个候选框对应的边界框  











