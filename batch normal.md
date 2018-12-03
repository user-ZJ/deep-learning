# batch normal

网络训练过程中参数不断改变导致后续每一层输入的分布也发生变化，而学习的过程又要使每一层适应输入的分布，因此我们不得不降低学习率、小心地初始化。  

数据归一化方法让数据具有0均值和单位方差，如果简单的这么干，会降低层的表达能力。比如，在使用sigmoid激活函数的时候，如果把数据限制到0均值单位方差，那么相当于只使用了激活函数中近似线性的部分，这显然会降低模型表达能力。  
![](https://i.imgur.com/T8XgLbs.png)  
![](https://i.imgur.com/kcN5jtT.png)  
为此，作者又为BN增加了2个参数，用来保持模型的表达能力。   
于是最后的输出为：   
![](https://i.imgur.com/QPnhQYv.png)  
上述公式中用到了均值E和方差Var，需要注意的是理想情况下E和Var应该是针对整个数据集的，但显然这是不现实的。因此，作者做了简化，用一个Batch的均值和方差作为对整个数据集均值和方差的估计。   
整个BN的算法如下：（e为一个很小的整数，保证不除零）   
![](https://i.imgur.com/Tgtv7WH.png)  

## 测试过程
实际测试网络的时候，我们依然会应用下面的式子：  
![](https://i.imgur.com/nL9N4vT.png)  
这里的均值和方差已经不是针对某一个Batch了，而是针对整个数据集而言。因此，在训练过程中除了正常的前向传播和反向求导之外，我们还要记录每一个Batch的均值和方差，以便训练完成之后按照下式计算整体的均值和方差：  
![](https://i.imgur.com/mWqJlMI.png)   
作者在文章中说应该把BN放在激活函数之前，这是因为Wx+b具有更加一致和非稀疏的分布。但是也有人做实验表明放在激活函数后面效果更好。这是实验链接，里面有很多有意思的对比实验：https://github.com/ducha-aiki/caffenet-benchmark  

BN统一了各层的方差，以适用一个统一的学习率，作用在激活函数之前，防止某个特征对网络优化起到主导作用，可以选择较大的初始学习率，加快训练速度，不需要适用dropout和L2正则化。  


# Group Normalization
## BN存在的问题
Batch Normalization，是以batch的维度做归一化，那么问题就来了，此归一化方式对batch是independent的，过小的batch size会导致其性能下降，一般来说每GPU上batch设为32最合适  
BN问题1：对于一些其他深度学习任务batch size往往只有1-2，比如目标检测，图像分割，视频分类上，输入的图像数据很大，较大的batchsize显存吃不消，对于batch size较小的训练，Batch Normalization较大；  
BN问题2：Batch Normalization是在batch这个维度上Normalization，但是这个维度并不是固定不变的，比如训练和测试时一般不一样，一般都是训练的时候在训练集上通过滑动平均预先计算好平均-mean，和方差-variance参数。在测试的时候，不再计算这些值，而是直接调用这些预计算好的来用，但是，当训练数据和测试数据分布有差别是时，训练机上预计算好的数据并不能代表测试数据，这就导致在训练，验证，测试这三个阶段存在inconsistency  

## GN原理
BN，LN（Layer Norm），IN（Instance Norm），GN区别：  
![](image/GroupNormalization.png)  
深度网络中的数据维度一般是[N, C, H, W]或者[N, H, W，C]格式，N是batch size，H/W是feature的高/宽，C是feature的channel，压缩H/W至一个维度，其三维的表示如上图，假设单个方格的长度是1，那么其表示的是[6, 6，*, * ]  
BN在batch的维度上norm，归一化维度为[N，H，W]，对batch中对应的channel归一化；  
LN避开了batch维度，归一化的维度为[C，H，W]；  
IN 归一化的维度为[H，W]；  
而GN介于LN和IN之间，其首先将channel分为许多组（group），对每一组做归一化，及先将feature的维度由[N, C, H, W]reshape为[N, G，C//G , H, W]，归一化的维度为[C//G , H, W]  
作者在论文中给出G设为32较好  

从深度学习上来讲，完全可以认为卷积提取的特征是一种非结构化的特征或者向量，每一层有很多的卷积核，这些核学习到的特征并不完全是独立的，某些特征具有相同的分布，因此可以被group

# Synchronized Batch Normalization



https://arxiv.org/pdf/1502.03167.pdf

参考：https://blog.csdn.net/u014114990/article/details/52290064