VAD算法的经典设计如下：
1. 首先是一个降噪过程，如通过spectral subtraction.  
2. 然后对输入信号的一个区块提取特征。  
3. 最后使用一个分类器对这个区块进行分类，确定是或不是语音信号。通常这个分类过程是将计算的值与一个阈值进行比较。


1. 基于信噪比（snr）的vad  
2. 基于DNN的vad  
3. 基于能量Energy的vad  
4. 基于Decoder的Vad  
5. 混合DNN和Decoder的Vad   
6. webrtc_vad

传统 VAD 方法有基于能量、音高、过零率，以及多种特征组合。VAD 的核心问题是判断一段数据为 silence 或 speech

VAD的主要困难表现为现实情况下语音信号信噪比较低，不能较好地区分语音段和非语音段。

最新研究表明，基于长时信号变化率参数[6]、长时Mel频谱特征[7]的VAD方法在带噪情形下有很好的鲁棒性，因为每帧长时特征的计算都综合了之前多帧的信息 

## 基于能量谱的VAD（8K）
1. 将立体声转换为单声道  
2. 沿音频数据移动20ms的窗口
3. 计算语音能量与窗口总能量之间的比率  
	> abs(fft)**2计算能量谱；取300-3000之间的能量，计算占总能量比值，将比值和阈值做比较  
4. 如果ratio大于阈值（默认为0.6），则标签窗口为语音
5. 应用长度为0.5s的中值滤波器来平滑检测到的语音区域
6. 将语音区域表示为时间间隔

代码实现：VAD-python-energy  
https://github.com/marsbroshok/VAD-python  

## 基于DNN的vad
https://zhuanlan.zhihu.com/p/24432663  
http://jst.tsinghuajournals.com/CN/rhhtml/20180511.htm#outline_anchor_7  
https://github.com/jtkim-kaist/VAD  
https://github.com/mounalab/LSTM-RNN-VAD  
https://github.com/Cocoxili/VAD


## webrtc_vad
webrtc的vad检测原理是根据人声的频谱范围，把输入的频谱分成六个子带（80Hz~250Hz，250Hz~500Hz,500Hz~1K,1K~2K,2K~3K,3K~4K。） 分别计算这六个子带的能量。然后使用高斯模型（GMM）的概率密度函数做运算，得出一个对数似然比函数。对数似然比分为全局和局部，全局是六个子带之加权之和，而局部是指每一个子带则是局部，所以语音判决会先判断子带，子带判断没有时会判断全局，只要有一方过了，就算有语音。  
注：我国交流电标准是220V~50Hz，电源50Hz的干扰会混入麦克风采集到的数据中且物理震动也会带来影响，所以取了80Hz以上的信号  　　

webrtc的vad检测代码比较简洁，核心代码只在三个文件中：  
1. webrtc/common_audio/vad/include/webrtc_vad.h  
2. webrtc_vad.c, 该文件是用户调用的API函数，使用vad一般只需要调用该里面的函数即可    
> WebRtcVad_Create  WebRtcVad_Init 申请内存和初始化一些参数
> WebRtcVad_set_mode 设置vad要处理的采样率，一般是8000或16000
> WebRtcVad_Process 核心函数，完成检测是否有人声的核心。  
3. vad_core.c,该文件是webrtc_vad.c 文件中函数的实现代码，也是vad最深层的核心代码

　　webrtc vad有三种帧长可以用到，分别是80/10ms，160/20ms，240/30ms。其它采样率的48k，32k，24k，16k会重采样到8k来计算VAD。  
　　之所以选择上述三种帧长度，是因为语音信号是短时平稳信号，其在10ms~30ms之间可看成平稳信号，高斯马尔科夫等比较的信号处理方法基于的前提是信号是平稳的，在10ms~30ms，平稳信号处理方法是可以使用的。   
　　vad检测共四种模式，用数字0~3来区分，激进程度与数值大小正相关。
0: Normal，1：low Bitrate， 2：Aggressive；3：Very Aggressive 可以根据实际的使用，数值越大，判断越粗略，连着的静音或者响声增多

### python-webrtcvad
参考：https://github.com/wiseman/py-webrtcvad  

	!pip install webrtcvad
	import webrtcvad
	sample_rate = 8000 
	frame_duration_ms = 0.02 #20ms
	vad = webrtcvad.Vad()
	vad.set_mode(1)
	data = [random.randint(0,255) for i in range(sample_rate*frame_duration_ms)]
	frame = np.int16(data).tobytes()  # vad输入为int16转换后的bytes数据
	vad.is_speech(frame, sample_rate)  

	代码实现：py-webrtc-vad.py
  


