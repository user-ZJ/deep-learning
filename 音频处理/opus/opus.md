opus_encode() frame_size参数说明：  
OPUS编解码器支持的帧大小如下：2.5ms,5ms,10ms,20ms,40ms或60ms,80ms,100ms,120ms的音频数据。  
以16KHz采样率音频为例，每毫秒采集16个数据点，因此frame_size可以设置为一下参数：  
16 * 2.5 = 40  
16 * 5 = 80  
16 * 10 = 160  
16 * 20 = 320  
16 * 40 = 640  
16 * 60 = 960  
16 * 80 = 1280    
16 * 100 = 1600  
16 * 120 = 1920  
使用opus进行编解码，20ms帧效果最好，所以16KHz音频frame_size建议设置为320  
参考：https://stackoverflow.com/questions/46786922/how-to-confirm-opus-encode-buffer-size  


>Usage: /home/zack/opus/opus-1.3/.libs/opus_demo [-e] &lt;application&gt; &lt;sampling rate (Hz)&gt; &lt;channels (1/2)&gt; &lt;bits per second&gt;  [options] &lt;input&gt; &lt;output&gt;
       /home/zack/opus/opus-1.3/.libs/opus_demo -d &lt;sampling rate (Hz)&gt; &lt;channels (1/2)&gt; [options] &lt;input&gt; &lt;output&gt;

>application: voip | audio | restricted-lowdelay  
options:  
-e                   : only runs the encoder (output the bit-stream)  
-d                   : only runs the decoder (reads the bit-stream as input)  
-cbr                 : enable constant bitrate; default: variable bitrate  启用恒定比特率; 默认值：可变比特率   
-cvbr                : enable constrained variable bitrate; default: unconstrained 启用约束变量比特率; 默认值：不受约束  
-delayed-decision    : use look-ahead for speech/music detection (experts only); default: disabled  使用前瞻语音/音乐检测（仅限专家）; 默认值：禁用  
-bandwidth &lt;NB|MB|WB|SWB|FB&gt; : audio bandwidth (from narrowband to fullband); default: sampling rate  音频带宽（从窄带到全频带）; 默认值：采样率  
-framesize &lt;2.5|5|10|20|40|60|80|100|120&gt; : frame size in ms; default: 20  帧大小，单位为毫秒; 默认值：20  
-max_payload &lt;bytes&gt; : maximum payload size in bytes, default: 1024  最大有效负载大小（字节），默认值：1024  
-complexity &lt;comp&gt;   : complexity, 0 (lowest) ... 10 (highest); default: 10  复杂度，0（最低）... 10（最高）; 默认值：10  
-inbandfec           : enable SILK inband FEC 启用SILK内置FEC  
-forcemono           : force mono encoding, even for stereo input  强制单声道编码，即使是立体声输入  
-dtx                 : enable SILK DTX  启用SILK DTX  
-loss &lt;perc&gt;   : simulate packet loss, in percent (0-100); default: 0  模拟丢包，百分比（0-100）; 默认值：0  

>eg：  
./opus_demo -e voip 8000 1 128000 demo.pcm demo.opus  
./opus_demo -d 8000 1 128000 demo.pcm demo-1.opus  

-e audio 8000 1 16000 E:\project\eclipse\java\opus\src\demo.pcm E:\project\eclipse\java\opus\src\demo.opus
-d 8000 1 E:\project\eclipse\java\opus\src\demo.opus E:\project\eclipse\java\opus\src\demo1.pcm