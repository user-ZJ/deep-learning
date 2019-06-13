# android VAD
本项目是将webrtc的VAD功能模块剥离，并移植到android上。

## 流程
1. 创建android native工程
2. 将[libvad](https://github.com/dpirch/libfvad)中的src和include中的文件复制到android工程的cpp目录
3. 修改CMakeLists.txt,添加cpp目录下的所有文件  
4. jni实现在vad-lib.cpp
5. java接口实现在WebRTCVad.java
6. MainActivity中读取assert中wav文件并播放，对wav文件去静音并播放