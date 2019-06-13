package com.example.vad;

class Frame {

    short [] audioData; //帧数据
    double timeStamp; //起始时间
    int duration; //帧长（ms）
    int isSpeech; //是否有声音

    Frame(short [] mAudioData,double mTimeStamp,int mDuration){
        audioData = mAudioData;
        timeStamp = mTimeStamp;
        duration = mDuration;
        isSpeech = 0;
    }

    void setIsSpeech(int is_speech) {
        isSpeech = is_speech;
    }

    int getIsSpeech() {
        return isSpeech;
    }

}
