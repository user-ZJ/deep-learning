package com.example.vad;

import android.util.Log;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;

public class WebRTCVad {
    //静音检测模式，数值越大，判断越粗略
    public final static int[] VALID_VAD_MODES = new int[] {0, 1, 2, 3};
    //有效采样率，8k、16k、32k、48k
    public final static int[] VALID_SAMPLE_RATES = new int[] {8000, 16000, 32000, 48000};
    //有效帧长 10ms、20ms、30ms
    public final static int [] VALID_FRAME_DURATION = new int [] {10,20,30};
    private static final String TAG = "WebRTCVad";
    private int sampleRate;
    private int mode;
    private int frameDuration;
    private int numPadding;


    public static boolean isValidVadMode(int mode){
        for(int i:VALID_VAD_MODES){
            if(i==mode) return true;
        }
        return false;
    }

    public static boolean isValidSampleRate(int sampleRate){
        for(int i:VALID_SAMPLE_RATES){
            if(i==sampleRate) return true;
        }
        return false;
    }

    public static boolean isValidDuration(int duration){
        for(int i:VALID_FRAME_DURATION){
            if(i==duration) return true;
        }
        return false;
    }


    public WebRTCVad(int mSampleRate, int mMode,int mFrameDuration){
        if(!isValidSampleRate(mSampleRate)){
            Log.e(TAG,"invalid samplerate "+mSampleRate);
            return;
        }
        if(!isValidVadMode(mMode)){
            Log.e(TAG,"invalid mode "+mMode);
            return;
        }

        if(!isValidDuration(mFrameDuration)){
            Log.e(TAG,"invalid mode "+mFrameDuration);
            return;
        }

        sampleRate = mSampleRate;
        mode = mMode;
        frameDuration = mFrameDuration;
        numPadding=10;
        nativeInitVAD(sampleRate,mode);
    }

    public short [] remove_silence(short [] audioData){
        ArrayList<Frame> frames = frame_generator(audioData,frameDuration,sampleRate);
        short [] voice_data = vad_collector(frames,sampleRate,frameDuration,numPadding);
        return voice_data;
    }

    private short[] vad_collector(ArrayList<Frame> frames, int sampleRate, int frameDuration, int numPadding) {
        int n = sampleRate * frameDuration/1000; //每帧数据
        int num_padding_frames = numPadding;
        int threshold = (int) (0.9 * num_padding_frames);
        LimitQueue<Frame> ring_buffer = new LimitQueue<Frame>(num_padding_frames);
        ArrayList<Frame> temp_frames = new ArrayList<Frame>();
        ArrayList<Frame> voiced_frames = new ArrayList<Frame>();
        boolean triggered = false;
        String masks = "";
        for(int i=0;i<frames.size();i++) {
            Frame frame = frames.get(i);
            int is_speech = nativeIsSpeech(frame.audioData);
            frame.setIsSpeech(is_speech);
            masks += is_speech;
            if(!triggered) {
                ring_buffer.offer(frame);
                int num_voiced = getnumvoiced(ring_buffer);
                if(num_voiced > threshold) {
                    triggered = true;
                    masks = masks + "+("+ring_buffer.getFirst().timeStamp+")";
                    for(int j=0;j<ring_buffer.size();j++) {
                        temp_frames.add(ring_buffer.get(j));
                    }
                    ring_buffer.clear();
                }
            }else {
                temp_frames.add(frame);
                ring_buffer.offer(frame);
                int num_unvoiced = getnumunvoiced(ring_buffer);
                if(num_unvoiced > threshold) {
                    masks = masks + "-("+(frame.timeStamp+frame.duration)+")";
                    triggered = false;
                    for(Frame f:temp_frames) {
                        voiced_frames.add(f);
                    }
                    ring_buffer.clear();
                    temp_frames.clear();
                }
            }
        }
        if(!temp_frames.isEmpty()) {
            for(Frame f:temp_frames) {
                voiced_frames.add(f);
            }
        }
        Log.d(TAG,"masks "+ masks);
        short [] voiced_data = new short[n*voiced_frames.size()];
        int pos = 0;
        for(Frame f:voiced_frames) {
            System.arraycopy(f.audioData, 0, voiced_data, pos, f.audioData.length);
            pos += f.audioData.length;
        }
        return voiced_data;
    }

    private int getnumunvoiced(LimitQueue<Frame> ring_buffer) {
        int count = 0;
        for(int i=0;i<ring_buffer.size();i++) {
            if(ring_buffer.get(i).getIsSpeech()!=1) {
                count++;
            }
        }
        return count;
    }

    private int getnumvoiced(LimitQueue<Frame> ring_buffer) {
        int count = 0;
        for(int i=0;i<ring_buffer.size();i++) {
            if(ring_buffer.get(i).getIsSpeech()==1) {
                count++;
            }
        }
        return count;
    }

    private ArrayList<Frame> frame_generator(short[] audioData, int frameDuration, int sampleRate) {
        ArrayList<Frame> frames = new ArrayList<Frame>();
        int n = sampleRate * frameDuration/1000; //每帧数据
        int offset = 0;
        double timestamp = 0.0;
        int duration = frameDuration;
        while(offset + n < audioData.length) {
            short [] frameData = new short[n];
            System.arraycopy(audioData,offset,frameData,0,n);
            Frame frame = new Frame(frameData, timestamp, duration);
            frames.add(frame);
            timestamp += duration/1000.0;
            offset += n;
        }

        return frames;
    }

    public void releaseVAD(){
        nativeReleaseVAD();

    }


    private native void nativeInitVAD(int sampleRate,int mode);
    private native int nativeIsSpeech(short[] audioSample);
    private native void nativeReleaseVAD();


}
