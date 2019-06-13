#include <jni.h>
#include <string>
extern "C"{
#include "fvad.h"
}

Fvad* vadPtr = NULL;

extern "C"
JNIEXPORT void JNICALL
Java_com_example_vad_WebRTCVad_nativeInitVAD(JNIEnv *env, jobject instance, jint sampleRate,
                                          jint mode) {

    vadPtr = fvad_new();
    fvad_set_mode(vadPtr,mode);
    fvad_set_sample_rate(vadPtr,sampleRate);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_vad_WebRTCVad_nativeIsSpeech(JNIEnv *env, jobject instance, jshortArray audioSample_) {
    jint len = env->GetArrayLength(audioSample_);
    jshort *audioSample = env->GetShortArrayElements(audioSample_, NULL);

    auto * audioData = new int16_t[len];
    for(int i=0;i<len;i++){
        audioData[i] = audioSample[i];
    }

    int result = fvad_process(vadPtr,audioData,(size_t)len);

    delete [] audioData;
    env->ReleaseShortArrayElements(audioSample_, audioSample, 0);
    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_vad_WebRTCVad_nativeReleaseVAD(JNIEnv *env, jobject instance) {

    fvad_free(vadPtr);
    vadPtr = NULL;

}