import collections
import librosa
import sys

import numpy as np
import webrtcvad
from pydub import AudioSegment
import soundfile as sf

def audio_predicate(fname):
    """
    Check if file name ends with supported audio formats
    :param fname: File name
    :return: True if file name ends with support audio formats
    """
    return fname.lower().endswith(".wav") or fname.lower().endswith(".mp3") or fname.lower().endswith(".aac")


# def remove_silence(sound, silence_threshold=-50.0, chunk_size=10):
#     """
#     Remove silence from audio clips
#     :param sound: Audio signal
#     :param silence_threshold: Threshold below which segment considered silent
#     :param chunk_size: audio segment in milli seconds to test for silence
#     :return: Audio without silent zones
#     """
#     clip = AudioSegment.empty()
#     cur_start = 0
#     trim_ms = 0
#     while trim_ms + chunk_size < len(sound):
#         if sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold:
#             cur_end = trim_ms
#             if cur_end != cur_start:
#                 clip += sound[cur_start:cur_end]
#             trim_ms += chunk_size
#             cur_start = trim_ms
#         else:
#             trim_ms += chunk_size
#     if sound[cur_start:].dBFS > silence_threshold:
#         clip += sound[cur_start:]
#     return clip

def resample(y, sr, target_sr):
    """
    Resample audio signal
    :param y: Audio signal
    :param sr: Original sampling rate
    :param target_sr: Target sampling rate
    :return: Sample audio signal
    """
    if sr != target_sr:
        y = librosa.core.resample(y=y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

def read_wav(path,dtype='int16'):
    """
    读wav文件
    :param path:文件路径
    :param dtype: 文件数据格式，支持'float64', 'float32', 'int32', 'int16'
    使用webrtcvad去静音是使用int16
    :return:
    """
    data, samplerate=sf.read(path,dtype=dtype) #data是numpy.ndarray
    audioinfo = sf.info(path)
    channels = audioinfo.channels #channels = len(data.shape)
    duration = audioinfo.duration
    name = audioinfo.name
    return data,samplerate,channels

def write_wav(path, data, samplerate):
    sf.write(path, data, samplerate)


class Frame(object):
    """
    音频帧数据格式
    """
    def __init__(self, audiodata, timestamp, duration):
        self.audiodata = audiodata  #帧数据
        self.timestamp = timestamp #起始时间
        self.duration = duration #正常（ms）


def frame_generator(frame_duration, audiodata, samplerate):
    """
    将音频按照给定长度分割为帧数据
    :param frame_duration:帧长（ms）
    :param audiodata:音频数据
    :param samplerate:采样率
    :return:list Frame
    """
    frames = []
    n = int(samplerate * frame_duration) #每帧数据
    offset = 0
    timestamp = 0.0
    duration = frame_duration
    while offset + n < len(audiodata):
        frame = Frame(audiodata[offset:offset + n], timestamp, duration)
        frames.append(frame)
        timestamp += duration
        offset += n
    return frames


def vad_collector(samplerate, frame_duration, padding_duration, vad, frames):
    """
    检测音频帧是否为静音数据，若连续padding_duration数据均为静音数据，则移除静音数据
    :param samplerate:采样率
    :param frame_duration:帧长（ms）
    :param padding_duration:平滑串口大小（ms）
    :param vad:webrtcvad静音检测对象
    :param frames:音频分帧后的数据
    :return:移除静音后的音频数据
    """
    num_padding_frames = int(padding_duration / frame_duration)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False
    voiced_data = []
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.audiodata.tobytes(), samplerate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                for f in voiced_frames:
                    voiced_data.append(f.audiodata.tolist())
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    if voiced_frames:
        for f in voiced_frames:
            voiced_data.append(f.audiodata.tolist())
    return np.array(voiced_data).astype(np.int16).flatten()


def remove_silence(audiodata,samplerate,num_padding=10):
    """
    去除音频数据中静音部分
    :param audiodata:int16类型的音频数据
    :param num_padding:平滑输出音频数据使用，
    连续chunk_size帧数据为静音则认为静音，连续chunk_size帧数据有音频则认为有音频
    :return: int16类型的音频数据array
    """
    vad = webrtcvad.Vad()
    vad.set_mode(1)
    frame_duration = 0.02 #帧长 20ms
    padding_duration = frame_duration*num_padding  # 平滑窗口大小
    frames = frame_generator(frame_duration, audiodata, samplerate)
    voiced_data = vad_collector(samplerate, frame_duration, padding_duration, vad, frames)
    return voiced_data

