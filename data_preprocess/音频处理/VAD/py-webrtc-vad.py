# encoding = utf-8
import collections
import random
import sys
import webrtcvad
import numpy as np
import soundfile as sf


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, audiodata, timestamp, duration):
        self.audiodata = audiodata
        self.timestamp = timestamp
        self.duration = duration



def frame_generator(frame_duration_ms, data, sample_rate):
    """
    将音频划分为帧
    :param frame_duration_ms:帧长（ms）
    :param data:音频数据
    :param sample_rate:采样率
    :return:帧数据列表
    """
    frames = []
    n = int(sample_rate*frame_duration_ms/1000.0)
    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0
    while offset + n < len(data):
        frame = Frame(data[offset:offset + n], timestamp, duration)
        frames.append(frame)
        timestamp += duration
        offset += n
    return frames

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """
    删除静音数据
    :param sample_rate:采样率
    :param frame_duration_ms:帧长（ms）
    :param padding_duration_ms:平滑使用，连续padding_duration_ms时间内有语音则认为有语音
    :param vad:webrtc vad对象
    :param frames:帧列表
    :return:
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False
    voiced_data = []
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.audiodata.tobytes(), sample_rate)
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


def main(args):
    data, sample_rate = sf.read("voice.wav",dtype='int16')
    vad = webrtcvad.Vad()
    vad.set_mode(1)
    frame_duration_ms = 20  # ms 帧长为20ms
    padding_duration_ms = 200  # ms 200ms内一直有语音则判断为有语音
    frames = frame_generator(frame_duration_ms, data, sample_rate)
    voiced_data = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)
    sf.write("audio_cut.wav", voiced_data, sample_rate)

if __name__ == '__main__':
    main(sys.argv[1:])