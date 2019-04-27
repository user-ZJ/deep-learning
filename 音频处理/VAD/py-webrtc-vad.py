# encoding = utf-8
import collections
import random
import sys
import webrtcvad
from deepspeaker_pa.utils import audio_utils
import numpy as np


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, audiodata, timestamp, duration):
        self.audiodata = audiodata
        self.timestamp = timestamp
        self.duration = duration



def frame_generator(frame_duration_ms, data, sample_rate):
    frames = []
    #n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
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
                #yield b''.join([f.bytes for f in voiced_frames])
                for f in voiced_frames:
                    voiced_data.append(f.audiodata.tolist())
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        #yield b''.join([f.bytes for f in voiced_frames])
        for f in voiced_frames:
            voiced_data.append(f.audiodata.tolist())
    return np.array(voiced_data).astype(np.int16).flatten()


def main(args):
    data, sample_rate, channels = audio_utils.read_wav("voice.wav")
    vad = webrtcvad.Vad()
    vad.set_mode(1)
    frame_duration_ms = 20  # ms 帧长为20ms
    padding_duration_ms = 200  # ms 200ms内一直有语音则判断为有语音
    # sample_window = int(sample_rate*frame_duration)
    # frame = data[0:sample_window]
    # print('Contains speech: %s' % (vad.is_speech(frame, rate)))
    frames = frame_generator(frame_duration_ms, data, sample_rate)
    voiced_data = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)
    audio_utils.write_wav("audio_cut.wav", voiced_data, sample_rate)

if __name__ == '__main__':
    main(sys.argv[1:])