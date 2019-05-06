from abc import ABCMeta

import numpy as np
import python_speech_features


class FeatureExtractor(object):
    __metaclass__ = ABCMeta

    @property
    def dim(self):
        """Return dimension of feature generated"""
        return NotImplementedError("Return dimension of feature generated")

    def __call__(self, *args, **kwargs):
        """Given y, sr return time-frequency representation"""
        return NotImplementedError("Implement method for feature extraction")


def mel_spectrogram(y, sr, nfilt=64, log_mel=True, **kwargs):
    """
    Compute Mel Spectrogram of Audio Signal
    :param y:  Audio samples
    :param sr: Sample rate of audio
    :param nfilt: Number of mel filters
    :param log_mel: If log mel spectrogram is required
    :param kwargs: Other parameters to pass on to python_speech_features library
    :return: Mel spectrogram (numpy array of shape n_frames * n_mels)
    """
    if log_mel:
        return python_speech_features.logfbank(signal=y, samplerate=sr, nfilt=nfilt, **kwargs)
    else:
        return python_speech_features.fbank(signal=y, samplerate=sr, nfilt=nfilt, **kwargs)[0]

def mfcc(y, sr, numcep=13, delta=False, delta_delta=False, width=2, **kwargs):
    """
        Compute MFCCs of Audio Signal
        :param y: Audio signal
        :param sr: Original sample rate
        :param numcep: Number of MFCCs to compute
        :param delta: If delta of MFCCs are required
        :param delta_delta: If acceleration of MFCCs are required
        :param width: Number of samples to consider for computing delta
        :param kwargs: Other parameters to pass on python_speech_features like hop length etc.
        :return: MFCCs (numpy array of shape n_frames * n_mfccs)
        """
    mfccs = python_speech_features.mfcc(signal=y, samplerate=sr, numcep=numcep, **kwargs)
    if delta:
        d1 = python_speech_features.delta(mfccs, N=width)
        mfccs = np.hstack((mfccs, d1))
    if delta_delta:
        d2 = python_speech_features.delta(mfccs[:, mfccs.shape[1] / 2:], N=width)
        mfccs = np.hstack((mfccs, d2))
    return mfccs



class LogMelFeatureExtractor(FeatureExtractor):
    def __init__(self, mels, context, log_mel=True, stride=2):
        self.mels = mels
        self.context = context
        self.log_mel = log_mel
        self.shape = None
        self.stride = stride

    def __call__(self, *args, **kwargs):
        features = mel_spectrogram(kwargs['y'], kwargs['sr'], nfilt=self.mels, log_mel=self.log_mel)
        # If audio is too small to have required context return None
        if features.shape[0] > self.context and (features.shape[0] - self.context) / self.stride > 0:
            return np.lib.stride_tricks.as_strided(
                features, ((features.shape[0] - self.context) // self.stride + 1, self.context, self.dim()[1]),
                (self.stride * features.strides[0], features.strides[0], features.strides[1]), writeable=False
            ).astype(np.float32)
        else:
            return None

    def dim(self):
        return self.context, self.mels

    def get_shape(self, model_type='dnn'):
        if model_type == 'dnn':
            # context * frames
            shape = self.dim()[0] * self.dim()[1]
        elif model_type == 'cnn':
            # context x frames x channel (1 in this case)
            shape = self.dim()[0], self.dim()[1], 1
        else:
            # context x frames
            shape = self.dim()[0], self.dim()[1]
        return shape


class MfccFeatureExtractor(FeatureExtractor):
    def __init__(self, mfccs, context, delta=True, delta_delta=True, stride=2):
        self.mfccs = mfccs
        self.context = context
        self.delta = delta
        self.delta_delta = delta_delta
        self.shape = None
        self.stride = stride

    def __call__(self, *args, **kwargs):
        features = mfcc(kwargs['y'], kwargs['sr'], numcep=self.mfccs)
        # If audio is too small to have required context return None
        if features.shape[0] > self.context and (features.shape[0] - self.context) / self.stride > 0:
            return np.lib.stride_tricks.as_strided(
                features, ((features.shape[0] - self.context) // self.stride + 1, self.context, self.dim()[1]),
                (self.stride * features.strides[0], features.strides[0], features.strides[1]), writeable=False
            ).astype(np.float32)
        else:
            return None

    def dim(self):
        dims = self.mfccs
        if self.delta:
            dims += self.mfccs
        if self.delta_delta:
            dims += self.mfccs
        return self.context, dims

    def shape(self, model_type='dnn'):
        if model_type == 'dnn':
            # context * frames
            shape = self.dim()[0] * self.dim()[1]
        elif model_type == 'cnn':
            # context x frames x channel (1 in this case)
            shape = self.dim()[0], self.dim()[1], 1
        else:
            # context x frames
            shape = self.dim()
        return shape
