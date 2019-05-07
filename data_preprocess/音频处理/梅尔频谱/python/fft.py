import librosa
import wave
import numpy as np

pcm_path = 'audio/demo.pcm'
utter_path = 'audio/demo.wav'
sr = 8000
fft_kernel_size=512
frame_width=0.025
frame_shift=0.01
utter, sr = librosa.core.load(utter_path, sr)
print(utter.shape)
S = librosa.core.stft(y=utter, n_fft=fft_kernel_size,win_length=512, hop_length=512,center=False)
#print(S[0:5,0:5])
#S = librosa.core.stft(y=utter, n_fft=fft_kernel_size,win_length=int(frame_width * sr), hop_length=int(frame_shift * sr))
S = np.abs(S) ** 2
print(S.shape)
#print(S[0:5,0:5])
#print("abs"+str(S.T[:,0:5]))
mel_basis = librosa.filters.mel(sr=sr, n_fft=fft_kernel_size, n_mels=40)
print(mel_basis.shape)
S = np.log10(np.dot(mel_basis, S) + 1e-6)
print(S.shape)

# with open(pcm_path, 'rb') as pcmfile:
#     pcmdata = pcmfile.read()
# with wave.open(pcm_path + '.wav', 'wb') as wavfile:
#     wavfile.setparams((1, 2, sr, 0, 'NONE', 'NONE'))
#     wavfile.writeframes(pcmdata)