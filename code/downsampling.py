import librosa
import soundfile as sf

y, sr = librosa.load("sound.wav", sr=None)
y_8k = librosa.resample(y, sr, 8000)

sf.write("resample.wav", y_8k, 8000)
print(y.shape, y_8k.shape)
