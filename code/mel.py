import numpy as np
import librosa


def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=2048, n_mels=128, fmin=20, fmax=8300, top_db=80):
  wav,sr = librosa.load(file_path,sr=sr)
  if wav.shape[0]<5*sr:
    wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
  else:
    wav=wav[:5*sr]
  spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
  spec_db=librosa.power_to_db(spec,top_db=top_db)
  return spec_db

p = "sound-s.wav"
mel = get_melspectrogram_db(file_path=p)

for m in mel:
    # for mm in m:
    print("M SHAMPE: ", m.shape)
print("SHAPE! ", mel.shape)
# print(get_melspectrogram_db(file_path=p))
