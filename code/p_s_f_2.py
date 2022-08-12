from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from pathlib import Path

(rate,sig) = wav.read("filtered/female_6_alef_d_e13d0bada3f546dbaf007f0d58361adf_852d861df8e843079ada7bd1ed2cf340-mono.wav")
mfcc_feat = mfcc(sig,rate, nfft=1200)
# d_mfcc_feat = delta(mfcc_feat, 2)
# fbank_feat = logfbank(sig,rate)

print(mfcc_feat)

path = Path('filtered').glob('*.wav')
wavs = [str(wavf) for wavf in path if wavf.is_file()]
mfccs = []
labels= []
for i, p in enumerate(wavs):
    (rate,sig) = wav.read(p)
    mfcc_feat = mfcc(sig,rate, nfft=1200)
    print(mfcc_feat.shape)
