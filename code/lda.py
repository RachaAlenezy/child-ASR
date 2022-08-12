import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import contextlib
from pathlib import Path
import librosa
import pickle

path = Path('filtered/all').glob('*.wav')
wavs = [str(wavf) for wavf in path if wavf.is_file()]
l = []
X = []
all = []
heighest = 0;
def pad(l, content, width):
    l.extend([content] * (width - len(l)))
    return l
features = []
labels = []
for i, p in enumerate(wavs):
    y, sr = librosa.load(p)
    labels.append(p)
    if (len(y) > heighest):
        heighest = len(y)
    print("Y Lenght:", len(y))
        # mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
        # print("mfcc, ", mfcc.shape)
        # X.append(mfcc)
        # l.append(label)
    # y.resize(64827)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    # s_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T
    # zcr = librosa.feature.zero_crossing_rate(y=y).T
    # s_rollof = librosa.feature.spectral_rolloff(y=y, sr=sr).T
    # rms = librosa.feature.rms(y=y).T
    # feature = np.hstack((mfcc, s_contrast, zcr, rms, s_rollof))
    features.append(mfcc)

with open("expirements/all_13_mfcc.txt", "wb") as fp:   #Pickling
    pickle.dump(features, fp)

# with open("expirements/all_labels.txt", "wb") as fp:   #Pickling
    # pickle.dump(labels, fp)

# print(l)
#
# # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# # y = np.array([1, 1, 1, 2, 2, 2])
# X = np.array(X, dtype=[np.float64, np.float64, np.float64])
# l = np.array(l)
# clf = LinearDiscriminantAnalysis()
# clf.fit(X, l)
# print(clf.predict([X[0]]))
