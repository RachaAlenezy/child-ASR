import librosa
import numpy as np
import os
import sklearn
import collections
import contextlib
import sys
import wave
import webrtcvad
from pathlib import Path
from numpy import array
from sklearn.model_selection import KFold
from sequentia.classifiers import GMMHMM, HMMClassifier
from sequentia.preprocessing import *
import pickle

path = Path('filtered/all').glob('*.wav')
wavs = [str(wavf) for wavf in path if wavf.is_file()]
labels= []

mfccs = []
mfcc_sc_sr_zcr_rms = []
mfcc_sc_zcr = []
sc_sr = []
rms_zcr = []
# index = 2

for i, p in enumerate(wavs):
    y, sr = librosa.load(p, sr=8000)
    S = np.abs(librosa.stft(y))
    label = p
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, win_length=79).T
    s_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T
    zcr = librosa.feature.zero_crossing_rate(y=y).T
    s_rollof = librosa.feature.spectral_rolloff(y=y, sr=sr).T
    rms = librosa.feature.rms(y=y).T
    print("mfcc shape:", mfcc.shape)
    print("sc shape:", s_contrast.shape)
    print("sr shape:", s_rollof.shape)
    print("zcr shape:", zcr.shape)
    print("rms shape:", rms.shape)

    mfccs.append(mfcc)
    mfcc_sc_sr_zcr_rms.append(np.hstack((mfcc, s_contrast, s_rollof, zcr, rms)))
    mfcc_sc_zcr.append(np.hstack((mfcc, s_contrast, zcr)))
    sc_sr.append(np.hstack((s_contrast, s_rollof)))
    rms_zcr.append(np.hstack((rms, zcr)))
    # mfccs.append(features)
    labels.append(label)


with open("expirements/8-features/mfcc.txt", "wb") as fp:   #Pickling
    pickle.dump(mfccs, fp)

with open("expirements/8-features/mfcc_sc_zcr.txt", "wb") as fp:   #Pickling
    pickle.dump(mfcc_sc_zcr, fp)

with open("expirements/8-features/mfcc_sc_sr_zcr_rms.txt", "wb") as fp:   #Pickling
    pickle.dump(mfcc_sc_sr_zcr_rms, fp)

with open("expirements/8-features/sc_sr.txt", "wb") as fp:   #Pickling
    pickle.dump(sc_sr, fp)

with open("expirements/8-features/rms_zcr.txt", "wb") as fp:   #Pickling
    pickle.dump(rms_zcr, fp)

with open("expirements/8-features/labels.txt", "wb") as fp:   #Pickling
    pickle.dump(labels, fp)
