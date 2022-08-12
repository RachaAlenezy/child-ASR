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
from sequentia.classifiers import GMMHMM, HMMClassifier, KNNClassifier
from sequentia.preprocessing import *
from playsound import playsound

# y, sr = librosa.load('data/ra_k/female_5_ra_k_7af00e69e2fe4b6a9e8ee9b7520bb218_1bee244748604db0a048d9b89db30f09.wav')

path = Path('filtered/s').glob('*.wav')
wavs = [str(wavf) for wavf in path if wavf.is_file()]
mfccs = []
labels= []

for i, p in enumerate(wavs):
    if (i % 600 == 0):

        y, sr = librosa.load(p)
        print("sr", sr)
        print("y lenght:", len(y))
        # for x in y:
            # print(x)
        half = y[:len(y)//3]
        # for x in half:
        #     print(x)
        print("half lenght:", len(half))
        label = p.split('_')[2] + "_" + p.split('_')[3]
        print("label", label)
        y = half
        mfcc_features = librosa.feature.mfcc(y=y, sr=sr/4, n_mfcc=13).T
        for m in mfcc_features:
            print(m)
        mfccs.append(mfcc_features)
        labels.append(label)
