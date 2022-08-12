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
h = 0
for i, p in enumerate(wavs):
    print("path: ", p)
    y, sr = librosa.load(p, sr=48000)
    print("sr", sr)
    if (len(y) > h):
        h = len(y)
    print("y lenght:", len(y))
    print("y shape: ", y.shape)

print("highest: ", h)
