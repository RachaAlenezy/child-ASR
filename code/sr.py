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

vowels = ['ya', 'wa', 'alef']
one = ['alef', 'ha', '7a', '3en']
two = ['5a', 'ghen', '8af', 'kaf']
three = ['ta', '6a', 'dal']
four = ['wa', 'ya']
five = ['9ad', 'sen', 'shen']
six = ['jeem', 'za']
seven = ['meem', 'non', 'ba', 'lam']
eight = ['fa','4al','tha','dha','dhad']
nine = ['ra']

path = Path('filtered/s').glob('*.wav')
wavs = [str(wavf) for wavf in path if wavf.is_file()]
mfccs = []
dlabels = []
llabels = []
for i, p in enumerate(wavs):
    y, sr = librosa.load(p)
    print(sr)
