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

# y, sr = librosa.load('data/ra_k/female_5_ra_k_7af00e69e2fe4b6a9e8ee9b7520bb218_1bee244748604db0a048d9b89db30f09.wav')



# path = Path('filtered/s').glob('*.wav')
path = Path('filtered/all/mono/mo').glob('*.wav')

wavs = [str(wavf) for wavf in path if wavf.is_file()]
mfccs = []
labels= []
for i, p in enumerate(wavs):
        if (i % 1 == 0):
            print(".... ", i , "....")
            y, sr = librosa.load(p)
            S = np.abs(librosa.stft(y))
            hop_length = 64
            # label = p.split('_')[2] + "_" + p.split('_')[3]
            label = p.split('_')[3]
            mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, win_length=1028).T
            # s_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T
            # zcr = librosa.feature.zero_crossing_rate(y=y).T
            # s_rollof = librosa.feature.spectral_rolloff(y=y, sr=sr).T
            # rms = librosa.feature.rms(y=y).T

            # p0 = librosa.feature.poly_features(S=S, order=0).T
            # p1 = librosa.feature.poly_features(S=S, order=1).T
            # p2 = librosa.feature.poly_features(S=S, order=2).T
            # oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            # f_tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length).T
            # tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length).T

            # print(tempogram.shape)
            # print("mfcc, ", mfcc_features.shape, "s_contrast, ", s_contrast.shape, ", zcr, ", zcr.shape, " f_tme, ", f_tempogram.shape)
            # features = np.hstack((mfcc_features, s_contrast, zcr, s_rollof))
            mfccs.append(mfcc_features)
            # mfccs.append(f_tempogram)
            # mfccs.append(features)
            labels.append(label)
# pre = Preprocess([
#      TrimZeros(),
#      Center(),
#      Standardize(),
#      Filter(window_size=5, method='median'),
#      Downsample(factor=2, method='decimate'),
#      MinMaxScale(scale=(0.0, 1.0), independent=True)
#  ])

# with open('ppp.txt', 'w') as f:
#     f.write("BEFORE\n" )
#     for m in mfccs:
#         f.write("%s\n" % m)
#
#     pre = Preprocess([
#          Center(),
#          Standardize(),
#          Equalize()
#      ])
#     f.write("AFTER\n" )
#     mfccs = pre(mfccs)
#     for m in mfccs:
#         f.write("%s\n" % m)
#
# pre = Preprocess([
#      Center(),
#      Equalize()
#  ])
# mfccs = pre(mfccs)

#
test = []
test_labels = []
train = []
train_labels = []
all_labels = []
for i, m in enumerate(mfccs):
    if labels[i] not in all_labels:
        all_labels.append(labels[i])
    if i % 200 == 0:
        test.append(mfccs[i])
        test_labels.append(labels[i])
    else:
        train.append(mfccs[i])
        train_labels.append(labels[i])
hmms = []

for i, label in enumerate(all_labels):
    print(".... ", i , "....", label)
    hmm_states = []
    for ii, t_label in enumerate(train_labels):
        if t_label == label:
            hmm_states.append(train[ii])
    # hmm = GMMHMM(label=label, n_states=len(hmm_states), n_components=1, topology='left-right') #original
    hmm = GMMHMM(label=label, n_states=len(hmm_states)//20, n_components=1, topology='left-right')

    hmm.set_random_initial()
    hmm.set_random_transitions()

    hmm.fit(hmm_states)
    hmms.append(hmm)

clf = HMMClassifier()
clf.fit(hmms)

predictions = clf.predict(test)
print("pred")
print(predictions)
print("actual")
print(test_labels)
accuracy, confusion = clf.evaluate(test, test_labels)
print("accuracy, ", accuracy, ", confusion, \n", confusion)
