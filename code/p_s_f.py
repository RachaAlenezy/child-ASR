from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
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
from sklearn.model_selection import KFold
from sequentia.classifiers import GMMHMM, HMMClassifier
from sequentia.preprocessing import *
import scipy.io.wavfile as wav
#
# (rate,sig) = wav.read("sound.wav")
# mfcc_feat = mfcc(sig,rate, nfft=1200)
# d_mfcc_feat = delta(mfcc_feat, 2)
# fbank_feat = logfbank(sig,rate, nfft=1200)
#



path = Path('filtered').glob('*.wav')
wavs = [str(wavf) for wavf in path if wavf.is_file()]
mfccs = []
labels= []
for i, p in enumerate(wavs):
    print(".... ", i , "....")

    (y,sr) = wav.read(p)
    mfcc_feat = mfcc(y,sr, nfft=1200)
    # d_mfcc_feat = delta(mfcc_feat, 2)
    # fbank_feat = logfbank(y,sr, nfft=1200)

    # y, sr = librosa.load(p)
    label = p.split('_')[2] + "_" + p.split('_')[3]
    # label = p.split('_')[3]
    # mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
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
    mfccs.append(mfcc_feat)
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
#      Standardize(),
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
    if i % 100 == 0:
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
    hmm = GMMHMM(label=label, n_states=len(hmm_states), n_components=1, topology='left-right')
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
