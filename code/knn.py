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
import pickle

path = Path('all').glob('*.wav')

wavs = [str(wavf) for wavf in path if wavf.is_file()]
mfccs = []
labels= []

for p in wavs:
    y, sr = librosa.load(p)
    label = p.split('_')[2] # + "_" + p.split('_')[3]
    print("label", label)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24).T
    mfccs.append(mfcc_features)
    labels.append(label)

pre = Preprocess([
     TrimZeros(),
     Center(),
     Standardize(),
     Filter(window_size=5, method='median'),
     Downsample(factor=2, method='decimate'),
     MinMaxScale(scale=(0.0, 1.0), independent=True)
 ])



test = []
test_labels = []
train = []
train_labels = []
all_labels = []
for i, m in enumerate(mfccs):
    if labels[i] not in all_labels:
        all_labels.append(labels[i])
    if i % 55 == 0:
        test.append(mfccs[i])
        test_labels.append(labels[i])
    else:
        train.append(mfccs[i])
        train_labels.append(labels[i])


# hmms = []
# for i, label in enumerate(all_labels):
#     hmm_states = []
#     for ii, t_label in enumerate(train_labels):
#         if t_label == label:
#             hmm_states.append(train[ii])
#     hmm = GMMHMM(label=label, n_states=len(hmm_states), n_components=1, topology='left-right')
#     hmm.set_random_initial()
#     hmm.set_random_transitions()
#

    # with open('before'+str(i)+'.txt', 'w') as f:
    #     for h in hmm_states:
    #         f.write("%s\n" % h)
    #
    # hmm_states = pre(hmm_states)
    # # with open('after'+str(i)+'.txt', 'w') as f:
    # #     for h in hmm_states:
    # #         f.write("%s\n" % h)
    # hmm.fit(hmm_states)
    # hmms.append(hmm)

clf = KNNClassifier(k=5, classes=all_labels)
clf.fit(train, train_labels)

# predictions = clf.predict(test)
# print("pred")
# print(predictions)
# print("actual")
# print(test_labels)

accuracy, confusion = clf.evaluate(test, test_labels)
print("accuracy, ", accuracy, ", confusion, \n", confusion)

# hmms = []
# for i, label in enumerate(train_labels):
#     hmm = GMMHMM(label=label, n_states=len(train), n_components=16, topology='left-right')
#     hmm.set_random_initial()
#     hmm.set_random_transitions()
#
#     hmm.fit(train[i])
#     hmms.append(hmm)

# with open('your_file.txt', 'w') as f:
#     for item in labels:
#         f.write("%s\n" % item)



# print("Mfcc features lenght:")
# print(len(mfcc_features))
# sample_rate, wave =  wavfile.read(full_audio_path)
# mfcc_features = mfcc(wave, nwin=int(sample_rate * 0.03), fs=sample_rate, nceps=12)[0]
# mfcc_features = mfcc(wave, nwin=int(sample_rate * 0.03), fs=sample_rate, nceps=12)[0]
# mfcc_features.shape
# scaler = sklearn.preprocessing.StandardScaler()
# mfcc_features_scaled = scaler.fit_transform(mfcc_features)

# print(mfcc_features.shape)
# print(mfcc_features)
