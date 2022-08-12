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



# y, sr = librosa.load('data/ra_k/female_5_ra_k_7af00e69e2fe4b6a9e8ee9b7520bb218_1bee244748604db0a048d9b89db30f09.wav')

path = Path('commands').glob('**/*.wav')
wavs = [str(wavf) for wavf in path if wavf.is_file()]
mfccs = []
labels= []
for i, p in enumerate(wavs):
    if (i % 2 == 0):
        y, sr = librosa.load(p)
        label = p.split('/')[1]
        mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
        mfccs.append(mfcc_features)
        labels.append(label)

test = []
test_labels = []
train = []
train_labels = []
all_labels = []
for i, m in enumerate(mfccs):
    if labels[i] not in all_labels:
        all_labels.append(labels[i])
    if i % 20 == 0:
        test.append(mfccs[i])
        test_labels.append(labels[i])
    else:
        train.append(mfccs[i])
        train_labels.append(labels[i])
hmms = []
for i, label in enumerate(all_labels):
    print("new label.")
    hmm_states = []
    for ii, t_label in enumerate(train_labels):
        if t_label == label:
            hmm_states.append(train[ii])
            print("appending.")
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
