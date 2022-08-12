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
import time
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import random
import pprint

#
# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i + n]


with open("expirements/features/labels.txt", "rb") as fp:   # Unpickling
    labels = pickle.load(fp)

with open("expirements/features/mfcc.txt", "rb") as fp:   # Unpickling
    features = pickle.load(fp)

# features = Center()(features)
# features = Standardize()(features)

highest = 0

test = []
test_labels = []
train = []
train_labels = []
all_labels = []
# subject = "bc89ac0753b9495b87cd00213f5b4863"
subject = "6d09cea703ab419199ee74b4e693b38b"
for i, m in enumerate(features):
    current_subject = labels[i].split("_")[4]
    letter_label = labels[i].split("_")[3] # + "_" + labels[i].split("_")[3]
    if letter_label not in all_labels:
        all_labels.append(letter_label)
    if subject == current_subject:
        test.append(features[i])
        test_labels.append(letter_label)
    else:
        train.append(features[i])
        train_labels.append(letter_label)

hmms = []
class_names = []

for i, label in enumerate(all_labels):
    print("training: ", label)
    hmm_states = []
    class_names.append(label)
    for ii, t_label in enumerate(train_labels):
        if t_label == label:
            hmm_states.append(train[ii])
    hmm = GMMHMM(label=label, n_states=5, n_components=1)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    # hmm.set_random_initial()
    # hmm.set_random_transitions()

    hmm.fit(hmm_states)
    hmms.append(hmm)

clf = HMMClassifier()
clf.fit(hmms)

# test = pre(test)
predictions = clf.predict(test, original_labels = True)
print("pred")
print(predictions)
print("actual")
print(test_labels)
accuracy, confusion = clf.evaluate(test, test_labels)

print("accuracy, ", accuracy, ", confusion, \n", confusion)
if accuracy > highest:
    # highestCombination = h
    highest = accuracy
fig, ax = plot_confusion_matrix(figsize=(20, 20), conf_mat=confusion,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=False,
                                class_names=class_names)
# plt.show()
fig.savefig("confusions/confusion" + "-mfcc-"+subject + ".png")

print("highest accuracy: ", highest)
# ----
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
