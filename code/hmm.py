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

with open("expirements/features/mfccs.txt", "rb") as fp:   # Unpickling
    features = pickle.load(fp)

# features = Center()(features)
# features = Standardize()(features)

# one = ['alef', 'ha', '7a', '5a', '3en', '8af' , 'ghen', 'kaf']
#
# two = ['9ad', 'sen', 'shen', 'tha', 'fa', 'jeem', 'za']
# three = [ 'dal', '6a', '4al','dha','dhad', 'ta']
#
# four = ['ba', 'meem', 'non']
# five = ['lam', 'ra', 'wa', 'ya']
highestCombination = []
one = ['5a','ta','tha','4al','dha','3en','7a','8af','fa','dal','6a','non','ra','lam'] #0.66 accuracy

# one = ['non','meem','sen','kaf','lam','3en','9ad','4al','ya','ba','dal','shen','5a','8af'] #0.57

# one =  ['jeem','ya','kaf','3en','dha','ghen','wa','ba','dhad','5a','non','shen','6a','tha'] #0.57

# one = ['alef', 'ta', 'tha', 'jeem', '7a', '5a', 'dal', '4al', 'za', 'sen', 'shen', '9ad', '3en', 'ghen'] #0.42

# one = ['ba', 'ta', 'jeem', '5a', '8af', 'kaf', 'lam', 'meem', 'non', '6a', '4al', 'za', 'shen', '9ad'] #0.42

# four = ['meem', 'non', 'ba', 'lam', 'ra']


# first iteration #

# one = ['sen', 'jeem', 'alef']
# two = ['ba', 'dha']
# three = ['tha', '6a', '4al']
# four = ['ta', 'dhad', 'dal', 'ra', 'ya', 'ha']
# five = ['7a', 'ghen', 'meem']
# six = ['9ad', 'kaf', 'non']
highest = 0

# one = ['dhad', 'dal', 'za', 'ba', 'fa', 'dha', 'alef', '6a', '5a']
# two = []
# all = ['alef', 'ba', 'ta', 'tha', 'jeem', '7a', '5a', 'dal', '4al', 'ra', 'za', 'sen', 'shen', '9ad', 'dhad', '6a', 'dha', '3en', 'ghen', 'fa', '8af', 'kaf', 'lam', 'meem', 'non', 'ha', 'wa', 'ya']


for i, label in enumerate(labels):
    labels[i] = labels[i].split("_")[0]
    # if (labels[i] in h[0]):
        # labels[i] = "0"
    if (labels[i] in one):
        labels[i] = "0"
    else:
        labels[i] = "1"
    # elif (labels[i] in h[1]):
        # labels[i] = "1"
    # elif (labels[i] in h[2]):
    #     labels[i] = "2"
    # elif (labels[i] in h[3]):
    #     labels[i] = "3"
    # elif (labels[i] in h[4]):
    #     labels[i] = "4"
    # elif (labels[i] in h[5]):
    #     labels[i] = "5"
    # elif (labels[i] in h[6]):
    #     labels[i] = "6"
    # elif (labels[i] in h[7]):
    #     labels[i] = "7"
    # elif (labels[i] in h[8]):
    #     labels[i] = "8"
    # elif (labels[i] in h[9]):
    #     labels[i] = "9"
    # elif (labels[i] in h[10]):
    #     labels[i] = "10"
    # elif (labels[i] in h[11]):
    #     labels[i] = "11"
    # elif (labels[i] in h[12]):
    #     labels[i] = "12"
    # elif (labels[i] in h[13]):
    #     labels[i] = "13"
    # elif (labels[i] in h[13]):
    #     labels[i] = "13"
    # elif (labels[i] in five):
    #     labels[i] = "7a, ghen, meem"
    # elif (labels[i] in six):
    #     labels[i] = "9ad, kaf, non"
    # else:
    #     labels[i] = labels[i]

test = []
test_labels = []
train = []
train_labels = []
all_labels = []

for i, m in enumerate(features):
    if labels[i] not in all_labels:
        all_labels.append(labels[i])
    if i % 90 == 0:
        test.append(features[i])
        test_labels.append(labels[i])
    else:
        train.append(features[i])
        train_labels.append(labels[i])

hmms = []
class_names = []

for i, label in enumerate(all_labels):
    hmm_states = []
    class_names.append(label)
    for ii, t_label in enumerate(train_labels):
        if t_label == label:
            hmm_states.append(train[ii])
    hmm = GMMHMM(label=label, n_states=len(hmm_states)//3, n_components=1, topology='linear', covariance_type='diag')
    # hmm.set_uniform_initial()
    # hmm.set_uniform_transitions()
    hmm.set_random_initial()
    hmm.set_random_transitions()

    hmm.fit(hmm_states)
    hmms.append(hmm)

clf = HMMClassifier()
clf.fit(hmms)

# test = pre(test)
predictions = clf.predict(test, prior='uniform', return_scores=True, original_labels = False)
print("pred")
print(predictions)
print("actual")
print(test_labels)
accuracy, confusion = clf.evaluate(test, test_labels)

print("accuracy, ", accuracy, ", confusion, \n", confusion)
if accuracy > highest:
    # highestCombination = h
    highest = accuracy
fig, ax = plot_confusion_matrix(conf_mat=confusion,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=False,
                                class_names=class_names)
# plt.show()
fig.savefig("confusions/confusion" + str(nn) + ".png")

print("highest accuracy: ", highest)

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
