# import numpy as np
# from sequentia.classifiers import GMMHMM
#
# # Create some sample data
# X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
#
# # Create and fit a left-right HMM with random transitions and initial state distribution
# hmm = GMMHMM(label='class1', n_states=10, n_components=3, topology='left-right', covariance_type='diag')
# hmm.set_random_initial()
# hmm.set_random_transitions()
# hmm.fit(X)


import numpy as np
from sequentia.classifiers import GMMHMM, HMMClassifier

# Set of possible labels
labels = ['class{}'.format(i) for i in range(5)]
print("labels: ", labels)
# Create and fit some sample HMMs
hmms = []
for i, label in enumerate(labels):
    hmm = GMMHMM(label=label, n_states=(i + 3), n_components=2, topology='left-right')
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit([np.arange((i + j * 20) * 30).reshape(-1, 3) for j in range(1, 4)])
    hmms.append(hmm)



# Create some sample test data and labels
X = [np.random.random((10 * i, 3)) for i in range(1, 3)]
print("X, ", X[0].shape)
# y = ['class0', 'class1', 'class1']
y = ['class0', 'class2']


# Create a classifier and calculate predictions and evaluations
clf = HMMClassifier()
clf.fit(hmms)

predictions = clf.predict(X)
print("pred")
print(predictions)
accuracy, confusion = clf.evaluate(X, y)
print("accuracy, ", accuracy, ", confusion, ", confusion)

# import numpy as np
# from sequentia.classifiers import KNNClassifier
#
# # Create some sample data
# X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
# y = ['class0', 'class1', 'class1']
#
# # Create and fit the classifier
# clf = KNNClassifier(k=1, classes=list(set(y)))
# clf.fit(X, y)
#
# # Predict labels for the training data (just as an example)
# print(clf.predict(X))
# clf.predict(X)
