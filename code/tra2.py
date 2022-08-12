import os
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# Set seed for reproducible randomness
seed = 1
np.random.seed(seed)
rng = np.random.RandomState(seed)
data = None
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/character-trajectories/mixoutALL_shifted.mat'

try:
    path = os.path.join(os.getcwd(), 'temp.mat')
    print('Downloading dataset from {} ...'.format(url))
    response = requests.get(url)
except:
    raise
else:
    with open(path, 'wb') as file:
        print('Temporarily writing data to file ...')
        file.write(response.content)
        print('Loading data into numpy.ndarray ...')
        data = loadmat(path)
        print('Done!')
finally:
    os.remove(path)

# Load the trajectories
# NOTE: Transpose from 3xT to Tx3



X = [x.T for x in data['mixout'][0]]

print('Number of trajectories: {}'.format(len(X)))

labels = [label[0] for label in data['consts'][0][0][3][0]]
n_labels = len(labels)
print('Labels: {}'.format(str(labels)))
print('Number of labels: {}'.format(n_labels))

plt.title('Histogram of observation sequence lengths')
plt.xlabel('Number of time frames')
plt.ylabel('Count')
plt.hist([len(x) for x in X], bins=n_labels)
plt.show()

from sequentia.preprocessing import *

pre = Preprocess([
    # Trim zero-observations
    TrimZeros(),
    # Downsample with averaging and a downsample factor of n=15
    Downsample(factor=15, method='mean')
])

# Display a summary of the preprocessing transformations
pre.summary()

x = X[0]

# Downsample the example trajectory, using a downsample factor of n=10
x_pre = pre.transform(x)

# Create the plot to visualize the downsampling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(x)
ax1.set_title('Original velocity and force pen-tip trajectory sample')
ax1.legend(labels=['$x$ velocity', '$y$ velocity', 'pen-tip force'])
ax2.plot(x_pre)
ax2.set_title('Transformed velocity and force pen-tip trajectory sample')
ax2.legend(labels=['$x$ velocity', '$y$ velocity', 'pen-tip force'])
plt.show()

plt.title('Histogram of observation sequence lengths')
plt.xlabel('Number of time frames')
plt.ylabel('Count')
plt.hist([len(x) for x in TrimZeros()(X)], bins=n_labels)
plt.show()

# Transform the entire dataset
X = pre.transform(X)

y = [labels[idx - 1] for idx in data['consts'][0][0][4][0]]

# Plot a histogram of the labels for each class
# plt.title('Histogram of the dataset label counts')
# plt.xlabel('Label (character)')
# plt.ylabel('Count')
# plt.hist(y, bins=n_labels)
# plt.show()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng, shuffle=True)
# print('Training set size: {}'.format(len(X_train)))
# print('Test set size: {}'.format(len(X_test)))
#
# def show_results(acc, cm, dataset):
#     df = pd.DataFrame(cm, index=labels, columns=labels)
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(df, annot=True)
#     plt.title('Confusion matrix for {} set predictions'.format(dataset), fontsize=14)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     # Fix for matplotlib bug that cuts off top/bottom of seaborn visualizations
#     b, t = plt.ylim()
#     plt.ylim(b + 0.5, t - 0.5)
#     plt.show()
#     print('Accuracy: {:.2f}%'.format(acc * 100))
#
# from sequentia.classifiers import KNNClassifier
#
# # Create and fit a kNN classifier using the single nearest neighbor and fast C compiled DTW functions
# clf = KNNClassifier(k=1, classes=labels, use_c=True)
# clf.fit(X_train, y_train)
#
# clf.predict(X_test[0])
#
# # Predict the first 50 test examples
# predictions = clf.predict(X_test[:50], n_jobs=-1)
# print(*predictions, sep=' ', end='\n\n')
# acc, cm = clf.evaluate(X_test, y_test, n_jobs=-1)
# show_results(acc, cm, dataset='test')
