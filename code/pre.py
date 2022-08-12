from sequentia.preprocessing import *
import pprint
import numpy as np

X = [np.random.random((10 * i, 2)) for i in range(1, 2)]
X = np.array([[1,2,9],[6,12,90]])
pprint.pprint(X)
# Center the data
X = Center()(X)
pprint.pprint(X)
