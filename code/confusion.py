from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


multiclass = np.array([[10,  1,  0],
 [ 2,  5,  0],
 [ 0,  1,  7]])

class_names = ['d', 'f', 'k']

fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                class_names=class_names)
plt.show()
plt.savefig('confusion.png')
