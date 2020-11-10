import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

a = [[225, 0, 0, 0, 1, 0, 0, 0, 0, 0, 30],
     [2, 204, 1, 3, 0, 0, 0, 0, 0, 5, 37],
     [0, 0, 211, 0, 0, 0, 0, 8, 2, 0, 51],
     [0, 16, 0, 197, 0, 0, 0, 0, 1, 9, 30],
     [11, 1, 0, 0, 214, 0, 0, 0, 0, 0, 41],
     [0, 1, 0, 0, 3, 198, 0, 1, 0, 0, 56],
     [0, 0, 1, 0, 1, 0, 205, 10, 0, 0, 29],
     [0, 0, 8, 0, 1, 0, 8, 221, 1, 2, 21],
     [0, 0, 4, 1, 1, 0, 0, 1, 206, 1, 35],
     [0, 1, 0, 1, 0, 0, 2, 8, 0, 197, 42],
     [7, 27, 3, 16, 9, 8, 23, 17, 14, 44, 4100]]

labels = 'yes no up down left right on off stop go silence'.split()
mpl.style.use('seaborn')


def save_confusion_matrix(x: np.array):
    print(x.shape)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    sns.heatmap(x, annot=True, vmin=0.0, vmax=200.0, fmt='.2f', cmap=cmap)
    plt.savefig("confusion.png")


a = np.array(a)
# sum = a.sum()
# a = a * 100 / sum
a_df = pd.DataFrame(a, index=labels, columns=labels)
save_confusion_matrix(a_df)
