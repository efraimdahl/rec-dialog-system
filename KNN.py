from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

data = make_blobs(n_samples = 1000, centers = 100, random_state = 8)
print(data)

