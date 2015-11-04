import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs(n_samples=100, centers=3, n_features=3, random_state=0, shuffle=False)

pca = PCA(n_components=2)
pca.fit(x)
print "principal component:\n", pca.components_
print "ratio:", pca.explained_variance_ratio_


