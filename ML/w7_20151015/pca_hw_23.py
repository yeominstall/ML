import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs(n_samples=100, centers=3, n_features=3, random_state=0, shuffle=False)

print "original data:\n", x

means = [(np.mean(x.T[i])) for i in range(0, 3)]
print "means:", means

for i in range(0, 3):
	x.T[i] = x.T[i] - means[i]

x_ = np.cov(x.T)
x_eigval, x_eigvec = LA.eig(x_)
x_eigvec = x_eigvec.T
print "cov matrix:\n", x_
print "eigenvalues:\n", x_eigval
print "eigenvectors:\n", x_eigvec
maxeidx = []
maxeval = []
maxevec = []
for i in range(0, 2):
	idx = np.argmax(x_eigval)
	maxeval.append(x_eigval[idx])
	maxeidx.append(idx)
	maxevec.append(list(x_eigvec[idx]))
	x_eigval[idx] = 0
print "2-top eigenvalues:\n", maxeval, "(", maxeidx, ")"
U = np.vstack(maxevec)
print "2-top eigenvectors(U):", U.shape, "\n", U
print "s` = s - mean(s):\n", x

reduced = np.dot(U, x.T).T
print "reduced data:\n", reduced

