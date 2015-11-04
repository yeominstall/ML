from sklearn import preprocessing
import numpy as np

X = np.array([[1., -1., 2.],[2., 0., 0.],[0., 1., -1.]])
X_scaled = preprocessing.scale(X)

print X_scaled

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X)

print X_train_minmax
