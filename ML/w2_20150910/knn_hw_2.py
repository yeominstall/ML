import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score

# 2015. 09. 17 Machine Learning Homework
# 201521068 Yeomin Nam

x, y = make_blobs(n_samples=100, centers=2, n_features=2, shuffle=True, random_state=0)
neigh = KNeighborsClassifier(n_neighbors=3)
train_x, train_y = [], []
test_x, test_y = [], []
test_y_pred = []
train_x_0, train_x_1 = [], []
test_x_0, test_x_1 = [], []

for i in range(len(x)):
	if i < len(x)/2:
		train_x.append(x[i])
		train_x_0.append(x[i][0])
		train_x_1.append(x[i][1])
		train_y.append(y[i])
	else:
		test_x.append(x[i])
		test_x_0.append(x[i][0])
		test_x_1.append(x[i][1])
		test_y.append(y[i])

neigh.fit(train_x, train_y)

for j in range(len(test_x)):
	test_y_pred.append(neigh.predict(test_x[j]))

print test_y_pred

plt.scatter(train_x_0, train_x_1, c='r', marker='x')
plt.scatter(test_x_0, test_x_1, c='b', marker='x')
plt.show()

