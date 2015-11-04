import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score

# 2015. 09. 17 Machine Learning Homework
# 201521068 Yeomin Nam

x, y = make_blobs(n_samples=100, centers=2, n_features=2, shuffle=True, random_state=0)
cov = np.cov(np.array(x[:len(x)/2]).T)
neigh = KNeighborsClassifier(n_neighbors=3, metric='mahalanobis', V=cov)
train_x, train_y = [], []
test_x, test_y = [], []
test_y_pred = []
train_x_0, train_x_1 = [], []
test_x_0, test_x_1 = [], []
err_0, err_1 = [], []

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
	if neigh.predict(test_x[j]) != test_y[j]:
		err_0.append(test_x[j][0])
		err_1.append(test_x[j][1])

print "accuracy : " + str(accuracy_score(test_y, test_y_pred))
print err_0
print err_1
plt.scatter(train_x_0, train_x_1, c='r', marker='x')
plt.scatter(test_x_0, test_x_1, c='b', marker='x')
plt.scatter(err_0, err_1, c='b', marker='o')
plt.show()

