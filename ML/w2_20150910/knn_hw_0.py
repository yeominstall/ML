import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier 

# 2015. 09. 17 Machine Learning Homework
# 201521068 Yeomin Nam

x, y = make_blobs(n_samples=100, centers=2, n_features=2, shuffle=True, random_state=3)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x,y)
train_x, train_y = [], []
test_x, test_y = [], []
for i in range(len(x)):
	if i < len(x)/2:
		train_x.append(x[i][0])
		train_y.append(x[i][1])
	else:
		test_x.append(x[i][0])
		test_y.append(x[i][1])


plt.scatter(train_x, train_y, c='r', marker='x')
plt.scatter(test_x, test_y, c='b', marker='o')
plt.show()

