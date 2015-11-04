import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier

# 2015. 09. 17 Machine Learning Homework
# 201521068 Yeomin Nam

x, y = make_blobs(n_samples=100, centers=2, n_features=2, shuffle=True, random_state=0)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x, y)

red_x, red_y = [], []
blue_x, blue_y = [], []

for i in range(len(x)):
	if y[i] == 0:
		red_x.append(x[i][0])
		red_y.append(x[i][1])
	else:
		blue_x.append(x[i][0])
		blue_y.append(x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='o')
plt.show()

