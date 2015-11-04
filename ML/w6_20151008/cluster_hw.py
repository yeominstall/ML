import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

seed_data = np.genfromtxt("seeds_dataset.txt", usecols=(0, 1, 2, 3, 4, 5, 6))
seed_res = np.genfromtxt("seeds_dataset.txt", usecols=(7))
datacount = len(seed_res)
meandist = []

# elbow method
for k in range(1, 10):
	elbow = KMeans(n_clusters=k)
	elbow.fit(seed_data)
	meandist.append(sum(np.min(cdist(seed_data, elbow.cluster_centers_, 'euclidean'), axis=1)) / seed_data.shape[0])

plt_info = plt.figure()
plt_info.add_subplot(111).set_xlabel("number of clusters")
plt_info.add_subplot(111).set_ylabel("Sum of squared error")
plt.plot(range(1, 10), meandist, 'rx-')
#plt.show()

# k-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(seed_data)
km_clcentr = kmeans.cluster_centers_


pre = km_clcentr[0][5]
prei = 0
for cc in range(0, 3):
	for i in range(cc+1, 3):
		if cmp(km_clcentr[i][5], km_clcentr[cc][5]) > 0:
			km_clcentr[i], km_clcentr[cc] = km_clcentr[cc], km_clcentr[i] 
#sorted(km_clcentr, key=km_clcentr[:][5])

kmeans2 = KMeans(n_clusters=3, init=km_clcentr[:])
kmeans2.fit(seed_data)
for i in range(datacount):
	kmeans2.labels_[i] += 1

print kmeans2.labels_[:]
# meanshift clustering
bw = estimate_bandwidth(seed_data, quantile=0.2)
#print "MeanShift bandwidth:", bw
ms = MeanShift(bandwidth=bw, bin_seeding=True)
ms.fit(seed_data)
#print ms.labels_[:]

#print seed_res
pred = ms.predict(seed_data)
for i in range(datacount):
	if pred[i] == 0:
		pred[i] = 3

print ms.labels_[:]

print "seedresult-Kmeans accuracy:", accuracy_score(seed_res, kmeans2.labels_)
print "seedresult-Meanshift accuracy:", accuracy_score(seed_res, pred)
print "Kmeans-Meanshift accuracy:", accuracy_score(kmeans2.labels_, pred)

#compdict = []
#for i in range(datacount):
#	compdict.append([seed_res[i], pred[i]])

