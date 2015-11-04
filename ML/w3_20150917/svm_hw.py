import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Teaching Assistant Evaluation
# 	Number of Instance : 151
#	Attribute Information
#	 1. Whether of not the TA is a native English speaker (binary)
#		1=English speaker, 2=non-English speaker
#	 2. Course instructor (categorical, 25 categories)
#	 3. Course (categorical, 26 categories)
#	 4. Summer or regular semester (binary) 1=Summer, 2=Regular
#	 5. Class size (numerical)
#	 6. Class attribute (categorical) 1=Low, 2=Medium, 3=High 

# configuration
train_num = 120			# number of training data

tae_data = np.genfromtxt("tae.txt", delimiter=",", usecols=(0,1,2,3,4))
tae_res = np.genfromtxt("tae.txt", delimiter=",", usecols=(5))

print "==========================="
print " Data before normalization "
print "==========================="

#KNN
cov = np.cov(np.array(tae_data[:len(tae_data)/2]).T)
neigh = KNeighborsClassifier(n_neighbors=3, metric='mahalanobis', V=cov)
neigh.fit(tae_data[:train_num], tae_res[:train_num])

knn_acr = accuracy_score(tae_res[train_num:], neigh.predict(tae_data[train_num:]))
print "KNN accuracy:\t\t " + str(knn_acr)

#linear SVM
lvm = svm.SVC(kernel="linear", C=1)
lvm.fit(tae_data[:train_num], tae_res[:train_num])

lvm_acr = accuracy_score(tae_res[train_num:], lvm.predict(tae_data[train_num:]))
print "Linear SVM accuracy:\t " + str(lvm_acr)

#Cross Validation Test
cvt = svm.SVC(kernel="linear", C=1)
scores = cross_validation.cross_val_score(cvt.fit(tae_data[:train_num], tae_res[:train_num]), tae_data, tae_res, cv=5)
print scores
print "CV Test accuracy:\t " + str(scores.mean()) + "\n"

print "==========================="
print "      Normalized data      "
print "==========================="

n_tae_data = preprocessing.scale(tae_data)

#KNN(standardization)
n_cov = np.cov(np.array(n_tae_data[:len(tae_data)/2]).T)
neigh = KNeighborsClassifier(n_neighbors=3, metric='mahalanobis', V=n_cov)
neigh.fit(n_tae_data[:train_num], tae_res[:train_num])

knn_acr = accuracy_score(tae_res[train_num:], neigh.predict(n_tae_data[train_num:]))
print "KNN accuracy:\t\t " + str(knn_acr)

#linear SVM(standardization)
lvm = svm.SVC(kernel="linear", C=1)
lvm.fit(n_tae_data[:train_num], tae_res[:train_num])

lvm_acr = accuracy_score(tae_res[train_num:], lvm.predict(n_tae_data[train_num:]))
print "Linear SVM accuracy:\t " + str(lvm_acr)

#Cross Validation Test(standardization)
cvt = svm.SVC(kernel="linear", C=1)
scores = cross_validation.cross_val_score(cvt.fit(n_tae_data[:train_num], tae_res[:train_num]), n_tae_data, tae_res, cv=5)
print "CV Test accuracy:\t " + str(scores.mean()) + "\n"
