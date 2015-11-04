import numpy as np

x = np.array([[0.9, 1.], [2.4, 2.6], [1.2, 1.7], [0.5, 0.7], [0.3, 0.7], [1.8, 1.4], [0.5, 0.6], [0.3, 0.6], [2.5, 2.6], [1.3, 1.1]])

print x
m1 =  np.mean(x.T[0])
m2 =  np.mean(x.T[1])
print m1, m2
x.T[0] = x.T[0] - m1
x.T[1] = x.T[1] - m2
print x

#print np.det(x)
#print np.inv(x)
x_cov = np.cov(x.T)
#print np.eig(x)
w, v = np.linalg.eig(x_cov)

print x_cov
print w
print v.T
