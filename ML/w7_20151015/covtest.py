import numpy as np

x = np.array([[2., 1.], [2., 4.], [4., 1.], [4., 3.]])

print x
m1 =  np.mean(x.T[0])
m2 =  np.mean(x.T[1])
print m1, m2
#x.T[0] = x.T[0] - m1
#x.T[1] = x.T[1] - m2
#print x

#print np.det(x)
#print np.inv(x)
x_cov = np.cov(np.array(x).T)
#print np.eig(x)
w, v = np.linalg.eig(x_cov)

print x_cov
print w
print v.T
