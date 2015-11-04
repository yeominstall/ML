from sklearn.datasets import load_iris
iris = load_iris()
y = iris.target
x = iris.data
n_samples = len(x)

print y
print x
print n_samples
