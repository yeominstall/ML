# y = 1.9655+0.976x
# model.intercept_ = 1.9655, model.coef_ = 0.976

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

model = LinearRegression()
model.fit(X, y)

print "coef: " + str(model.coef_)
print "intercept: " + str(model.intercept_)
print "error: " + str(np.mean((model.predict(X)-y)**2))

plt.figure()
plt.plot(X, y, 'k.')
plt.plot(X, model.predict(X), color='blue', linewidth=3)
plt.show()
