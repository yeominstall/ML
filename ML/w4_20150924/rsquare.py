import numpy as np
import matplotlib.pyplot as okt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

model = LinearRegression()
model.fit(X, y)

print model.coef_
print model.intercept_
print r2_score(y, model.predict(X))
