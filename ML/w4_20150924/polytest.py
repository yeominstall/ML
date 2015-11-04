# polynomial regression: y = -8.39+2.95x-0.082x^2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
model.fit(X, y)
model.named_steps['linear'].coef_
print "??? : " + str(r2_score(y, model.predict(X)))
model_1 = LinearRegression()
model_1.fit(X, y)

print model_1.coef_
print model_1.intercept_
print "linear : " + str(r2_score(y, model_1.predict(X)))
