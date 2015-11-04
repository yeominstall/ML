import numpy as np
import pylab as pl
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

hs_data = np.genfromtxt("housing.data", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
hs_res = np.genfromtxt("housing.data", usecols=(13))
# normalized data
hs_scale = preprocessing.scale(hs_data)

# Linear Regression
print "==========================="
print "   1. Linear Regression    "
print "==========================="

lr = LinearRegression()
lr.fit(hs_data, hs_res)

#print "coefficient : " + str(lr.coef_)
#print "intercept : " + str(lr.intercept_)
#print "R-square : " + str(r2_score(hs_res, lr.predict(hs_data)))

lr.fit(hs_scale, hs_res)
print "coefficient : " + str(lr.coef_)
print "coef max : " + str(np.max(np.abs(lr.coef_)))
print "coef min : " + str(np.min(np.abs(lr.coef_)))
print "nomalized R-square : " + str(r2_score(hs_res, lr.predict(hs_scale)))

# Polynomial Regression
print "==========================="
print " 2. Polynomial Regression  "
print "==========================="

pl = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])
pl.fit(hs_data, hs_res)

pl.scale = preprocessing.scale
#print "pl coef mean : " + str(np.mean(pl.named_steps['linear'].coef_))
#print "coefficient : " + str(pl.named_steps['linear'].coef_)
#print "coef length : " + str(len(pl.named_steps['linear'].coef_))
print "intercept : " + str(pl.named_steps['linear'].intercept_)
print "R-square : " + str(r2_score(hs_res, pl.predict(hs_data)))

pl.fit(hs_scale, hs_res)
print "coef max : " + str(np.max(np.abs(pl.named_steps['linear'].coef_))) 
print "coef min : " + str(np.min(np.abs(pl.named_steps['linear'].coef_)))
print "nomalized R-square : " + str(r2_score(hs_res, pl.predict(hs_scale)))

# Ridge Regression(CV)
print "==========================="
print "  3. Ridge Regression-CV   "
print "==========================="

rcv = RidgeCV(alphas=[0.001, 0.01, 0.05, 0.1, 1.0])
rcv.fit(hs_data, hs_res)

print "alpha : " + str(rcv.alpha_)
#print "coefficient : " + str(rcv.coef_)
#print "intercept : " + str(rcv.intercept_)
#print "R-square : " + str(r2_score(hs_res, rcv.predict(hs_data)))

rcv.fit(hs_scale, hs_res)
print "coefficient : " + str(rcv.coef_)
print "coef max : " + str(np.max(np.abs(rcv.coef_)))
print "coef min : " + str(np.min(np.abs(rcv.coef_)))
print "nomalized R-square : " + str(r2_score(hs_res, rcv.predict(hs_scale)))
