from sklearn.linear_model import LinearRegression

X = [[1], [2], [4], [5]]
y = [[1], [4], [2], [6]]

model = LinearRegression()
model.fit(X, y)

print model.coef_
print model.intercept_
