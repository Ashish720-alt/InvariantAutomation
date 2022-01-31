from sklearn import svm
X = [[0, 0], [1, 1] , [2, 2]]
y = [0, 1, 1]
clf = svm.SVC(C = 0.01, kernel = "linear")
clf.fit(X, y)
t = clf.predict([[2., 2.]])
print(t)
i = clf.fit_status_
print(i)
sv = clf.support_
print(sv)
constant = clf.intercept_
coef = clf.coef_
print(constant)
print(coef)
print( str(coef[0][0]) + "x" + str(coef[0][1]) + "y" - str(constant) + "= 0")
