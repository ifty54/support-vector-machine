#Gaussian Kernel Radial Basis Function (RBF)
from sklearn.svm import SVC
classifier = SVC(kernel ='rbf', random_state = 0)
# training set in x, y axis
classifier.fit(x_train, y_train)

#Sigmoid Kernel
from sklearn.svm import SVC
classifier = SVC(kernel ='sigmoid')
classifier.fit(x_train, y_train) # training set in x, y axis

#Polynomial Kernel
from sklearn.svm import SVC
classifier = SVC(kernel ='poly', degree = 4)
classifier.fit(x_train, y_train) # training set in x, y axis
