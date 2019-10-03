import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
digits=datasets.load_digits()
clf=svm.SVC(gamma=0.001,C=100)
print(digits.data)
print(digits.target)
x,y=digits.data[:-10],digits.target[:-10]
plt.plot(x,y)
-plt.show()
clf.fit(x,y)
print('Prediction:',clf.predict(digits.data[-2].reshape(1,-1)))
plt.imshow(digits.images[-2],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()
