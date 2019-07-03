import numpy as np
from sklearn import svm

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf = svm.SVC(probability=True,random_state=1,gamma='auto')
clf.fit(X,y)
scores = clf.predict_proba([[1,0]])
# 输出属于每个聚类的概率
print (scores)