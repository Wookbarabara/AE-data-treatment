import numpy as np
from sklearn import svm

# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
#
# clf = svm.SVC(probability=True,random_state=1,gamma='auto')
# clf.fit(X,y)
# scores = clf.predict_proba([[1,0]])
# # 输出属于每个聚类的概率
# print (scores)
fre = [[1,1,1], [[1,1,1],[2,25,2], [3,3,3]]]
temp_fre=0
ave_fre = []
for i in range(len(fre[1][0])):
    for j in range(len(fre[1])):
        temp_fre = temp_fre + float(fre[1][j][i])
    temp_fre = temp_fre / len(fre[1])
    ave_fre.append(temp_fre)
    temp_fre = 0
print(ave_fre)