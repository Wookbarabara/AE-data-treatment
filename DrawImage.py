# coding: utf-8
# test1 for compression test
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import MachineLearning
from sklearn import preprocessing


# 绘制分类结果二维视图
# 暂时 svm只针对2个cluster
# result = [label, [[cluster1],[cluster2],[cluster3],...]]
# 0为twin，1为kink
def draw_clusterings_svm(result, filetrace, filename_mark):
    label = result[0]
    print(len(label))
    X = np.array(result[1])
    # PCA 降维
    X = MachineLearning.skl_pca(X, demen=2)
    x_standard = X[0]
    # 对数据进行[0,1]标准化
    min_max_scaler = preprocessing.MinMaxScaler()
    # 标准化训练集数据
    x_standard = min_max_scaler.fit_transform(x_standard)
    cluster_twin = []
    cluster_kink = []
    for i in range(len(label)):
        if label[i] == 0:
            cluster_twin.append(x_standard[i])
        if label[i] == 1:
            cluster_kink.append(x_standard[i])
    # 保存文件
    filename1 = 'SVM_cluster-twin-Normalization'
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename1 + filename_mark + '.csv'
    np.savetxt(f, cluster_twin, fmt='%s', delimiter=',')
    filename2 = 'SVM_cluster-kink-Normalization'
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename2 + filename_mark + '.csv'
    np.savetxt(f, cluster_kink, fmt='%s', delimiter=',')
    print('SVM 2D Image File made!')


# kmean 绘制分类结果二维视图
# result = [label, [[cluster1],[cluster2],[cluster3],...]]
def draw_clusterings_kmeans(result, filetrace):
    label = result[0]
    n_cluster = len(set(label))
    X = np.array(result[1])
    # PCA 降维
    X = MachineLearning.skl_pca(X, demen=2)
    x_standard = X[0]
    # 对数据进行[0,1]标准化
    min_max_scaler = preprocessing.MinMaxScaler()
    # 标准化训练集数据
    x_standard = min_max_scaler.fit_transform(x_standard)

    cluster = [[] for i in range(n_cluster)]
    for i in range(len(label)):
        for j in range(n_cluster):
            if label[i] == j:
                cluster[j].append(x_standard[i])
    # 保存文件
    filenumber = 0
    for i in cluster:
        filenumber = filenumber + 1
        filename = 'KMeans_cluster-cluter' + str(filenumber) + r'-Normalization.csv'
        f = filetrace + '\\'+ 'File after Processing' + '\\' + filename
        np.savetxt(f, i, fmt='%s', delimiter=',')
    print('KMeans 2D Image File made!')


# 数据光滑化，将频谱值改为相邻5个值的平均值
def fre_smoothing(frequency):
    result = []
    length = len(frequency)
    for i in range(len(frequency)):
        if i == 0:
            temp = float(frequency[i]) + float(frequency[i+1])+ float(frequency[i+2])
            temp = temp / 3
            result.append(temp)
            continue
        if i == 1:
            temp = float(frequency[i-1]) + float(frequency[i]) + float(frequency[i+1]) + float(frequency[i+2])
            temp = temp / 4
            result.append(temp)
            continue
        if i == length-2:
            temp = float(frequency[i-2]) + float(frequency[i-1]) + float(frequency[i])+ float(frequency[i+1])
            temp = temp / 4
            result.append(temp)
            continue
        if i == length-1:
            temp = float(frequency[i-2]) + float(frequency[i-1])+ float(frequency[i])
            temp = temp / 3
            result.append(temp)
            continue
        temp = float(frequency[i-2]) + float(frequency[i-1]) + float(frequency[i])+ float(frequency[i+1]) + float(frequency[i+2])
        temp = temp / 5
        result.append(temp)
    return result