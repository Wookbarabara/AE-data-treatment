# coding: utf-8
# test1 for compression test
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.externals import joblib
import DrawImage
import Screening
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ===================================================
# SVM
# 将twin（包含一部分滑移先忽视），kink（包含一部分滑移，先忽视）合并，并统一编号，twin在前（0），kink在后（1）
def merge_fre(data_fre_twin, data_fre_kink):
    print('merge_fre is running')
    fre = data_fre_twin
    result = []
    label_twin = [0 for i in range(len(data_fre_twin))]
    label_kink = [1 for i in range(len(data_fre_kink))]
    label = label_twin
    for i in label_kink:
        label.append(i)
    for i in data_fre_kink:
        fre.append(i)
    result = [label, fre]
    print('merge_fre is over')
    return result      # result = [[label], [fre]] = [[0, 0, ..., 1, 1], [[fre1, fre2, ... ,fre999], [...], ... [...]]]


# 用支持向量机（svm），第一次训练时使用，顺便创建模型
# 输入为[[fre1],[fre2],[fre3]...[fre-n]]
def skl_svm(data_twin, data_kink, data_fre, filetrace, filename_mark):
    print('slk_svm is running')
    data_fre_label = merge_fre(data_twin, data_kink)    # [[label], [fre]]
    train_label = data_fre_label[0]
    train_fre = data_fre_label[1]
    test_data = data_fre
    # 现在用KINK信号检测模型
    data_fre_label = [1 for i in range(len(data_fre))]
    accuracy = 0
    print('Model Training Start!')
    model_svm = svm.SVC(C=10, kernel='linear', gamma=1, probability=True)
    while accuracy < 0.8:
        # 划分训练集
        x_train, x_test, y_train, y_test = train_test_split(train_fre, train_label, test_size=0.4)
        # 排除所有训练数据都来自同一类型，机率很小，数据量大的时候可以不要
        if len(set(y_train)) == 1:
            continue
        # 训练模型
        model_svm.fit(x_train, y_train)
        # 查看模型精度
        y_predicted = model_svm.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        print('accuracy: ', accuracy)
        print('y_test: ', y_test)
        # print('y_predicted: ', list(y_predicted))
    save_model(model_svm, filetrace)
    cluster_label = model_svm.predict(test_data)
    accuracy1 = accuracy_score(data_fre_label, cluster_label)
    print('The Accuracy of test: ', accuracy1)
    result = [cluster_label, data_fre]

    # 绘制可视化二维图
    DrawImage.draw_clusterings_svm(result, filetrace, filename_mark)
    return result   # result = [cluster_label, data_fre]


# 保存pvm模型
def save_model(model,filetrace):
    file = filetrace + r'\Model\train31_model.m'
    joblib.dump(model, file)
    print("Done\n")
    print("Model Saving Done!\n")


# 支持向量机，导入模型
def svm_model(data_fre, filetrace):
    # 导入模型
    file = filetrace + r'\Model\train31_model.m'
    model_svm = joblib.load(file)
    cluster_label = model_svm.predict(data_fre)
    result = [cluster_label, data_fre]
    return result


# 计算出分类cluster的平均频率分布，result里存储的是频率的counts，频率储存在fre_range里
# [cluster_label, data_fre]
def ave_fre(result, fre_range):
    print('ave_fre is running')
    label = result[0]  # 0是twin，1是kink
    fre = result[1]
    # print('label: ', label, '\n', 'fre: ', fre)
    # 计算有几类cluster
    n = len(set(label))
    # print('n: ', n)
    fre_cluster = [[] for i in range(n)]
    for i in range(len(label)):
        for j in range(n):
            if label[i] == j:
                # if j==1:
                #     print(fre[i])
                fre_cluster[j].append(fre[i])
            else:
                continue
    ave_fre_cluster = [[] for i in range(n)]
    temp_fre = 0
    # print(fre_cluster)
    for num in range(n):
        for i in range(len(fre_cluster[num][0])):
            for j in range(len(fre_cluster[num])):
                temp_fre = temp_fre + float(fre_cluster[num][j][i])
            temp_fre = temp_fre / len(fre_cluster[num])
            ave_fre_cluster[num].append(temp_fre)
            temp_fre = 0
    result = [fre_range, ave_fre_cluster]
    return result  # [[frequency range], [[fre1_twin],[fre2_kink]]]


# 计算出分类cluster的平均频率分布，result里存储的是频率的counts，频率储存在fre_range里
# [cluster_label, data_fre]，检测过，这个函数ok
def kmeans_ave_fre(result, fre_range):
    print('kmeans_ave_fre is running')
    label = result[0]  # 0是cluster0，1是cluster1,...
    fre = result[1]
    # print('label: ', label, '\n', 'fre: ', fre)
    # 计算有几类cluster, 不一定是cluster个分类
    n = len(set(label))
    print('n: ', n)
    fre_cluster = [[] for i in range(n)]
    for i in range(len(label)):
        for j in range(n):
            if label[i] == j:
                fre_cluster[j].append(fre[i])
            else:
                continue
    for i in fre_cluster:
        print(fre_cluster)
    ave_fre_cluster = [[] for i in range(n)]
    temp_fre = 0
    for num in range(n):
        for i in range(len(fre_cluster[num][0])):
            for j in range(len(fre_cluster[num])):
                temp_fre = temp_fre + float(fre_cluster[num][j][i])
            temp_fre = temp_fre / len(fre_cluster[num])
            ave_fre_cluster[num].append(temp_fre)
            temp_fre = 0
    result = [fre_range, ave_fre_cluster]
    return result  # [[frequency range], [[fre1_cluster1],[fre2_cluster2],...]]


# ===================================================
# K-Means


# PCA降维
# 输入为[[fre1],[fre2],[fre3]...[fre-n]]
def skl_pca(fre, demen=2):
    result = []
    data_fre = fre
    result_pca = PCA(n_components=demen)
    result.append(result_pca.fit_transform(data_fre))
    return result


# k-means分类
# 输入为[[fre1],[fre2],[fre3]...[fre-n]]
def skl_kmeans(fre, cluster=2):
    data_fre = fre      # [[fre1],[fre2],[fre3]...[fre-n]]
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(data_fre)
    cluster_label = kmeans.labels_
    result = [cluster_label, data_fre]
    return result       # result = [cluster_label, data_fre]


# 利用k-means的分类结果，对svm使用的分类数据进行筛选
# 原因：数据中存在一些极端数据，会对整个结果产生影响
# 方法：先k-means分类，查看结果中主要类别和其他类别的平均频谱的差别，以及数量上的差别。
#      再决定是否选取一组数据使用
# 编码思路：先获得Kmeans的label，然后对DeNoise文件进行进一步操作，删除主管判断为噪声的文件
# file: Demoise 文件，filetrace: 文件存放位置, label：kmeans分类结果的标签,为文件名，需要读取文件
def svm_origin_data_process(file, filename, filetrace, label_kmeans, aim_cluster):
    # 读取label文件
    label_file = filetrace + '\\' + r'File after Processing' + '\\' + label_kmeans + '.csv'
    label = Screening.read_file(label_file)
    # 读取DeNoise文件
    data = Screening.read_file(file)
    file_treated = [data[0]]
    for i in range(len(label)):
        if float(label[i]) == float(aim_cluster):
            file_treated.append(data[i+1])
    # 保存文件
    f = filetrace + '\\' + filename + r'-kmeans_treated.csv'
    np.savetxt(f, file_treated, fmt='%s', delimiter=',')
    print('DeNoise file-kmeans_treated File made')


# 随机森林算法
def random_forest(data_twin, data_kink, filetrace, modelname):
    print('random_forest is running')
    data_fre_label = merge_fre(data_twin, data_kink)  # [[label], [fre]]
    train_label = data_fre_label[0]
    train_fre = data_fre_label[1]
    rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    accuracy = 0
    while accuracy < 0.8:
        # 划分训练集
        x_train, x_test, y_train, y_test = train_test_split(train_fre, train_label, test_size=0.4)
        # 排除所有训练数据都来自同一类型，机率很小，数据量大的时候可以不要
        if len(set(y_train)) == 1:
            continue
        # 训练模型
        rf0.fit(x_train, y_train)
        # 查看模型精度
        y_predicted = rf0.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        print('accuracy: ', accuracy)
        print('y_test: ', y_test)
        # print('y_predicted: ', list(y_predicted))
    # 保存模型
    file = filetrace + r'\RF_Model' + '\\' + modelname
    joblib.dump(rf0, file)
    print("Done\n")
    print("Model Saving Done!\n")
    # 返回模型
    return rf0


# RF模型
def RF_svm_model(data_fre, filetrace, modelname):
    # 导入模型
    file = filetrace + r'\RF_Model' + '\\' + modelname
    model_svm = joblib.load(file)
    cluster_label = model_svm.predict(data_fre)
    result = [cluster_label, data_fre]
    return result