# coding: utf-8
# test1 for compression test
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.externals import joblib
import DrawImage


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
    data_fre_label = merge_fre(data_twin, data_kink)
    train_label = data_fre_label[0]
    train_fre = data_fre_label[1]
    test_data = data_fre
    accuracy = 0
    print('Model Training Start!')
    while accuracy < 0.8:
        # 划分训练集
        x_train, x_test, y_train, y_test = train_test_split(train_fre, train_label, test_size=0.4)
        # 排除所有训练数据都来自同一类型，机率很小，数据量大的时候可以不要
        if len(set(y_train)) == 1:
            continue
        # 训练模型
        model_svm = svm.SVC(C=1, kernel='linear', gamma=1)
        model_svm.fit(x_train, y_train)
        # 查看模型精度
        y_predicted = model_svm.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        print('accuracy: ', accuracy)
        print('y_test: ', y_test)
        print('y_predicted: ', list(y_predicted))
    save_model(model_svm, filetrace)
    cluster_label = model_svm.predict(test_data)
    result = [cluster_label, data_fre]

    # 绘制可视化二维图
    DrawImage.draw_clusterings_svm(result, filetrace, filename_mark)
    return result   # result = [cluster_label, data_fre]


# 保存pvm模型
def save_model(model,filetrace):
    file = filetrace + r'\Model\train0_model.m'
    joblib.dump(model, file)
    print("Done\n")
    print("Model Saving Done!\n")


# 支持向量机，导入模型
def svm_model(data_fre, filetrace):
    # 导入模型
    file = filetrace + r'\Model\train0_model.m'
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
# [cluster_label, data_fre]
def kmeans_ave_fre(result, fre_range, cluster):
    print('kmeans_ave_fre is running')
    label = result[0]  # 0是cluster0，1是cluster1,...
    fre = result[1]
    print('label: ', label, '\n', 'fre: ', fre)
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
