# coding: utf-8
# test1 for compression test
import numpy as np
import Screening
import RelationFile
import MachineLearning
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import DrawImage


# 生成deNoise, position-time，peak-frequency-time, amplitude-counts, time-counts文件
def denoise(file, filetrace, filename, voice_speed, channel=1):
    f = Screening.read_file(file)
    f0 = Screening.data_filter(f, channel=channel)
    f1 = Screening.exclude_noise(f0, voice_speed)
    # 生成一个筛选后文件序号的文件
    temp = []
    for i in f1:
        temp.append(i)
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename + '-DeNoise_Number.csv'
    np.savetxt(f, temp, fmt='%s', delimiter=',')

    f2 = Screening.renumber(f1)
    Screening.make_file(f2, filetrace, filename + '-DeNoise.csv')
    print('DeNoise File made!')


# =============================
# 读取DeNoise文件
def p_t(file, filetrace, filename, voice_speed, channel=1):
    f = Screening.read_file(file)
    RelationFile.position_time(f, filetrace, filename, voice_speed, channel=channel)
    print('Position-Time File made!')


def p_f_t(file, filetrace, filename, channel=1):
    f = Screening.read_file(file)
    RelationFile.peak_fre_time(f, filetrace, filename, channel=channel)
    print('PeakFrequency-Time File made!')


def a_c(file, filetrace, filename, channel=1):
    f = Screening.read_file(file)
    RelationFile.amp_counts(f, filetrace, filename)
    print('Amplitude-Count File made')


def t_c(file, filetrace, filename, channel=1):
    f = Screening.read_file(file)
    RelationFile.time_count(f, filetrace, filename)
    print('Time-Count File made')


# =====================
# 读取别的文件
# 绘制不同cluster中AE信号个数与时间关系
def n_t_c(file, filetrace, filename, label):
    f = Screening.read_file(file)
    RelationFile.num_time_cluster(f, filetrace, filename, label)
    print('NumberOfCluster-Time File made')


# 机器学习部分
# 已经除去噪音的twin文件和kink文件，以及两者所有event的路径
# 参数：twin文件，kink文件，待测文件，twin文件event路径，kink文件event路径，待测文件event路径, 主文件目录
def svm_machine_learn_no_model(file_twin, file_kink, file_cluster, filetrace_twin, filetrace_kink,
                               filetrace_cluster, filetrace, filename_mark, smooth=0):
    # 读twin的数据文件
    # 没有模型可以导入的话
    print('svm_machine_learn with no model is running')
    f_twin = Screening.read_file(file_twin)
    # 提取出所有有用信号的频域向量
    result1 = Screening.data_fre(f_twin, filetrace_twin, smooth)
    fre_range = result1[0]
    fre_twin = result1[1]

    # 读kink的数据文件
    f_kink = Screening.read_file(file_kink)
    result2 = Screening.data_fre(f_kink, filetrace_kink, smooth)    # [[frequency range], [[fre1],[fre2],[fre3]...[fre-n]]]
    fre_kink = result2[1]

    # 读需要分类的数据文件
    f = Screening.read_file(file_cluster)
    result3 = Screening.data_fre(f, filetrace_cluster, smooth)
    fre = result3[1]

    # svm学习
    label_fre = MachineLearning.skl_svm(fre_twin, fre_kink, fre, filetrace, filename_mark)    # [cluster_label, data_fre]
    result = MachineLearning.ave_fre(label_fre, fre_range)          # [[frequency range], [[fre1_twin],[fre2_kink]]]
    temp = [['fre-range', 'twin', 'kink']]
    array = np.array([result[0], result[1][0], result[1][1]])
    array = array.T  # 第一列fre-range，第二列twin，第三列kink
    arr = array.tolist()
    for i in arr:
        temp.append(i)
    # 保存文件
    filename = 'SVM_averange_frequency.csv'
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename
    np.savetxt(f, temp, fmt='%s', delimiter=',')
    print('SVM_averange_frequency File made')

    # 保存文件， 每个cluster的平均频谱
    filename = 'SVM_label.csv'
    f =  filetrace + '\\'+ 'File after Processing' + '\\' + filename
    np.savetxt(f, label_fre[0], fmt='%s', delimiter=',')
    print('SVM_label File made')
    return label_fre[0]


def svm_machine_learn_model(file_cluster, filetrace_cluster, filetrace, draw=0, smooth=0):
    # 有模型可以导入的话
    # 读需要分类的数据文件
    print('svm_machine_learn with model is running')
    f = Screening.read_file(file_cluster)
    # 提取出所有有用信号的频域向量
    result = Screening.data_fre(f, filetrace_cluster, smooth)
    fre_range = result[0]
    fre = result[1]
    label_fre = MachineLearning.svm_model(fre, filetrace)              # [cluster_label, data_fre]
    result = MachineLearning.ave_fre(label_fre, fre_range)  # [[frequency range], [[fre1_twin],[fre2_kink]]]
    temp = [['fre-range', 'twin', 'kink']]
    array = np.array([result[0], result[1][0], result[1][1]])
    array = array.T  # 第一列fre-range，第二列twin，第三列kink
    arr = array.tolist()
    for i in arr:
        temp.append(i)
    # 保存文件
    filename = 'SVM_averange_frequency.csv'
    # filetrace = r'C:\Users\liuhanqing\Desktop\test\AE'
    f =  filetrace + '\\' + filename
    np.savetxt(f, temp, fmt='%s', delimiter=',')
    print('SVM_averange_frequency File made')

    # 保存文件， 每个cluster的平均频谱
    filename = 'SVM_label.csv'
    # filetrace = r'C:\Users\liuhanqing\Desktop\test\AE'
    f =  filetrace + '\\' + filename
    np.savetxt(f, label_fre[0], fmt='%s', delimiter=',')
    print('SVM_label.csv File made')
    return label_fre[0]


def kmeans_machine_learn(file, filetrace, filetrace_file_cluster, smooth, filename_mark, cluster):
    print('kmeans_machine_learn is running')
    f = Screening.read_file(file)
    # 提取出所有有用信号的频域向量
    frequency = Screening.data_fre(f, filetrace_file_cluster, smooth)   # [[frequency range], [[fre1],[fre2],[fre3]...[fre-n]]]
    fre_range = frequency[0]
    fre = frequency[1]
    # kmeans 学习部分
    label_fre = MachineLearning.skl_kmeans(fre, cluster=cluster)    # [cluster_label, data_fre]
    result = MachineLearning.kmeans_ave_fre(label_fre, fre_range)   # [[frequency range], [[fre1_cluster1],[fre2_cluster2],[fre3_cluster3],[fre4_cluster4], ...]]
    n = len(set(label_fre[0]))
    title = ['fre-range']
    for i in range(n):
        title.append(str(i))
    temp = [result[0]]
    for i in result[1]:
        temp.append(i)
    temp1 = [title]
    temp = np.transpose(temp).tolist()
    for i in temp:
        temp1.append(i)
    # 保存文件
    filename = 'Kmeans_averange_frequency'
    f =  filetrace + '\\'+ 'File after Processing' + '\\' + filename + filename_mark + '.csv'
    np.savetxt(f, temp1, fmt='%s', delimiter=',')
    print('Kmeans_averange_frequency File made')

    # 保存文件， 每个cluster的平均频谱
    filename = 'Kmeans_label'
    f =  filetrace + '\\'+ 'File after Processing' + '\\' + filename + filename_mark + '.csv'
    np.savetxt(f, label_fre[0], fmt='%s', delimiter=',')
    print('Kmeans_label File made')

    # 绘制分类结果二维视图
    # result = [label, [[cluster1],[cluster2],[cluster3],...]]
    DrawImage.draw_clusterings_kmeans(label_fre, filetrace, filename_mark)
    return label_fre[0]


# 数据标准化，专门对分类后的平均频谱进行分类
def data_standard(file, filename, filetrace):
    f = Screening.read_file(file)
    f1 = f[1:]
    result = [f[0]]
    # 对数据进行[0,1]标准化
    min_max_scaler = preprocessing.MinMaxScaler()
    # 标准化训练集数据
    f2 = min_max_scaler.fit_transform(f1)
    for i in range(len(f2)):
        result.append(f2[i])
        result[i+1][0] = f1[i][0]
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename + '-Standard.csv'
    np.savetxt(f, result, fmt='%s', delimiter=',')
    print('SVM_label.csv File made')


# 检测模型的准确率
def model_test(data_type, file, filename, filetrace, filetrace_file_cluster, smooth):
    f = Screening.read_file(file)
    data_fre_range_fre = Screening.data_fre(f, filetrace_file_cluster, smooth)    # [[frequency range], [[fre1],[fre2],[fre3]...[fre-n]]]
    print(data_fre_range_fre[1])
    result = MachineLearning.svm_model(data_fre_range_fre[1], filetrace)
    file = filetrace + r'\Model Test' + '\\' + filename + '-Label.csv'
    np.savetxt(file, result[0], fmt='%s', delimiter=',')
    print('Model Test Over!')
    # 测试文件为twin：
    n_twin = 0
    n_kink = 0
    print(len(result[0]))
    for i in result[0]:
        if i == 0:
            n_twin = n_twin + 1
        if i == 1:
            n_kink = n_kink + 1
    # print('kink: ',n_kink)
    # print('twin: ', n_twin)
    if data_type == 'twin':
        accuracy = n_twin / (n_twin + n_kink)
    if data_type == 'kink':
        accuracy = n_kink / (n_twin + n_kink)
    print('Accuracy: ', accuracy)
    txtfile = ['Accuracy: ' + str(accuracy)]
    file = filetrace + r'\Model Test' + '\\' + filename + '-Accuracy.txt'
    np.savetxt(file, txtfile, fmt='%s', delimiter=',')


# 用模型对用建立模型的数据进行分类，查看建立模型时的分类效果, 0:twin, 1:kink
def model_origin_data(file_twin, file_kink, filetrace_twin, filetrace_kink, filetrace, filename_mark, smooth):
    # 读twin的数据文件
    # 没有模型可以导入的话
    print('svm_machine_learn with no model is running')
    f_twin = Screening.read_file(file_twin)
    # 提取出所有有用信号的频域向量
    result1 = Screening.data_fre(f_twin, filetrace_twin, smooth)
    fre_range = result1[0]
    fre_twin = result1[1]

    # 读kink的数据文件
    f_kink = Screening.read_file(file_kink)
    result2 = Screening.data_fre(f_kink, filetrace_kink, smooth)  # [[frequency range], [[fre1],[fre2],[fre3]...[fre-n]]]
    fre_kink = result2[1]

    # 合并数据，并保存label
    label_true = [ 0 for i in range(len(fre_twin))]
    for i in range(len(fre_kink)):
        label_true.append(1)
    fre = fre_twin
    for i in fre_kink:
        fre.append(i)
    # for i in fre:
    #     print(i)
    # 测试模型精确度
    model_path = filetrace + '\\' + r'Model\train0_model.m'
    model_svm = joblib.load(model_path)
    label_cluster = model_svm.predict(fre)
    accuracy = accuracy_score(label_true, label_cluster)
    print('accuracy: ', accuracy)
    print('y_test: ', label_true)
    print('y_predicted: ', list(label_cluster))

    # 生成两个cluster文件
    cluster_twin = []
    cluster_kink = []
    for i in range(len(label_cluster)):
        if label_cluster[i] == 0:
            cluster_twin.append(fre[i])
        if label_cluster[i] == 1:
            cluster_kink.append(fre[i])
    filename = 'SVM_cluster-twin'
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename + filename_mark + '.csv'
    np.savetxt(f, cluster_twin, fmt='%s', delimiter=',')
    filename = 'SVM_cluster-kink'
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename + filename_mark + '.csv'
    np.savetxt(f, cluster_kink, fmt='%s', delimiter=',')
    print('Twin-kink File made')

    # 生成average-frequency
    label_fre = [label_cluster, fre]  # [cluster_label, data_fre]
    result = MachineLearning.ave_fre(label_fre, fre_range)  # [[frequency range], [[fre1_twin],[fre2_kink]]]
    temp = [['fre-range', 'twin', 'kink']]
    array = np.array([result[0], result[1][0], result[1][1]])
    array = array.T  # 第一列fre-range，第二列twin，第三列kink
    arr = array.tolist()
    for i in arr:
        temp.append(i)
    # 保存文件
    filename = 'SVM_averange_frequency'
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename + filename_mark + '.csv'
    np.savetxt(f, temp, fmt='%s', delimiter=',')
    print('SVM_averange_frequency-origin_data File made')

    # 绘制2D可视化视图
    result1 = [label_cluster, fre]
    DrawImage.draw_clusterings_svm(result1, filetrace, filename_mark)


# 用kmeans的结果对svm处理的DeNosie文件进行处理（删除可能为噪声的数据）
def svm_file_kmeans_treat(file, filename, filetrace, label_kmeans, aim_cluster):
    MachineLearning.svm_origin_data_process(file, filename, filetrace, label_kmeans, aim_cluster)


# 将主要cluster进行再一次的分类（检测里面是否有异常）
def main_cluster_kmeans_treat(file, filename, filetrace, smooth=1):
    f = Screening.read_file(file)
    event_filetrace = filetrace + '\\' + filename
    fre = Screening.data_fre(f, event_filetrace, smooth)        # [[frequency range], [[fre1],[fre2],[fre3]...[fre-n]]]
    result = MachineLearning.skl_kmeans(fre[1], cluster=2)         # result = [[cluster_label], [[fre1],[fre2]]]
    result1 = MachineLearning.kmeans_ave_fre(result, fre[0])    # [[frequency range], [[fre1_cluster1],[fre2_cluster2],...]]
    n = 2
    title = ['fre-range']
    for i in range(n):
        title.append(str(i))
    temp = [result1[0]]
    for i in result1[1]:
        temp.append(i)
    temp1 = [title]
    temp = np.transpose(temp).tolist()
    for i in temp:
        temp1.append(i)
    print(temp1)
    # 保存文件
    f = filetrace + '\\'  + filename + r'-kmeans_treated-main_clustering.csv'
    np.savetxt(f, temp1, fmt='%s', delimiter=',')
    return result[0]