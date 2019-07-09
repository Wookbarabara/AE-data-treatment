# coding: utf-8
# test1 for compression test
import QuickFunction
import Screening
import MachineLearning
import numpy as np
from sklearn.externals import joblib
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def svm_machine_learn(data_twin, data_kink, filetrace, modelname):
    print('svm is running')
    data_fre_label = MachineLearning.merge_fre(data_twin, data_kink)  # [[label], [fre]]
    train_label = data_fre_label[0]
    train_fre = data_fre_label[1]
    model_svm = svm.SVC(C=10, kernel='linear', gamma=1, probability=True)
    accuracy = 0
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
    # 保存模型
    file = filetrace + r'\SVM_Model' + '\\' + modelname
    joblib.dump(model_svm, file)
    print("Done\n")
    print("Model Saving Done!\n")
    # 返回模型
    return model_svm


def svm_model_local(data_fre, filetrace, modelname):
    # 导入模型
    file = filetrace + r'\SVM_Model' + '\\' + modelname
    model_svm = joblib.load(file)
    cluster_label = model_svm.predict(data_fre)
    result = [cluster_label, data_fre]
    return result


def svm_model_test(modelname, data_type, file, filename, filetrace, filetrace_file_cluster, smooth):
    f = Screening.read_file(file)
    data_fre_range_fre = Screening.data_fre(f, filetrace_file_cluster,
                                            smooth)  # [[frequency range], [[fre1],[fre2],[fre3]...[fre-n]]]
    # print(data_fre_range_fre[1])
    result = svm_model_local(data_fre_range_fre[1], filetrace, modelname)
    print('RF_Model Test Over!')
    # 测试文件为twin还是kink：
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
    return accuracy


def svm_learning(file_twin, file_kink, filetrace_twin, filetrace_kink, modelname, path, smooth):
    # 读twin的数据文件
    print('svm_machine_learn with no model is running')
    f_twin = Screening.read_file(file_twin)
    # 提取出所有有用信号的频域向量
    result1 = Screening.data_fre(f_twin, filetrace_twin, smooth)
    fre_range = result1[0]
    fre_twin = result1[1]

    # 读kink的数据文件
    f_kink = Screening.read_file(file_kink)
    result2 = Screening.data_fre(f_kink, filetrace_kink,
                                 smooth)  # [[frequency range], [[fre1],[fre2],[fre3]...[fre-n]]]
    fre_kink = result2[1]

    # svm学习
    model = svm_machine_learn(fre_twin, fre_kink, path, modelname)  # [cluster_label, data_fre]
    return model


path = r'C:\Users\liuhanqing\Desktop\research\Academic conference\data\Frequency Smoothing\normalize\event-200ms\AE'
kink_list = ['KINK-train-LPSOMg-0deg-Test1-EVT43dB-DeNoise', 'KINK-test-LPSOMg-0deg-Test2-EVT38dB-DeNoise',
             'KINK-LPSO-0deg-Test3-EVT38dB-DeNoise']
twin_list = ['TWIN-training-Mg-Test1-EVT40dB-DeNoise', 'TWIN-test-Mg-Test2-EVT40dB-DeNoise',
             'TWIN-test-Mg-Test3-EVT40dB-DeNoise']
# kink_list = ['KINK-train-LPSOMg-0deg-Test1-EVT43dB-DeNoise', 'KINK-test-LPSOMg-0deg-Test2-EVT38dB-DeNoise']
# twin_list = ['TWIN-training-Mg-Test1-EVT40dB-DeNoise']
model_num = []
for i in range(1, 4):
    for j in range(1, 4):
        # kink+twin
        model_num.append(str(i) + str(j))
smooth = 1
n = 0
accuracy = []
model_accuracy = [['model number', 'accuracy']]
for kink in kink_list:
    for twin in twin_list:
        modelname = 'SVM_model-' + model_num[n] + '.m'
        print(modelname)

        # 建立模型
        file_kink = path + '\\' + kink + '.csv'
        file_twin = path + '\\' + twin + '.csv'
        filetrace_kink = path + '\\' + kink
        filetrace_twin = path + '\\' + twin
        model = svm_learning(file_twin, file_kink, filetrace_twin, filetrace_kink, modelname, path, smooth)
        result = [['SVM_Model' + model_num[n], 'Accuracy']]
        # 模型测试
        for kink_test in kink_list:
            print(kink_test)
            data_type = 'kink'
            file = path + '\\' + kink_test + '.csv'
            filetrace_file_cluster = path + '\\' + kink_test
            if kink_test != kink:
                temp_accuracy = svm_model_test(modelname, data_type, file, kink_test, path, filetrace_file_cluster, smooth)
                result.append([kink_test, temp_accuracy])
        for twin_test in twin_list:
            print(twin_test)
            data_type = 'twin'
            file = path + '\\' + twin_test + '.csv'
            filetrace_file_cluster = path + '\\' + twin_test
            if twin_test != twin:
                temp_accuracy = svm_model_test(modelname, data_type, file, twin_test, path, filetrace_file_cluster, smooth)
                result.append([twin_test, temp_accuracy])
        # 生成一个file
        f = path + '\\' + 'SVM_Accuracy' + '\\' + '_model-' + model_num[n] + r'.csv'
        np.savetxt(f, result, fmt='%s', delimiter=',')
        n = n + 1
