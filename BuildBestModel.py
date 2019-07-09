# coding: utf-8
# test1 for compression test
import QuickFunction
import Screening
import MachineLearning
import numpy as np

def model_test(modelname, data_type, file, filename, filetrace, filetrace_file_cluster, smooth):
    f = Screening.read_file(file)
    data_fre_range_fre = Screening.data_fre(f, filetrace_file_cluster,
                                            smooth)  # [[frequency range], [[fre1],[fre2],[fre3]...[fre-n]]]
    # print(data_fre_range_fre[1])
    result = MachineLearning.RF_svm_model(data_fre_range_fre[1], filetrace, modelname)
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


path = r'C:\Users\liuhanqing\Desktop\research\Academic conference\data\Frequency Smoothing\normalize\event-200ms\AE'
kink_list = ['KINK-train-LPSOMg-0deg-Test1-EVT43dB-DeNoise', 'KINK-test-LPSOMg-0deg-Test2-EVT38dB-DeNoise', 'KINK-LPSO-0deg-Test3-EVT38dB-DeNoise']
twin_list = ['TWIN-training-Mg-Test1-EVT40dB-DeNoise', 'TWIN-test-Mg-Test2-EVT40dB-DeNoise', 'TWIN-test-Mg-Test3-EVT40dB-DeNoise']
# kink_list = ['KINK-train-LPSOMg-0deg-Test1-EVT43dB-DeNoise', 'KINK-test-LPSOMg-0deg-Test2-EVT38dB-DeNoise']
# twin_list = ['TWIN-training-Mg-Test1-EVT40dB-DeNoise']
model_num = []
for i in range(1,4):
    for j in range(1,4):
        # kink+twin
        model_num.append(str(i)+str(j))
smooth = 1
n = 0
accuracy = []
model_accuracy = [['model number', 'accuracy']]
for kink in kink_list:
    for twin in twin_list:
        modelname = 'RF_model-' + model_num[n] + '.m'
        print(modelname)

        # 建立模型
        file_kink = path + '\\' + kink + '.csv'
        file_twin = path + '\\' + twin + '.csv'
        filetrace_kink = path + '\\' + kink
        filetrace_twin = path + '\\' + twin
        model = QuickFunction.random_forest_learn(file_twin, file_kink, filetrace_twin, filetrace_kink, modelname, path, smooth)
        result = [['Model'+model_num[n], 'Accuracy']]
        # 模型测试
        for kink_test in kink_list:
            print(kink_test)
            data_type = 'kink'
            file = path + '\\' + kink_test + '.csv'
            filetrace_file_cluster = path + '\\' + kink_test
            if kink_test != kink:
                temp_accuracy = model_test(modelname,data_type, file, kink_test, path, filetrace_file_cluster, smooth)
                result.append([kink_test, temp_accuracy])
        for twin_test in twin_list:
            print(twin_test)
            data_type = 'twin'
            file = path + '\\' + twin_test + '.csv'
            filetrace_file_cluster = path + '\\' + twin_test
            if twin_test != twin:
                temp_accuracy = model_test(modelname, data_type, file, twin_test, path, filetrace_file_cluster, smooth)
                result.append([twin_test, temp_accuracy])
        # 生成一个file
        f = path + '\\' + 'RF_Accuracy'+ '\\' + '_model-'+ model_num[n] + r'.csv'
        np.savetxt(f, result, fmt='%s', delimiter=',')
        n = n + 1
