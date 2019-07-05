# coding: utf-8
# test1 for compression test
import QuickFunction


def main(method=1, model=0, smooth=0):
    # 文件目录
    filetrace = r'C:\Users\liuhanqing\Desktop\research\Academic conference\data\Frequency Smoothing\normalize\event-200ms\AE'
    # 处理用的文件名，若没有DeNoise，则先用method=3生成DeNoise文件
    filename = r'KINK-test-LPSOMg-0deg-Test2-EVT38dB-DeNoise'
    # 对生成文件进行标记
    filename_mark = ''

    # file_cluster是所有函数用于处理的文件
    file_cluster = filetrace + '\\' + filename + '.csv'
    # 放event文件的文件夹路径
    filetrace_file_cluster = filetrace + '\\' + filename

    # SVM 中训练模型文件
    # twin和kink文件名
    filename_twin = r'TWIN-training-Mg-Test1-EVT40dB-DeNoise'
    filename_kink = r'KINK-train-LPSOMg-0deg-Test1-EVT43dB-DeNoise'
    file_twin = filetrace + '\\' + filename_twin + '.csv'
    file_kink = filetrace + '\\' + filename_kink + '.csv'
    # twin和kink中event文件路径
    filetrace_twin = filetrace + '\\' + filename_twin
    filetrace_kink = filetrace + '\\' + filename_kink

    # 用于制图所用表格的文件
    file = filetrace + '\\' + filename + '.csv'

    # 调用svm机器学习，生成：
    # 1.训练模型中不同cluster的平均频谱
    # 2.分类结果的label，对应DeNoise文件的顺序
    if method == 0:
        QuickFunction.svm_machine_learn_no_model(file_twin, file_kink, file_cluster, filetrace_twin, filetrace_kink,
                                                 filetrace_file_cluster, filetrace, filename_mark, smooth)
        print('svm machine learning 0 over!')

    # 直接导入模型
    if method == 1:
        QuickFunction.svm_machine_learn_model(file_cluster, filetrace_file_cluster, filetrace, smooth)
        print('svm machine learning 1 over!')

    # 调用K-Means机器学习
    if method == 2:
        QuickFunction.kmeans_machine_learn(file_cluster, filetrace, filetrace_file_cluster, smooth, cluster=3)
        print('kmeans machine learning over!')

    # ==================================
    # 生成各种制图文件

    # DeNoise
    # voice_speed: 纯Mg = 4.8， 纯Zn = 4.2，  纯LPSO = 5.5
    voice_speed = 5.5
    if method == 3:
        filename = r'KINK-LPSO-0deg-Test3-EVT38dB'
        file = filetrace + '\\' + filename + '.csv'
        QuickFunction.denoise(file, filetrace, filename, voice_speed, channel=1)
        print('denoise over!')

    # position-time
    if method == 4:
        QuickFunction.p_t(file, filetrace, filename, voice_speed, channel=1)
        print('p_t over!')

    # peak-frequency-time
    if method == 5:
        QuickFunction.p_f_t(file, filetrace, filename, channel=1)
        print('p_f_t over!')

    # amplitude-counts
    if method == 6:
        QuickFunction.a_c(file, filetrace, filename, channel=1)
        print('a_c over!')

    # time - counts
    if method == 7:
        QuickFunction.t_c(file, filetrace, filename, channel=1)
        print('t_c over!')

    # SVM number-time of different cluster
    if method == 8:
        # 获得分类结果
        # 没模型的情况
        if model == 0:
            label = QuickFunction.svm_machine_learn_no_model(file_twin, file_kink, file_cluster, filetrace_twin,
                                                             filetrace_kink, filetrace_file_cluster, filetrace, filename_mark, smooth)
            print('svm machine learning model-0 over!')
            QuickFunction.n_t_c(file_cluster, filetrace, filename, label)
        # 有模型的情况
        if model == 1:
            label = QuickFunction.svm_machine_learn_model(file_cluster, filetrace_file_cluster, filetrace, smooth)
            print('svm machine learning model-1 over!')
            QuickFunction.n_t_c(file_cluster, filetrace, filename, label)

    # Kmeans number-time of different cluster
    if method == 9:
        # 获得分类结果
        label = QuickFunction.kmeans_machine_learn(file_cluster, filetrace, filetrace_file_cluster, smooth, filename_mark, cluster=3)
        print('kmeans machine learning over!')
        # 生成文件
        QuickFunction.n_t_c(file_cluster, filetrace, filename, label)

    # 将分类结果降至二维
    # ===========还没写完===========
    # svm 学习
    if method == 10:
        # 获得分类结果
        # 没模型的情况
        if model == 0:
            label = QuickFunction.svm_machine_learn_no_model(file_twin, file_kink, file_cluster, filetrace_twin,
                                                             filetrace_kink, filetrace_file_cluster, filetrace, filename_mark, smooth)
            print('svm machine learning model-0 over!')
        # 有模型的情况
        if model == 1:
            label = QuickFunction.svm_machine_learn_model(file_cluster, filetrace_file_cluster, filetrace, smooth)
            print('svm machine learning model-1 over!')

    # kmeans 学习
    if method == 11:
        label = QuickFunction.kmeans_machine_learn(file_cluster, filetrace, filetrace_file_cluster, smooth, cluster=3)
        print('kmeans machine learning over!')
    # ==============================

    # 数据标准化，专门对分类后的平均频谱进行分类
    if method == 12:
        filename = r'SVM_averange_frequency-origin_file'
        file = filetrace + '\\' + 'File after Processing' + '\\' + filename + '.csv'
        QuickFunction.data_standard(file, filename, filetrace)

    # 测试svm模型
    if method == 13:
        filename = r'KINK-LPSO-0deg-Test3-EVT38dB-DeNoise'
        filetrace_file_cluster = filetrace + '\\' + filename
        file = filetrace + '\\' + filename + '.csv'
        data_type = 'kink'
        QuickFunction.model_test(data_type, file, filename, filetrace, filetrace_file_cluster, smooth)

    # 用模型对用建立模型的数据进行分类，查看建立模型时的分类效果, 0:twin, 1:kink
    if method == 14:
        filename_mark = '-origin_file'
        QuickFunction.model_origin_data(file_twin, file_kink, filetrace_twin, filetrace_kink, filetrace, filename_mark, smooth)
        print('model_origin_data over!')

    # 用kmeans的结果对svm处理的DeNosie文件进行处理（删除可能为噪声的数据）
    if method == 15:
        label_kmeans = r'Kmeans_label-LPSO-0deg-Test1'
        filename = r'KINK-train-LPSOMg-0deg-Test1-EVT43dB-DeNoise'
        aim_cluster = 2
        QuickFunction.svm_file_kmeans_treat(file_cluster, filename, filetrace, label_kmeans, aim_cluster)
        print('svm_file_kmeans_treat over!')

    # 将主要cluster进行再一次的分类（检测里面是否有异常）
    if method == 16:
        filename = r'KINK-train-LPSOMg-0deg-Test1-EVT43dB-DeNoise-kmeans_treated'
        filetrace = r'C:\Users\liuhanqing\Desktop\research\Academic conference\data\Frequency Smoothing\event-200ms\AE_kmean_treat-Test'
        file = filetrace + '\\' + filename + '.csv'
        label = QuickFunction.main_cluster_kmeans_treat(file, filename, filetrace, smooth)
        QuickFunction.n_t_c(file, filetrace, filename, label)

    # 计算一个实验样本的标准化平均频谱
    if method == 17:
        filename = r'TWIN-training-Mg-Test1-EVT40dB-DeNoise'
        filetrace = r'C:\Users\liuhanqing\Desktop\research\Academic conference\data\Frequency Smoothing\normalize\event-200ms\AE'
        file = filetrace + '\\' + filename + '.csv'
        QuickFunction.averange_intensity_fre(file, filename, filetrace, smooth)

# 说明：
# 0：svm机器学习（无model）；
#    生成label并保存模型

# 1：svm机器学习（有model）；
#    生成label

# 2：K-Means机器学习（默认cluster=3）；
#    生成label

# 3：DeNoise；
#    需要手动设置文件名

# 4：position-time；

# 5：peak-frequency-time；

# 6：amplitude-counts（简介数据是否正常，分布是否为三角形）；

# 7：time - counts（所有AE的Count和Time）；

# 8：SVM number-time of different cluster（model=0为无model，model=1为有model）；
#    生成num_time_cluster的N_T_C文件

# 9：Kmeans number-time of different cluster；

# 10：SVM将结果将至2维进行可视化，还没写完

# 11：KMeans将结果将至2维进行可视化，还没写完

# 12：数据标准化，专门对分类后的平均频谱进行分类，需
#     要手动设置文件名

# 13：用已知信号类型的数据对模型准确性进行检测

# 14：用模型对用建立模型的数据进行分类，查看建立模型时的分类效果, 0:twin, 1:kink

# 15: 用kmeans的结果对svm处理的DeNosie文件进行处理（删除可能为噪声的数据），并生成主要cluster的频域csv文件

#     model:0没有模型，1有模型；smooth：0不光滑化，1光滑化
if __name__ == '__main__':
    main(method=8, model=0, smooth=1)