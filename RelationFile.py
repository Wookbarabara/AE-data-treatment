# coding: utf-8
# test1 for compression test
import numpy as np
import Screening
import math


# 绘制AE信号位置及时间关系的CSV文件
def position_time(file, filetrace, filename, voice_speed,channel=1):
    result = [['type', '2', ''],
              ['',   '',    ''],
              ['TriggerTime/s', '位置', 'Amp/dB']]
    temp = []
    if channel == 1:
        print('P_T 1 is running!')
        for i in range(1, len(file)):
            temp.append(file[i][2])
            temp.append(Screening.get_location(file[i], voice_speed, method=2))
            if file[i][6] == '      ':
                temp.append(float(file[i][7]))
            if file[i][7] == '    ':
                temp.append(float(file[i][6]))
            if file[i][6] != '      ' and file[i][7] != '    ':
                ave = (float(file[i][6]) + float(file[i][7])) / 2
                temp.append(ave)
            result.append(temp)
            temp = []
    if channel == 2:
        print('P_T 2 is running!')
        for i in range(1, len(file)):
            temp.append(file[i][2])
            temp.append(Screening.get_location(file[i], method=2))
            ave = (float(file[i][6]) + float(file[i][7])) / 2
            temp.append(ave)
            result.append(temp)
            temp = []
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename + '-P_T.csv'
    np.savetxt(f, result, fmt='%s', delimiter=',')


# 绘制AE信号peak frequency及时间关系的CSV文件
def peak_fre_time(file, filetrace, filename, channel=1):
    result = [['type', '2', ''],
              ['',   '',    ''],
              ['TriggerTime/s', 'PeakFre/Hz', 'Amp/dB']]
    temp = []
    if channel == 1:
        print('P_T 1 is running!')
        for i in range(1, len(file)):
            temp.append(file[i][2])
            temp.append(file[i][24])
            if file[i][6] == '      ':
                temp.append(float(file[i][7]))
            if file[i][7] == '    ':
                temp.append(float(file[i][6]))
            if file[i][6] != '      ' and file[i][7] != '    ':
                ave = (float(file[i][6]) + float(file[i][7])) / 2
                temp.append(ave)
            result.append(temp)
            temp = []
    if channel == 2:
        print('P_T 2 is running!')
        for i in range(1, len(file)):
            temp.append(file[i][2])
            temp.append(Screening.get_location(file[i], method=2))
            ave = (float(file[i][6]) + float(file[i][7])) / 2
            temp.append(ave)
            result.append(temp)
            temp = []
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename + '-PF_T.csv'
    np.savetxt(f, result, fmt='%s', delimiter=',')


# 绘制AE信号amplitude和counts之间关系的CSV文件
def amp_counts(file, filetrace, filename):
    amp_count = {}
    print('A_C 1 is running!')
    for i in range(1, len(file)):
        if file[i][7] != '    ':
            if float(file[i][7]) not in amp_count:
                amp_count[float(file[i][7])] = 1
            else:
                amp_count[float(file[i][7])] = amp_count[float(file[i][7])] + 1
        else:
            if float(file[i][6]) not in amp_count:
                amp_count[float(file[i][6])] = 1
            else:
                amp_count[float(file[i][6])] = amp_count[float(file[i][6])] + 1
    result = [['Amplitude', 'Counts']]
    # 对key进行排序
    # print(amp_count.keys())
    temp = ([[float(k), math.log(amp_count[k])] for k in sorted(amp_count.keys())])
    for i in temp:
        result.append(i)
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename + '-A_C.csv'
    np.savetxt(f, result, fmt='%s', delimiter=',')


# 绘制AE信号个数与时间关系
def time_count(file, filetrace, filename):
    num = 0
    result = [['Time', 'Count']]
    for i in range(len(file)-1):
        num = num + 1
        # 第一位是’TiggerTime‘
        result.append([file[i+1][2],num])
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename + '-T_C.csv'
    np.savetxt(f, result, fmt='%s', delimiter=',')


# 绘制不同cluster中AE信号个数与时间关系
# 在label里 0 是twin， 1是kink， file 是DeNoise文件
def num_time_cluster(file, filetrace, filename, label):
    temp_num = []
    temp_time = []
    result_dict = {}
    num_cluters = [i for i in range(len(set(label)))]
    for i in num_cluters:
        result_dict[i] = 0
    # 对所有元素进行遍历
    for j in range(len(label)):
        # 如果元素数属于这个cluster
        temp = []
        if label[j] in result_dict:
            result_dict[label[j]] = result_dict[label[j]] + 1  # 字典记录最新个数
            for key in result_dict:
                temp.append(result_dict[key])
            # print(temp)
            temp_num.append(temp)  # 更新一次每个cluster的数量
            temp_time.append(float(file[j + 1][2]) + 1)  # 更新对应时间
    # 将每种cluster里的个数随时间变化存到list里。
    # temp_num = [[0,0,0],[1,0,0],[2,0,0]...[x,y,z]]
    # temp_time = [time1,time2,time3...time-n]
    temp = ['Time']
    result = []
    for i in range(len(num_cluters)):
        temp.append(i)
    result.append(temp)
    temp = []
    for i in range(len(temp_num)):
        temp.append(temp_time[i])
        for j in temp_num[i]:
            temp.append(j)
        result.append(temp)
        temp = []
    # result = [
    #              ['Time', '0', '1', '2'],
    #              [                     ]
    #           ]
    f = filetrace + '\\'+ 'File after Processing' + '\\' + filename + '-N_T_C.csv'
    np.savetxt(f, result, fmt='%s', delimiter=',')
