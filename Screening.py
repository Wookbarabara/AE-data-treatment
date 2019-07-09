# coding: utf-8
# test1 for compression test
import numpy as np
import DrawImage


# read the file
def read_file(file):
    f = np.loadtxt(file, dtype=np.str, delimiter=',')
    return f


# delete the data that both of its channels do not have signal or only one channel has a signal.
# 删除两个通道都没有值，或者删除只有一个通道有值的数据
def data_filter(file, channel=1):
    # remain signals that only one channel has a signal
    # 删除只有一个通道有值的数据
    data_meaningful = []
    if channel == 1:
        print('data_filter 1 is running')
        for i in file:
            if i[4] != '      ' or i[5] != '    ':
                data_meaningful.append(i)
    # remain signals with signals in both channel
    # 只保留两个通道有值的数据
    if channel == 2:
        print('data_filter 2 is running')
        for i in file:
            if i[4] != '      ' and i[5] != '    ':
                data_meaningful.append(i)
    return data_meaningful


# calculate the distance between the origin of AE signal and the center of sample
# 计算AE源相对中心的距离
# when the signal is out of the sample, delete this data
# the velocity of voice in Zn is 3.7km/s
# receive one AE data
def get_location(file, voice_speed, method=1):
    # set time
    if method == 1:
        time = float(file[12]) - float(file[13])
        # set velocity of voice in Zn
        speed = voice_speed
        distance = speed * time / 2
        return distance
    if method == 2:
        time1 = float(file[10]) - float(file[11])
        time2 = float(file[12]) - float(file[13])
        time3 = float(file[14]) - float(file[15])
        time = min(time1, time2, time3)
        # set velocity of voice in Zn
        speed = 3.7
        distance = speed * time / 2
        return distance


# select the AE sign which is out of the specimen and delete it
# 删除处于样品之外的噪音信号
# receive a full data
def exclude_noise(file, voice_speed):
    # set size of specimen
    specimen_size = 5.0
    data_meaningful = [file[0]]
    for i in range(1, len(file)):
        distance = abs(get_location(file[i], voice_speed, method=2))
        if distance <= specimen_size / 2:
            data_meaningful.append(file[i])
    return data_meaningful


# if the file need to be renumbered
# 对文件的序号进行重新编号
def renumber(file):
    n = 0
    for i in range(1, len(file)):
        n = n + 1
        file[i][0] = str(n)
    return file


# 提取出所有有用信号的频域向量
# event文件，以及要提取的event文件的文件夹
def data_fre(file, filetrace, smooth=1):
    print('data_fre is running')
    if smooth == 1:
        print('Smooth! ')
        fre = []
        frequency = []
        # 设定截取频域的上下限
        fre_min = 100
        fre_max = 1000
        file_name = 'event'
        file_type = '.csv'
        # 确定截取频率段
        f0 = filetrace + '\\' + file_name + '1' + file_type
        event1 = np.loadtxt(f0, dtype=np.str, delimiter='/t')
        temp0 = []
        for i in event1:
            temp0.append(i.strip(',').split(','))
        event1 = temp0
        for i in range(1, len(event1)):
            if float(event1[i][4]) <= fre_min:
                continue
            if float(event1[i][4]) >= fre_max:
                break
            frequency.append(event1[i][4])
        # frequency 里就是截取的频率段
        # 遍历所有event
        for file_num in range(1, len(file)):
            temp = []
            f = filetrace + '\\' + file_name + str(file_num) + file_type
            # print('data_fre for ', file_num/len(file)*100, '%')
            # 文件内部有','，不能用这个来隔开元素
            # 分类之后，将数据变成二重list
            event = np.loadtxt(f, dtype=np.str, delimiter='/t')
            temp2 = []
            for i in event:
                temp2.append(i.strip(',').split(','))
            event = temp2
            # 变成二重list了
            # 取两个ch的平均值
            for i in range(1, len(event)):
                if float(event[i][4]) <= fre_min:
                    continue
                if float(event[i][4]) >= fre_max:
                    break
                temp.append((float(event[i][5])+float(event[i][6]))/2)
            fre.append(temp)
        # 对fre进行光滑化, 和标准化
        for i in range(len(fre)):
            # 光滑化
            fre[i] = DrawImage.fre_smoothing(fre[i])
            # 标准化
            fre[i] = DrawImage.normalize(fre[i])
        result = [frequency, fre]
    if smooth == 0:
        print('No Smooth! ')
        fre = []
        frequency = []
        # 设定截取频域的上下限
        fre_min = 100
        fre_max = 1000
        file_name = 'event'
        file_type = '.csv'
        # 确定截取频率段
        f0 = filetrace + '\\' + file_name + '1' + file_type
        event1 = np.loadtxt(f0, dtype=np.str, delimiter='/t')
        temp0 = []
        for i in event1:
            temp0.append(i.strip(',').split(','))
        event1 = temp0
        for i in range(1, len(event1)):
            if float(event1[i][4]) <= fre_min:
                continue
            if float(event1[i][4]) >= fre_max:
                break
            frequency.append(event1[i][4])
        # frequency 里就是截取的频率段
        # 遍历所有event
        for file_num in range(1, len(file)):
            temp = []
            f = filetrace + '\\' + file_name + str(file_num) + file_type
            print('data_fre for ', file_num/len(file)*100, '%')
            # 文件内部有','，不能用这个来隔开元素
            # 分类之后，将数据变成二重list
            event = np.loadtxt(f, dtype=np.str, delimiter='/t')
            temp2 = []
            for i in event:
                temp2.append(i.strip(',').split(','))
            event = temp2
            for i in range(1, len(event)):
                if float(event[i][4]) <= fre_min:
                    continue
                if float(event[i][4]) >= fre_max:
                    break
                temp.append((float(event[i][5])+float(event[i][6]))/2)
            fre.append(temp)
            # 对fre进行光滑化, 和标准化
        for i in range(len(fre)):
            # 光滑化
            fre[i] = DrawImage.fre_smoothing(fre[i])
            # 标准化
            fre[i] = DrawImage.normalize(fre[i])
        result = [frequency, fre]
    return result  # [[frequency range], [[fre1],[fre2],[fre3]...[fre-n]]]


# 除去噪音数据
def noise_treat(file, filetrace, filename, channel=1):
    f = data_filter(file, channel=channel)
    make_file(f, filetrace, filename)


# make csv file
# 创建一个CSV文件
# receive the csv file and the name of the new file
def make_file(file, filetrace, filename):
    f = filetrace + '\\' + filename
    np.savetxt(f, file, fmt='%s', delimiter=',')
    print('Make file running!')