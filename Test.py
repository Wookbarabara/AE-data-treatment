import numpy as np
import math

n = 1435
time = []
pro = []
file_path = r'C:\Users\liuhanqing\Desktop\AE\LPSOMg-0deg-Test1-EVT43dB-DeNoise'
for i in range(n):
    print('number of event: ', i+1)
    f = file_path + '\\' + 'event' + str(i+1) + r'.csv'
    event = np.loadtxt(f, dtype=np.str, delimiter='/t')
    temp2 = []
    for i in event:
        temp2.append(i.strip(',').split(','))
    event = temp2
    temp = []
    for i in range(1, len(event)):
        temp.append(event[i][2])
    dict1 = {}
    for i in temp:
        if float(i) in dict1:
            dict1[float(i)] = dict1[float(i)] + 1
            # print(dict1)
        else:
            dict1[float(i)] = 1
    file_length = len(event) - 1
    for key in dict1:
        dict1[key] = dict1[key] / file_length
    total_pro = 0
    for key in dict1:
        total_pro = total_pro - dict1[key] * math.log(dict1[key])
    pro.append(total_pro)
    time.append(float(event[1][0]))
result = [['Time', 'H']]
for i in range(len(time)):
    result.append([time[i], pro[i]])

filename = r'result.csv'
f = file_path + '\\'+  filename
np.savetxt(f, result, fmt='%s', delimiter=',')