n = 3
label = [1,1,1,0,0,0,2,2,2]
fre = [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9]]
ave_fre_cluster = [[] for i in range(n)]
temp_fre = 0

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

print(ave_fre_cluster)