import DrawImage

fre = [1,2,3,4,5,6,7]

fre = DrawImage.fre_smoothing(fre)
print(fre)
# 标准化
fre = DrawImage.normalize(fre)

print(fre)