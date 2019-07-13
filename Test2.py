from sklearn.model_selection import train_test_split
train_fre = [1,2,3,4,5,6,7,8,9,0]
train_label = [1,1,1,1,1,0,0,0,0,0]



for i in range(10):
    print(i)
    x_train, x_test, y_train, y_test = train_test_split(train_fre, train_label, test_size=0.4)
    print('x-train: ',x_train)
    print('x-test: ', x_test)
    print('y_train: ', y_train)
    print('y_test: ', y_test)