# ********************* TE_data_str2float.py **********************
"""
实现功能：
1.处理字符型数据：识别并处理数据集中无法识别的字符型数据（将其置为0，个数很少）
2.缺失值处理：用前面的数据填补缺失值

"""
import pandas as pd

# 导入数据
data_path = "D:\\Python Codes\\TE\\TE_data\\csv"
train_data = pd.read_csv(data_path + "\\Train\\d" + str(0) + ".csv", header=None, prefix='V', skiprows=[0])
test_data = pd.read_csv(data_path + "\\Test\\d" + str(0) + "_te.csv", header=None, prefix='V', skiprows=[0])
for i in range(1, 22):
    # 训练数据
    train_data_path = data_path + "\\Train\\d" + str(i) + ".csv"
    train_next_data = pd.read_csv(train_data_path, header=None, prefix='V')
    train_data = pd.concat([train_data, train_next_data])
    # 测试数据
    test_data_path = data_path + "\\Test\\d" + str(i) + "_te.csv"
    test_next_data = pd.read_csv(test_data_path, header=None, prefix='V')
    test_data = pd.concat([test_data, test_next_data])


# 处理训练集
for row in range(train_data.shape[0]):
    print(row)
    for col in range(train_data.shape[1]):
        if isinstance(train_data.iloc[row, col], str):
            if len(train_data.iloc[row, col]) > 15:
                train_data.iloc[row, col] = train_data.iloc[row, col][0:13]
# 异常值处理
train_data = train_data.fillna(method='ffill')
# 写入csv文件
train_data.to_csv('D:\\Python Codes\\TE\\TE_Train_data.csv', index=False)


# 处理测试集
for row in range(test_data.shape[0]):
    print(row)
    for col in range(test_data.shape[1]):
        if isinstance(test_data.iloc[row, col], str):
            if len(test_data.iloc[row, col]) > 15:
                test_data.iloc[row, col] = test_data.iloc[row, col][0:13]
# 异常值处理
test_data = test_data.fillna(method='ffill')
# 写入csv文件
test_data.to_csv('D:\\Python Codes\\TE\\TE_Test_data.csv', index=False)









