from scipy.io import loadmat  # 导入loadmat函数，用于加载MATLAB文件

# 加载MAT文件
mat_data = loadmat(r'D:\git_test\origin\凯斯西储大学数据基测试\1.基于卷积神经网络的故障诊断\data\0HP\12k_Drive_End_B007_0_118.mat')

# 查看MAT文件中的键
keys = list(mat_data.keys())  # 获取MAT文件中所有键，并将其转换为列表
print("Keys in the MAT file:")  # 打印提示信息
for key in keys:  # 遍历所有键
    print(str(key))  # 打印每个键并转换为字符串类型
