import pandas as pd
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import warnings  # 导入warnings模块，用于忽略警告
import webbrowser
warnings.filterwarnings('ignore')  # 忽略警告信息
# 设置中文显示 将默认字体设置为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']

# 驱动端数据列名
data_columns = ['X118_DE_time', 'X185_DE_time', 'X222_DE_time', 'X105_DE_time', 'X169_DE_time',
                'X209_DE_time', 'X130_DE_time', 'X197_DE_time', 'X234_DE_time', 'X097_FE_time']

# 创建DataFrame的列名
columns_name = ['de_7_ball', 'de_14_ball', 'de_21_ball', 'de_7_inner', 'de_14_inner',
                'de_21_inner', 'de_7_outer', 'de_14_outer', 'de_21_outer', 'de_normal']

data_12k_1797_10c = pd.DataFrame()

# 指定文件夹路径
folder_path = r'D:\git_test\origin\凯斯西储大学数据基测试\1.基于卷积神经网络的故障诊断\data\0HP'

# 获取文件夹中的所有MAT文件
mat_files = [file for file in os.listdir(folder_path) if file.endswith('.mat')]

for index, mat_file in enumerate(mat_files):
    # 构建完整的文件路径
    文件路径 = os.path.join(folder_path, mat_file)

    # 读取MAT文件
    数据 = loadmat(文件路径)

    # 提取DE_time数据
    数据列表 = 数据[data_columns[index]].reshape(-1)

    # 将数据分配给DataFrame
    data_12k_1797_10c[columns_name[index]] = 数据列表[:119808]  # 根据需要调整切片范围

# 保存CSV文件
data_12k_1797_10c.to_csv('data_12k_1797_10c.csv', index=False)

# 读取CSV文件的前1000行数据
df = pd.read_csv('data_12k_1797_10c.csv', nrows=1000)

# 获取当前脚本所在的目录路径
script_directory = os.path.dirname(os.path.abspath(__file__))

# 拼接目录路径，创建 'sequence_chart' 文件夹的完整路径
output_directory = os.path.join(script_directory, 'sequence_chart')

# 获取数据框的列名
columns = df.columns
# 创建包含10个子图的2x5网格
fig, axs = plt.subplots(5, 2, figsize=(14, 20), sharex=True)
# 绘制每个标签的单独图
# 子图循环：将列名按顺序分配给子图，并在每个子图上绘制对应列的振动信号图
for i in range(5):
    for j in range(2):
        index = i * 2 + j
        if index < len(columns):
            axs[i, j].plot(df[columns[index]])
            axs[i, j].set_title(columns[index])
            axs[i, j].set_xlabel('Time')
            axs[i, j].set_ylabel('Vibration Signal')
# 将整个图表保存为PNG文件
plt.savefig('sequence_chart/my_sequence_plot.png')

# 调整布局
plt.tight_layout()

# 自动打开生成的图表
webbrowser.open(os.path.join(output_directory, 'my_sequence_plot.png'))
