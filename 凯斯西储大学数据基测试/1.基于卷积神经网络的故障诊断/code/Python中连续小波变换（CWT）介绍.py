import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import loadmat
import pandas as pd
import warnings  # 导入warnings模块，用于忽略警告
warnings.filterwarnings('ignore')  # 忽略警告信息
# 设置中文显示 将默认字体设置为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
# 加载数据
data3 = loadmat(r'D:\git_test\origin\凯斯西储大学数据基测试\十分类数据集\12k\105.mat')

# 将数据展平为一维数组
data_list3 = data3['X105_DE_time'].reshape(-1)

# 获取前512个数据点
data = data_list3[0:512]

# 设置采样周期
sampling_period = 1.0 / 12000

# 设置小波变换参数
totalscal = 128
wavename = 'cmor1-1'
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 0, -1)

# 进行连续小波变换，将变换系数存储在coefficients中，频率信息存储在frequencies
coefficients, frequencies = pywt.cwt(data, scales, wavename, sampling_period)

# 计算变换系数的幅度
amp = abs(coefficients)
frequ_max = frequencies.max()

# 根据采样周期sampling_period生成时间轴t
t = np.linspace(0, sampling_period * 512, 512)

# 设置全局字体
plt.rcParams['font.family'] = 'Arial Unicode MS'

# 绘制等高线图
plt.contourf(t, frequencies, amp, cmap='jet')
plt.title('滚珠-512-128-cmor1-1')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude')
plt.show()
