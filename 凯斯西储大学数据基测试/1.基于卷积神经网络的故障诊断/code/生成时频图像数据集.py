import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import loadmat
import pandas as pd
import warnings  # 导入warnings模块，用于忽略警告
warnings.filterwarnings('ignore')  # 忽略警告信息


# 导入数据集加载数据集
train_set = pd.read_csv(r'D:\git_test\origin\凯斯西储大学数据基测试\1.基于卷积神经网络的故障诊断\code\dataset\train_data.csv')
val_set = pd.read_csv(r'D:\git_test\origin\凯斯西储大学数据基测试\1.基于卷积神经网络的故障诊断\code\dataset\val_data.csv')
test_set = pd.read_csv(r'D:\git_test\origin\凯斯西储大学数据基测试\1.基于卷积神经网络的故障诊断\code\dataset\test_data.csv')

data = [train_set, val_set, test_set]


#定义一个连续小波变换的函数
def continuous_wavelet_transform(data, wavename='cmor1-1',
                                 totalscal=128,
                                 sampling_period=1.0/12000):
    # 将数据展平为一维数组
    data = data.reshape(-1)

    # 获取前512个数据点
    data = data[:512]

    # 设置小波变换参数
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 0, -1)

    # 进行连续小波变换，将变换系数存储在coefficients中，频率信息存储在frequencies
    coefficients, frequencies = pywt.cwt(data, scales, wavename, sampling_period)
    # 计算变换系数的幅度
    amp = abs(coefficients)

    # 根据采样周期sampling_period生成时间轴t
    t = np.linspace(0, sampling_period * 512, 512)
    return coefficients, frequencies, amp, t

# 生成时频图片
def makeTimeFrequencyImage(data, img_path, img_size):
    """
    生成时频图像并保存为图像文件。

    参数:
        data (DataFrame): 包含时间序列数据的DataFrame。
        img_path (字符串): 图像文件保存路径。
        img_size (元组): 图像的尺寸 (width, height)。

    返回:
        无
    """
    # 获取数据
    time_series = data.values

    # 绘制时频图像
    plt.figure(figsize=img_size)
    plt.imshow(np.abs(np.fft.fftshift(np.fft.fft(time_series, axis=1))), aspect='auto', cmap='hot', origin='lower')
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.title('Time-Frequency Image')
    plt.colorbar(label='Amplitude')
    plt.savefig(img_path)
    plt.close()


# 生成图片数据集
def GenerateImageDataset(path_list, data_set):
    for i, path in enumerate(path_list):
        dataset_path = path
        os.makedirs(dataset_path, exist_ok=True)
        for j, data in enumerate(data_set[i]):
            img_path = os.path.join(dataset_path, f'sample_{j}_time_frequency.png')
            makeTimeFrequencyImage(data, img_path, img_size=(8, 6))  # 这里设置图片大小为 (8, 6) 英寸

# 定义保存图片数据集的文件夹路径
image_dataset_folder = r'D:\git_test\origin\凯斯西缪斯基测试\1.基于卷积神经网络的故障诊断\code\image_dataset'
# 调用生成图片数据集函数
GenerateImageDataset([os.path.join(image_dataset_folder, 'train'),
                      os.path.join(image_dataset_folder, 'val'),
                      os.path.join(image_dataset_folder, 'test')], data)



