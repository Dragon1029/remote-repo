import os
import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt


# 定义连续小波变换函数
def continuous_wavelet_transform(data, wavename='cmor1-1', totalscal=128, sampling_period=1.0 / 12000):
    data = data.values.reshape(-1)
    data = data[:512]
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 0, -1)
    coefficients, frequencies = pywt.cwt(data, scales, wavename, sampling_period)
    return coefficients


# 读取CSV文件并进行连续小波变换
def process_csv_file(csv_file_path, output_folder):
    data = pd.read_csv(csv_file_path)
    for idx, row in data.iterrows():
        # 进行连续小波变换
        coefficients = continuous_wavelet_transform(row)

        # 绘制时频图像
        plt.imshow(np.abs(np.fft.fftshift(np.fft.fft(coefficients, axis=1))), aspect='auto', cmap='hot', origin='lower')
        plt.colorbar(label='Amplitude')
        plt.xlabel('Frequency')
        plt.ylabel('Time')
        plt.title('Time-Frequency Image')

        # 保存时频图像
        img_path = os.path.join(output_folder, f'image_{idx}.png')
        plt.savefig(img_path)
        plt.close()


# 定义CSV文件夹和输出文件夹
csv_folder = r'D:\git_test\origin\凯斯西储大学数据基测试\1.基于卷积神经网络的故障诊断\code\dataset'
output_folder = r'D:\git_test\origin\凯斯西储大学数据基测试\1.基于卷积神经网络的故障诊断\code\image_dataset'

# 为训练集、测试集和验证集的每个CSV文件执行处理
for csv_file_name in ['train_data.csv', 'test_data.csv', 'val_data.csv']:
    csv_file_path = os.path.join(csv_folder, csv_file_name)
    process_csv_file(csv_file_path, output_folder)
