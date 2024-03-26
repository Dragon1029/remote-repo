import pandas as pd
from sklearn.model_selection import train_test_split
import os
# 切割划分
def split_data_with_overlap(data, time_steps, overlap_ratio=0.5):
    """
    拆分具有重叠的数据。

    参数:
        data (类数组): 要拆分的输入数据。
        time_steps (整数): 每个样本的长度。
        overlap_ratio (浮点数): 相邻样本之间重叠的比例。默认为0.5。

    返回:
        DataFrame: 包含重叠样本的DataFrame。
    """
    step_size = int(time_steps * (1 - overlap_ratio))
    num_samples = (len(data) - time_steps) // step_size + 1

    samples = []
    for i in range(num_samples):
        start = i * step_size
        end = start + time_steps
        samples.append(data[start:end].values.flatten())  # 将数据转换为数组后再进行展平操作

    return pd.DataFrame(samples)

#归一化数据
def normalize_data(data):
    """
        归一化数据。

        参数:
            data (DataFrame): 要归一化的数据。

        返回:
            DataFrame: 归一化后的数据。
        """
    mean = data.mean()
    std = data.std()
    normalized_data = (data - mean) / std
    return normalized_data
#数据集的制作
def make_dataset(data, label_list=[], split_rate=[0.7, 0.2, 0.1], random_state=None):
    """
    将数据集划分为训练集、验证集和测试集。

    参数:
        data (DataFrame): 要划分的数据集。
        label_list (列表): 数据标签列表。默认为空列表。
        split_rate (列表): 数据集划分比例，包括训练集、验证集和测试集。默认为[0.7, 0.2, 0.1]。
        random_state (整数或None): 随机种子。默认为None。

    返回:
        DataFrame: 训练集。
        DataFrame: 验证集。
        DataFrame: 测试集。
    """
    assert len(split_rate) == 3, "split_rate参数应包括三个值，分别为训练集、验证集和测试集的比例"

    # 如果没有提供标签列表，则将数据的最后一列作为标签列
    if not label_list:
        features = data.iloc[:, :-1]
        labels = data.iloc[:, -1]
    else:
        # 如果提供了标签列表，从数据中提取对应的标签列和特征列
        features = data.drop(columns=label_list)
        labels = data[label_list]

    # 划分数据集
    train_size, val_size, test_size = split_rate
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    train_data, val_data, train_labels, val_labels = train_test_split(train_val_data, train_val_labels, test_size=val_size / (val_size + train_size), random_state=random_state)

    return train_data, val_data, test_data

# 从CSV文件中读取数据
data = pd.read_csv(r'D:\git_test\origin\凯斯西储大学数据基测试\1.基于卷积神经网络的故障诊断\code\data_12k_1797_10c.csv')  # 假设数据是以CSV格式保存的

# 切分数据
time_steps = 512
overlap_ratio = 0.5
samples_df = split_data_with_overlap(data, time_steps, overlap_ratio)

#划分训练集、验证集、测试集[0.7,0.2,0.1]
train_data, val_data, test_data = make_dataset(samples_df)
# 创建保存数据集文件的文件夹
dataset_folder = 'dataset'
os.makedirs(dataset_folder, exist_ok=True)
# 保存训练集、验证集和测试集为CSV文件
train_data.to_csv(os.path.join(dataset_folder, 'train_data.csv'), index=False)
val_data.to_csv(os.path.join(dataset_folder, 'val_data.csv'), index=False)
test_data.to_csv(os.path.join(dataset_folder, 'test_data.csv'), index=False)

