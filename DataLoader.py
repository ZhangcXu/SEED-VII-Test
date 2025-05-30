import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat

batch_size = 32

class SEEDVIIEEGDataset(Dataset):
    def __init__(self, root_dir, feature_type='psd', 
                 sample_ratio=1.0, label_mapping=None):
        """
        SEED-VII EEG数据集加载器
        
        参数:
            root_dir (str): 包含.mat文件的根目录
            feature_type (str): 使用的特征类型 ('psd', 'de', 或 'de_LDS')
            sample_ratio (float): 数据采样比例 (0.0-1.0)
            label_mapping (dict): 情绪名称到数字标签的映射
        """
        self.root_dir = root_dir
        self.feature_type = feature_type
        self.sample_ratio = sample_ratio
        self.data = []
        self.labels = []
        self.domain_labels = []
        
        # 情绪标签映射 (默认映射)
        self.label_mapping = label_mapping or {
            'Disgust': 0, 'Fear': 1, 'Sad': 2, 'Neutral': 3,
            'Happy': 4, 'Anger': 5, 'Surprise': 6
        }
        
        # 加载情绪标签映射
        self._load_emotion_labels()
        
        # 加载所有被试的数据
        for subject_id in range(1, 21):
            subject_data, subject_labels, subject_domain_labels = self._load_subject_data(subject_id)
            self.data.extend(subject_data)
            self.labels.extend(subject_labels)
            self.domain_labels.extend(subject_domain_labels)
        
        # 数据采样
        if sample_ratio < 1.0:
            self._sample_data()
    
    def _load_emotion_labels(self):
        """从Excel文件加载情绪标签映射"""
        excel_path = os.path.join(self.root_dir, 'emotion_label_and_stimuli_order.xlsx')
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Label file not found: {excel_path}")
        
        # 读取Excel文件
        df = pd.read_excel(excel_path, header=None)
        
        # 解析标签数据
        self.segment_to_emotion = {}
        current_session = None
        
        for i in range(len(df)):
            row = df.iloc[i].values
            # 检查是否是会话标题行
            if "Session" in str(row[0]):
                current_session = str(row[0]).split()[0] + " " + str(row[0]).split()[1]
                continue
            
            # 处理视频索引行
            if "Video index" in str(row[0]):
                video_indices = [int(x) for x in row[1:21] if not pd.isna(x)]
                next_row = df.iloc[i+1].values
                emotions = [str(x).strip() for x in next_row[1:21] if not pd.isna(x)]
                
                # 将视频索引映射到情绪
                for idx, emotion in zip(video_indices, emotions):
                    self.segment_to_emotion[idx] = emotion
    
    def _load_subject_data(self, subject_id):
        """加载单个被试的数据"""
        file_path = os.path.join(self.root_dir, f'EEG_features/{subject_id}.mat')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Subject data not found: {file_path}")
        
        mat_data = loadmat(file_path)
        subject_data = []
        subject_labels = []
        subject_doamin_labels = []
        
        for seg_id in range(1, 81):
            # 获取情绪标签
            emotion = self.segment_to_emotion.get(seg_id)
            if emotion is None:
                raise ValueError(f"Label not found for segment {seg_id}")
            
            label = self.label_mapping.get(emotion)
            if label is None:
                raise ValueError(f"Unknown emotion: {emotion}")
            
            # 构建特征键名
            key = f"{self.feature_type}_{seg_id}"
            if key not in mat_data:
                raise KeyError(f"Feature {key} not found in {file_path}")
            
            # 获取特征数组 (time_steps, 5, 62)
            feature_array = mat_data[key]
            
            # 展平最后两个维度 (5, 62) -> (310)
            flattened_features = feature_array.reshape(feature_array.shape[0], -1)
            for idx in range(len(flattened_features)):              
                subject_data.append(flattened_features[idx])
                subject_labels.append(label)
                subject_doamin_labels.append(subject_id - 1) # pytorch的标签必须从0开始标记 
        
        return subject_data, subject_labels, subject_doamin_labels
    
    def _sample_data(self):
        """随机采样部分数据"""
        total_samples = len(self.data)
        sample_size = int(total_samples * self.sample_ratio)
        
        # 随机选择索引
        indices = np.random.choice(total_samples, sample_size, replace=False)
        
        # 更新数据和标签
        self.data = [self.data[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.domain_labels = [self.domain_labels[i] for i in indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取特征数组和时间步数量
        features = self.data[idx]
        label = self.labels[idx]
        domain_label = self.domain_labels[idx]
        
        # 转换为PyTorch张量
        features = torch.tensor(features, dtype=torch.float64)
        label = torch.tensor(label, dtype=torch.long)
        domain_label = torch.tensor(domain_label, dtype=torch.long)
        
        return features, label, domain_label

# 使用示例
if __name__ == "__main__":
    # 初始化数据集
    full_dataset = SEEDVIIEEGDataset(
        root_dir='../SEED-VII',
        feature_type='psd',  # 使用PSD特征
        sample_ratio=1.0  
    )
    
    # 划分训练集和测试集 (6:1)
    train_size = int(0.857 * len(full_dataset))  # 6/7 ≈ 0.857
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size]
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 测试数据加载
    for batch_idx, (features, labels, domain_labels) in enumerate(train_loader):
        print(f"Train Batch {batch_idx}:")
        print(f"Features shape: {features.shape}")  # 应为 (batch_size, 310)
        print(f"Labels shape: {labels.shape}")      # 应为 (batch_size)
        
        if batch_idx == 2:  # 只测试前3个batch
            break
    
    for batch_idx, (features, labels, domain_labels) in enumerate(test_loader):
        print(f"Test Batch {batch_idx}:")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
        if batch_idx == 2:
            break