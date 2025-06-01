import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
import pandas as pd
from sklearn.metrics import accuracy_score
import time
from DataLoader import SEEDVIIDataset
from model import MAET

# 设置超参数
batch_size = 32
learning_rate = 1e-3
epochs = 30
embed_dim = 32
depth = 3
num_heads = 4
drop_rate = 0.1
device = torch.device("mps" if torch.mps.is_available() else "cpu")
root_dir = '../SEED-VII'  # 根据实际路径修改

# 结果文件
result_file = "./output/EEG/MAET_results.txt"
accuracies = []

# 确保结果目录存在
os.makedirs(os.path.dirname(result_file), exist_ok=True)

# 20次循环（每个被试作为测试集一次）
for i in range(1, 21):
    print(f"\n{'-'*20}")
    print(f"Training and Testing with subject {i} as test set")
    print(f"{'-'*20}")
    
    # 创建数据集
    train_dataset = SEEDVIIDataset(
        root_dir=root_dir,
        feature_type='psd',
        sample_ratio=1.0,
        picked=i,
        is_test=False,
        eye_multi=False
    )
    
    test_dataset = SEEDVIIDataset(
        root_dir=root_dir,
        feature_type='psd',
        sample_ratio=1.0,
        picked=i,
        is_test=True,
        eye_multi=False
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
    
    # 初始化模型
    model = MAET(
        eeg_dim=310,
        eye_dim=33,
        num_classes=7,
        embed_dim=embed_dim,
        depth=depth,
        eeg_seq_len=5,
        eye_seq_len=5,
        num_heads=num_heads,
        drop_rate=drop_rate,
        domain_generalization=False,
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # 训练模型
    best_acc = 0.0
    print(f"\nStarting training for subject {i} test set...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for eeg, eye, labels, _ in train_loader:
            # 转换数据类型并发送到设备
            eeg = eeg.float().to(device)
            labels = labels.long().to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(eeg=eeg)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * eeg.size(0)
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader.dataset)
        
        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for eeg, eye, labels, _ in test_loader:
                eeg = eeg.float().to(device)
                labels = labels.long().to(device)
                eye = None
                
                outputs = model(eeg=eeg, eye=eye)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        scheduler.step(acc)  # 根据验证性能调整学习率
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f} - Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"./output/EEG/model_subject_{i}_best.pth")
    
    # 记录最终准确率
    accuracies.append(best_acc)
    print(f"\nSubject {i} Test Accuracy: {best_acc:.4f}")
    
    # 保存本轮结果
    with open(result_file, "a") as f:
        f.write(f"Subject {i} Test Accuracy: {best_acc:.4f}\n")

# 计算统计结果
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print(f"\n{'-'*20}")
print(f"Final Results - Mean Accuracy: {mean_acc:.4f}, Std: {std_acc:.4f}")
print(f"{'-'*20}")

# 保存最终结果
with open(result_file, "a") as f:
    f.write("\n" + "-"*20 + "\n")
    f.write(f"Mean Accuracy: {mean_acc:.4f}\n")
    f.write(f"Std: {std_acc:.4f}\n")
    f.write("-"*20 + "\n")
    f.write("All accuracies:\n")
    for i, acc in enumerate(accuracies, 1):
        f.write(f"Subject {i}: {acc:.4f}\n")

print("Training and evaluation completed. Results saved to", result_file)