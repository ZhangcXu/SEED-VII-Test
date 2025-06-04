import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import time
from DataLoader import SEEDVIIDataset
from model import MAET
import math

# 设置超参数
output_dir = "./output/Improve/single_EEG+EYE"
single_test = False
eye_multi = True
dg_choice = True
pick_ratio = 6/7  # 单被试验证下训练集占总体的比例
batch_size = 64
learning_rate = 1e-4
epochs = 100
embed_dim = 32
depth = 3
num_workers = 0
num_heads = 4
drop_rate = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = '../SEED-VII'  # 根据实际路径修改


# 结果文件
result_file = os.path.join(output_dir, "MAET_results.txt")
accuracies = []

# 确保结果目录存在
os.makedirs(os.path.dirname(result_file), exist_ok=True)

if dg_choice:
    num_domains = 20
else:
    num_domains = None

# 20次循环
for i in range(1, 21):
    print(f"\n{'-'*20}")
    print(f"Training and Testing with subject {i} as test set")
    print(f"{'-'*20}")
    
    # 创建数据集
    if single_test:
        full_dataset = SEEDVIIDataset(
            root_dir=root_dir,
            feature_type='psd',
            sample_ratio=1.0,
            picked=i,
            eye_multi=eye_multi,
            single_test=True
        )

        train_size = int(pick_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    else:
        train_dataset = SEEDVIIDataset(
            root_dir=root_dir,
            feature_type='psd',
            sample_ratio=1.0,
            picked=i,
            is_test=False,
            eye_multi=eye_multi
        )

        test_dataset = SEEDVIIDataset(
            root_dir=root_dir,
            feature_type='psd',
            sample_ratio=1.0,
            picked=i,
            is_test=True,
            eye_multi=eye_multi
        )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
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
        domain_generalization=dg_choice,
        num_domains=num_domains
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 训练模型
    best_acc = 0.0
    print(f"\nStarting training for subject {i} test set...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for eeg, eye, labels, dom_labels in train_loader:
            # 转换数据类型并发送到设备
            eeg = eeg.float().to(device)
            if len(eye) == 0:
                eye = None
            else:
                eye = eye.float().to(device)

            # 如果启用域泛化，需要处理域标签
            if dg_choice:
                dom_labels = dom_labels.long().to(device)
            labels = labels.long().to(device)
            alpha = 2 / (1 + math.exp(-10 * epoch / epochs)) - 1
            
            # 前向传播
            optimizer.zero_grad()
            if dg_choice:
                # 域泛化模式：模型返回两个输出
                outputs, domain_outputs = model(eeg=eeg, eye=eye, alpha_=alpha)

                # 计算两个损失
                cls_loss = criterion(outputs, labels)
                domain_loss = criterion(domain_outputs, dom_labels)
                loss = cls_loss + domain_loss
            else:
                # 非域泛化模式
                outputs = model(eeg=eeg, eye=eye)
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
                if len(eye) == 0:
                    eye = None
                else:
                    eye = eye.float().to(device)
                labels = labels.long().to(device)

                if dg_choice:
                    outputs, _ = model(eeg=eeg, eye=eye, alpha_=0)
                else:
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
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_subject_{i}_best.pth"))
    
    # 记录最终准确率
    accuracies.append(best_acc)
    print(f"\nSubject {i} Test Accuracy: {best_acc:.4f}")
    
    # 保存本轮结果
    with open(result_file, "a") as f:
        f.write(f"Subject {i} Test Accuracy: {best_acc:.4f}\n")

# 计算统计结果
mean_acc = np.mean(accuracies)
idx, best_acc = np.argmax(accuracies), np.max(accuracies)
std_acc = np.std(accuracies)

print(f"\n{'-'*20}")
print(f"Final Results - Mean Accuracy: {mean_acc:.4f}, Std: {std_acc:.4f}")
print(f"{'-'*20}")

# 保存最终结果
with open(result_file, "a") as f:
    f.write("\n" + "-"*20 + "\n")
    f.write(f"Mean Accuracy: {mean_acc:.4f}\n")
    f.write(f"Std: {std_acc:.4f}\n")
    f.write(f"Best_acc: {best_acc:.4f}, at location: {idx+1}\n")
    f.write("-"*20 + "\n")
    f.write("All accuracies:\n")
    for i, acc in enumerate(accuracies, 1):
        f.write(f"Subject {i}: {acc:.4f}\n")

print("Training and evaluation completed. Results saved to", result_file)