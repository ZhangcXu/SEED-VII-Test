import scipy.io
import numpy as np

# 读取 .mat 文件
mat_data = scipy.io.loadmat('../SEED-VII/EYE_features/1.mat')  # 替换为你的文件名

# 打印所有键（变量名）
print("所有键:", list(mat_data.keys()))

# 过滤掉 MATLAB 系统变量（通常以双下划线开头）
user_keys = [key for key in mat_data.keys() if not key.startswith('__')]
print("\n用户数据键:", user_keys)

cnt = 0
# 分析每个用户变量的数据类型和尺寸
#print("\n变量详情:")
for key in user_keys:
    data = mat_data[key]
    # print(f"\n变量名: {key}")
    # print(f"数据类型: {type(data).__name__}")
    
    # 如果是 NumPy 数组
    if isinstance(data, np.ndarray):
        print(f"数组尺寸: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"最小值: {np.nanmin(data)} | 最大值: {np.nanmax(data)}")
        cnt += data.shape[0]
    
    # 如果是其他类型（如字符串、单元格等）
    else:
        print(f"内容: {data}")
        print(f"长度: {len(data)}" if hasattr(data, '__len__') else "非序列类型")
        
print("总时长：{}".format(cnt))