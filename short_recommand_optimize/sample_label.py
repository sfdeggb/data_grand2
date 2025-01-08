import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

def generate_watch_count_labels(df):
    """
    为每条记录生成用户从该时间点到当天结束的观看视频数量标签
    优化版本，使用groupby和向量化操作
    """
    # 确保时间戳格式正确
    df['oper_time'] = pd.to_datetime(df['oper_time'], format='mixed')
    
    # 添加日期列，避免重复计算
    df['date'] = df['oper_time'].dt.date
    
    # 按用户和日期分组，提前计算每组的大小
    result = df.copy()
    result['label'] = 0
    
    # 使用groupby进行批处理
    for (user, date), group in tqdm(df.groupby(['serv_number', 'date']), desc="处理用户日期组"):
        # 对该组数据按时间排序
        group = group.sort_values('oper_time')
        group_idx = group.index
        
        # 使用numpy的广播特性计算标签
        times = group['oper_time'].values
        labels = np.sum(times.reshape(-1, 1) <= times.reshape(1, -1), axis=1)
        
        # 批量更新结果
        result.loc[group_idx, 'label'] = labels
    
    return result

# 如果内存足够，可以使用更快的全向量化版本
def generate_watch_count_labels_vectorized(df):
    """
    完全向量化的版本，适用于内存充足的情况
    """
    # 确保时间戳格式正确
    df['oper_time'] = pd.to_datetime(df['oper_time'], format='mixed')
    df['date'] = df['oper_time'].dt.date
    
    result = df.copy()
    
    # 使用apply替代循环
    def count_remaining(group):
        group = group.sort_values('oper_time')
        times = group['oper_time'].values
        return np.sum(times.reshape(-1, 1) <= times.reshape(1, -1), axis=1)
    
    # 按用户和日期分组进行向量化计算
    result['label'] = df.groupby(['serv_number', 'date']).apply(
        lambda x: pd.Series(count_remaining(x), index=x.index)
    ).values
    
    return result

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    data = [
        {"statis_date": "20241106", "serv_number": "13487028369", "video_id": "45211", "env_video_time": "38.0", "oper_time": "2024-11-06 00:32:29.130"},
        {"statis_date": "20241106", "serv_number": "13457692971", "video_id": "44963", "env_video_time": "38.0", "oper_time": "2024-11-06 00:32:37"},
        {"statis_date": "20241106", "serv_number": "13487028369", "video_id": "48948", "env_video_time": "100.0", "oper_time": "2024-11-06 00:47:47"},
        {"statis_date": "20241107", "serv_number": "13548466410", "video_id": "48082", "env_video_time": "72.0", "oper_time": "2024-11-07 00:23:46.269"},
        {"statis_date": "20241107", "serv_number": "13457692971", "video_id": "44963", "env_video_time": "0.0", "oper_time": "2024-11-07 00:24:58"}
    ]
    
    df = pd.DataFrame(data)
    
    # 生成标签
    #labeled_df = generate_watch_count_labels(df)  # 使用优化版本
    # 或者使用完全向量化版本（如果内存允许）
    labeled_df = generate_watch_count_labels_vectorized(df)
    
    print("带标签的数据:")
    print(labeled_df)