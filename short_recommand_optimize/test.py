from datetime import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd 

def generate_watch_count_labels_vectorized(df):
    """
    完全向量化的版本，适用于内存充足的情况
    """
    print(f"开始处理数据，总数据量: {len(df):,} 条")
    start_time = datetime.now()
    
    # 确保时间戳格式正确
    print("正在处理时间戳...")
    df['oper_time'] = pd.to_datetime(df['oper_time'], format='mixed')
    df['date'] = df['oper_time'].dt.date
    
    result = df.copy()
    
    # 使用apply替代循环
    def count_remaining(group):
        group = group.sort_values('oper_time')
        times = group['oper_time'].values
        return np.sum(times.reshape(-1, 1) <= times.reshape(1, -1), axis=1)
    
    # 获取分组信息并显示
    groups = df.groupby(['serv_number', 'date'])
    group_count = len(groups)
    print(f"总共需要处理 {group_count:,} 个用户日期组")
    
    # 使用tqdm包装groupby的apply操作
    tqdm.pandas(desc="处理进度", ncols=100)
    result['label'] = groups.progress_apply(
        lambda x: pd.Series(count_remaining(x), index=x.index)
    ).values
    
    # 计算总耗时
    end_time = datetime.now()
    process_time = end_time - start_time
    print(f"\n处理完成！总耗时: {process_time}")
    
    return result

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    data = [
        {"statis_date": "20241106", "serv_number": "13487028369", "video_id": "45211", "env_video_time": "38.0", "oper_time": "2024-11-06 00:32:29.130"},#2
        {"statis_date": "20241106", "serv_number": "13487028369", "video_id": "45211", "env_video_time": "38.0", "oper_time": "2024-11-06 00:32:30.130"},#2       
        {"statis_date": "20241106", "serv_number": "13457692971", "video_id": "44963", "env_video_time": "38.0", "oper_time": "2024-11-06 00:32:37"},#1
        {"statis_date": "20241106", "serv_number": "13487028369", "video_id": "48948", "env_video_time": "100.0", "oper_time": "2024-11-06 00:47:47"},#1
        {"statis_date": "20241107", "serv_number": "13548466410", "video_id": "48082", "env_video_time": "72.0", "oper_time": "2024-11-07 00:23:46.269"},#3
        {"statis_date": "20241107", "serv_number": "13548466410", "video_id": "44963", "env_video_time": "0.0", "oper_time": "2024-11-07 00:24:58"},#2
        {"statis_date": "20241107", "serv_number": "13548466410", "video_id": "44964", "env_video_time": "0.0", "oper_time": "2024-11-07 00:24:59"}#1
    ]
    
    df = pd.DataFrame(data)
    
    # 生成标签
    labeled_df = generate_watch_count_labels_vectorized(df)
    
    print("\n处理结果示例:")
    print(labeled_df)