import pandas as pd
from datetime import datetime

def generate_watch_count_labels(df):
    """
    为每条记录生成用户从该时间点到当天结束的观看视频数量标签
    
    参数:
    df: DataFrame, 包含用户ID、时间戳等信息的数据框
    
    返回:
    带有观看次数标签的DataFrame
    """
    # 确保时间列为datetime类型并排序
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # 创建结果DataFrame
    result = df.copy()
    result['watch_count'] = 0
    
    # 按用户ID分组处理
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id]
        
        for idx, row in user_data.iterrows():
            current_time = row['timestamp']
            current_date = current_time.date()
            
            # 计算从当前时间到当天结束的观看次数
            count = len(user_data[
                (user_data['timestamp'] >= current_time) & 
                (user_data['timestamp'].dt.date == current_date)
            ])
            
            result.loc[idx, 'watch_count'] = count
    
    return result

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    data = {
        'user_id': [1, 1, 1, 2, 2],
        'video_id': [101, 102, 103, 201, 202],
        'timestamp': [
            '2024-12-24 15:30:42',
            '2024-12-24 16:20:00',
            '2024-12-25 10:00:00',
            '2024-12-24 09:00:00',
            '2024-12-24 18:00:00'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 生成标签
    labeled_df = generate_watch_count_labels(df)
    print("带标签的数据:")
    print(labeled_df)