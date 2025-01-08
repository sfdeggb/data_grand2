import numpy as np
from keras.preprocessing.sequence import pad_sequences


def merge_behavior_sequences_time_decay(browse_seq, click_seq, purchase_seq, 
                                      current_timestamp):
    """
    带时间衰减的序列合并
    """
    def calculate_time_weight(timestamp):
        time_diff = current_timestamp - timestamp
        return np.exp(-0.1 * time_diff)  # 指数衰减
    
    # 合并所有行为并添加行为类型和时间权重
    all_behaviors = []
    
    # 浏览行为
    for item, timestamp in browse_seq:
        time_weight = calculate_time_weight(timestamp)
        all_behaviors.append((item, 1, timestamp, time_weight))
    
    # 点击行为
    for item, timestamp in click_seq:
        time_weight = calculate_time_weight(timestamp)
        all_behaviors.append((item, 2, timestamp, time_weight * 2))  # 点击权重加倍
    
    # 购买行为
    for item, timestamp in purchase_seq:
        time_weight = calculate_time_weight(timestamp)
        all_behaviors.append((item, 3, timestamp, time_weight * 3))  # 购买权重三倍
    
    # 按时间戳排序
    all_behaviors.sort(key=lambda x: x[2])
    
    # 生成最终序列
    merged_seq = [(item * 10 + behavior_type) 
                 for item, behavior_type, _, _ in all_behaviors]
    
    # 截断到最大长度
    max_len = 100
    if len(merged_seq) > max_len:
        merged_seq = merged_seq[-max_len:]
    
    return merged_seq

# 使用示例
def prepare_data_for_din():
    # 示例数据
    sample_data = {
        'user_id': [1, 2, 3],
        'item_id': [1001, 1002, 1003],
        'browse_seq': [
            [101, 102, 103],
            [201, 202],
            [301, 302, 303, 304]
        ],
        'click_seq': [
            [501, 502],
            [601],
            [701, 702]
        ],
        'purchase_seq': [
            [901],
            [902, 903],
            [904]
        ],
        'label': [1, 0, 1]
    }
    
    # 合并行为序列
    merged_sequences = [
        merge_behavior_sequences_time_decay(browse, click, purchase)
        for browse, click, purchase in zip(
            sample_data['browse_seq'],
            sample_data['click_seq'],
            sample_data['purchase_seq']
        )
    ]
    
    # 构建模型输入
    model_input = {
        'user_id': np.array(sample_data['user_id']),
        'item_id': np.array(sample_data['item_id']),
        'hist_item_id': pad_sequences(merged_sequences, maxlen=100)
    }
    
    return model_input, np.array(sample_data['label'])

if __name__ == "__main__":
    prepare_data_for_din()
