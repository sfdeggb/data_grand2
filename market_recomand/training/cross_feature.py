import pandas as pd
import numpy as np
from tqdm import tqdm

def get_cross_feature(df, user_id='mobile', item_id='skuid', sequence_type='view', cycle=7, batch_size=10000):
    """计算用户-商品交叉特征的优化版本
    
    Args:
        df: 输入数据集
        user_id: 用户ID列名
        item_id: 商品ID列名
        sequence_type: 行为类型 - 'view'/'click'/'purchase'
        cycle: 统计周期
        batch_size: 批处理大小
    """
    # 获取对应的行为序列列名
    seq_col_map = {
        'view': 'qysc_view_seq',
        'click': 'qysc_clk_seq', 
        'purchase': 'qysc_order_seq'
    }
    sequence_col = seq_col_map[sequence_type]
    
    def process_batch(batch_df):
        """处理单个批次的数据"""
        #current_time = pd.Timestamp('20241205')
        
        # 将序列字符串转换为列表（批量处理）
        sequences = batch_df[sequence_col].fillna('[]').apply(eval)
        
        # 预分配结果数组
        result_arrays = {
            f'u2i_{cycle}days_{sequence_type}_count': np.zeros(len(batch_df)),
            f'u2i_type1_{cycle}days_{sequence_type}_count': np.zeros(len(batch_df)),
            f'u2i_type2_{cycle}days_{sequence_type}_count': np.zeros(len(batch_df))
        }
        
        # 向量化处理序列
        for idx, (seq, current_time, current_item, type1, type2) in enumerate(zip(
            sequences, 
            batch_df['statis_date'],
            batch_df[item_id],
            batch_df['goods_class_name'],
            batch_df['class_name']
        )):
            current_time = pd.Timestamp(current_time)
            try:
                # 过滤时间范围内的记录
                filtered_seq = [
                    s for s in seq 
                    if (current_time - pd.Timestamp(s['oper_time'])).days <= cycle
                ]
            except Exception as e:
                print(f"Error processing sequence for row {idx}: {e}")
                filtered_seq = []
            if filtered_seq:
                # 使用numpy数组操作进行统计
                seq_array = np.array([
                    (s['sku_id'], s['frist_class_name'], s['second_class_name'])
                    for s in filtered_seq
                ], dtype=object)
                
                result_arrays[f'u2i_{cycle}days_{sequence_type}_count'][idx] = np.sum(seq_array[:, 0] == current_item)
                result_arrays[f'u2i_type1_{cycle}days_{sequence_type}_count'][idx] = np.sum(seq_array[:, 1] == type1)
                result_arrays[f'u2i_type2_{cycle}days_{sequence_type}_count'][idx] = np.sum(seq_array[:, 2] == type2)
        
        return pd.DataFrame(result_arrays)
    
    # 批量处理数据
    result_dfs = []
    for start_idx in tqdm(range(0, len(df), batch_size), desc=f"Processing {sequence_type} features"):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        result_df = process_batch(batch_df)
        result_dfs.append(result_df)
    
    # 合并结果
    final_result = pd.concat(result_dfs, axis=0)
    
    final_result.index = df.index
    return pd.concat([df, final_result], axis=1)

# 主函数也需要相应修改
def process_all_features(df, batch_size=10000):
    """处理所有类型的交叉特征"""
    result_df = df.copy()
    
    # 使用tqdm显示总体进度
    with tqdm(total=1, desc="Processing batches") as pbar:
        # 处理各类行为特征
        for sequence_type in ['view', 'click', 'purchase']:
            result_df = get_cross_feature(
                result_df, 
                sequence_type=sequence_type,
                batch_size=batch_size
            )
        pbar.update(1)
    
    return result_df

def generate_sample_data(): 
    # 生成10条样例数据
    data2 = pd.DataFrame({
        # 商品特征
    'statis_date': ['20241201', '20241201', '20241202', '20241202', '20241203', '20241203', '20241204', '20241204', '20241205', '20241205'],
    'skuid': ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010'],
    'class_name': ['食品', '饮料', '零食', '食品', '饮料', '零食', '食品', '饮料', '零食', '食品'],
    'c_price': [15.9, 25.5, 12.8, 18.5, 22.0, 9.9, 16.5, 28.0, 11.5, 19.9],
    'line_price': [19.9, 29.9, 15.9, 22.9, 25.9, 12.9, 19.9, 32.9, 14.9, 23.9],
    'goods_class_id': ['A001', 'B001', 'C001', 'A002', 'B002', 'C002', 'A003', 'B003', 'C003', 'A004'],
    'goods_class_name': ['速溶咖啡', '果汁', '薯片', '豆腐干', '茶饮', '糖果', '方便面', '碳酸饮料', '饼干', '肉脯'],
    
    # 商品统计特征
    'goods_7day_views': [120, 85, 95, 150, 110, 75, 130, 90, 100, 140],
    'goods_14day_views': [250, 180, 200, 310, 230, 160, 270, 190, 210, 290],
    'goods_30day_views': [520, 380, 420, 650, 480, 340, 560, 400, 440, 600],
    'goods_7day_sales': [45, 30, 35, 55, 40, 25, 48, 32, 38, 50],
    'goods_14day_sales': [95, 65, 75, 115, 85, 55, 100, 70, 80, 105],
    'goods_30day_sales': [200, 140, 160, 240, 180, 120, 210, 150, 170, 220],
    
    # 用户特征
    'mobile': ['user001', 'user002', 'user003', 'user004', 'user005', 'user006', 'user007', 'user008', 'user009', 'user010'],
    'prov_code': ['110000', '310000', '440000', '330000', '510000', '420000', '320000', '370000', '610000', '500000'],
    
    # 用户行为序列（示例）
    'qysc_view_seq': [
        '[{"sku_id":"1001","oper_time":"20241201","is_add_buy":"0","frist_class_name":"饮料","second_class_name":"果汁"}]',
        '[{"sku_id":"1002","oper_time":"20241201","is_add_buy":"0","frist_class_name":"饮料","second_class_name":"果汁"}]',
        '[{"sku_id":"1003","oper_time":"20241202","is_add_buy":"1","frist_class_name":"零食","second_class_name":"薯片"}]',
        '[{"sku_id":"1004","oper_time":"20241202","is_add_buy":"0","frist_class_name":"食品","second_class_name":"豆腐干"}]',
        '[{"sku_id":"1005","oper_time":"20241203","is_add_buy":"1","frist_class_name":"饮料","second_class_name":"茶饮"}]',
        '[{"sku_id":"1006","oper_time":"20241203","is_add_buy":"0","frist_class_name":"零食","second_class_name":"糖果"}]',
        '[{"sku_id":"1007","oper_time":"20241204","is_add_buy":"1","frist_class_name":"食品","second_class_name":"方便面"}]',
        '[{"sku_id":"1008","oper_time":"20241204","is_add_buy":"0","frist_class_name":"饮料","second_class_name":"碳酸饮料"}]',
        '[{"sku_id":"1009","oper_time":"20241205","is_add_buy":"1","frist_class_name":"零食","second_class_name":"饼干"}]',
        '[{"sku_id":"1010","oper_time":"20241205","is_add_buy":"0","frist_class_name":"食品","second_class_name":"肉脯"}]'
    ],
    
    'qysc_clk_seq': [
        '[{"sku_id":"1001","oper_time":"20241201","is_add_buy":"1","frist_class_name":"食品","second_class_name":"速溶咖啡"}]',
        '[{"sku_id":"1002","oper_time":"20241201","is_add_buy":"0","frist_class_name":"饮料","second_class_name":"果汁"}]',
        '[{"sku_id":"1003","oper_time":"20241202","is_add_buy":"1","frist_class_name":"零食","second_class_name":"薯片"}]',
        '[{"sku_id":"1004","oper_time":"20241202","is_add_buy":"0","frist_class_name":"食品","second_class_name":"豆腐干"}]',
        '[{"sku_id":"1005","oper_time":"20241203","is_add_buy":"1","frist_class_name":"饮料","second_class_name":"茶饮"}]',
        '[{"sku_id":"1006","oper_time":"20241203","is_add_buy":"0","frist_class_name":"零食","second_class_name":"糖果"}]',
        '[{"sku_id":"1007","oper_time":"20241204","is_add_buy":"1","frist_class_name":"食品","second_class_name":"方便面"}]',
        '[{"sku_id":"1008","oper_time":"20241204","is_add_buy":"0","frist_class_name":"饮料","second_class_name":"碳酸饮料"}]',
        '[{"sku_id":"1009","oper_time":"20241205","is_add_buy":"1","frist_class_name":"零食","second_class_name":"饼干"}]',
        '[{"sku_id":"1010","oper_time":"20241205","is_add_buy":"0","frist_class_name":"食品","second_class_name":"肉脯"}]'
    ],
    
    'qysc_order_seq': [
        '[{"sku_id":"1001","oper_time":"20241201","is_add_buy":"1","frist_class_name":"食品","second_class_name":"速溶咖啡"}]',
        '[{"sku_id":"1002","oper_time":"20241201","is_add_buy":"0","frist_class_name":"饮料","second_class_name":"果汁"}]',
        '[{"sku_id":"1003","oper_time":"20241202","is_add_buy":"1","frist_class_name":"零食","second_class_name":"薯片"}]',
        '[{"sku_id":"1004","oper_time":"20241202","is_add_buy":"0","frist_class_name":"食品","second_class_name":"豆腐干"}]',
        '[{"sku_id":"1005","oper_time":"20241203","is_add_buy":"1","frist_class_name":"饮料","second_class_name":"茶饮"}]',
        '[{"sku_id":"1006","oper_time":"20241203","is_add_buy":"0","frist_class_name":"零食","second_class_name":"糖果"}]',
        '[{"sku_id":"1007","oper_time":"20241204","is_add_buy":"1","frist_class_name":"食品","second_class_name":"方便面"}]',
        '[{"sku_id":"1008","oper_time":"20241204","is_add_buy":"0","frist_class_name":"饮料","second_class_name":"碳酸饮料"}]',
        '[{"sku_id":"1009","oper_time":"20241205","is_add_buy":"1","frist_class_name":"零食","second_class_name":"饼干"}]',
        '[{"sku_id":"1010","oper_time":"20241205","is_add_buy":"0","frist_class_name":"食品","second_class_name":"肉脯"}]'
    ]
})

    print("样例数据生成完成，数据形状:", data2.shape)
    return data2

if __name__ == "__main__":
    data2 = generate_sample_data()
    data2 = process_all_features(data2)
    print(data2.head())