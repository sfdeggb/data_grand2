import pandas as pd


def get_cross_feature(df, user_id='mobile', item_id='skuid', sequence_type='view', cycle=7):
    """计算用户-商品交叉特征
    
    Args:
        df (pd.DataFrame): 输入数据集
        user_id (str): 用户ID列名
        item_id (str): 商品ID列名 
        sequence_type (str): 行为类型 - 'view'/'click'/'purchase'
        cycle (int): 统计周期 - 7/14/30/60天
    """
    # 获取对应的行为序列列名
    seq_col_map = {
        'view': 'user_view_seq',
        'click': 'user_clk_seq', 
        'purchase': 'user_purchase_seq'
    }
    sequence_col = seq_col_map[sequence_type]
    
    def process_sequence(row, days):
        """处理单条记录的行为序列"""
        current_item = row[item_id]
        sequences = eval(row[sequence_col]) # 将字符串转为列表
        
        # 获取当前时间戳
        #current_time = pd.Timestamp.now()
        current_time = pd.Timestamp('20241205')  # 测试用 这里的日期要从实际数据中获取 row['static_date']
        # 过滤指定天数内的记录
        filtered_seq = [
            s for s in sequences 
            if (current_time - pd.Timestamp(s['oper_time'])).days <= days
        ]
        
        # 计算商品ID维度统计
        item_count = sum(1 for s in filtered_seq if s['sku_id'] == current_item)
        
        # 计算一级类目维度统计
        type1_count = sum(1 for s in filtered_seq 
                         if s['frist_order_type'] == row['goods_class_name'])
        
        # 计算二级类目维度统计
        type2_count = sum(1 for s in filtered_seq 
                         if s['second_order_type'] == row['class_name'])
        
        return pd.Series({
            f'u2i_{days}days_{sequence_type}_count': item_count,
            f'u2i_type1_{days}days_{sequence_type}_count': type1_count,
            f'u2i_type2_{days}days_{sequence_type}_count': type2_count
        })
    
    # 计算不同时间窗口的特征
    result_df = df.copy()
    
    # 对于点击行为额外计算1天的实时特征
    if sequence_type == 'click':
        result_df = pd.concat([
            result_df,
            df.apply(lambda x: process_sequence(x, 1), axis=1)
        ], axis=1)
    
    # 计算常规时间窗口的特征
    for days in [7, 14, 30, 60]:
        result_df = pd.concat([
            result_df,
            df.apply(lambda x: process_sequence(x, days), axis=1)
        ], axis=1)
    
    return result_df

# # 样例数据准备
# 生成10条样例数据
data = pd.DataFrame({
    # 商品特征
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
    'user_view_seq': [
        '[{"sku_id":"1001","oper_time":"20241201","is_add_buy":"1","frist_order_type":"食品","second_order_type":"速溶咖啡"}]',
        '[{"sku_id":"1002","oper_time":"20241201","is_add_buy":"0","frist_order_type":"饮料","second_order_type":"果汁"}]',
        '[{"sku_id":"1003","oper_time":"20241202","is_add_buy":"1","frist_order_type":"零食","second_order_type":"薯片"}]',
        '[{"sku_id":"1004","oper_time":"20241202","is_add_buy":"0","frist_order_type":"食品","second_order_type":"豆腐干"}]',
        '[{"sku_id":"1005","oper_time":"20241203","is_add_buy":"1","frist_order_type":"饮料","second_order_type":"茶饮"}]',
        '[{"sku_id":"1006","oper_time":"20241203","is_add_buy":"0","frist_order_type":"零食","second_order_type":"糖果"}]',
        '[{"sku_id":"1007","oper_time":"20241204","is_add_buy":"1","frist_order_type":"食品","second_order_type":"方便面"}]',
        '[{"sku_id":"1008","oper_time":"20241204","is_add_buy":"0","frist_order_type":"饮料","second_order_type":"碳酸饮料"}]',
        '[{"sku_id":"1009","oper_time":"20241205","is_add_buy":"1","frist_order_type":"零食","second_order_type":"饼干"}]',
        '[{"sku_id":"1010","oper_time":"20241205","is_add_buy":"0","frist_order_type":"食品","second_order_type":"肉脯"}]'
    ],
    
    'user_clk_seq': [
        '[{"sku_id":"1001","oper_time":"20241201","is_add_buy":"1","frist_order_type":"食品","second_order_type":"速溶咖啡"}]',
        '[{"sku_id":"1002","oper_time":"20241201","is_add_buy":"0","frist_order_type":"饮料","second_order_type":"果汁"}]',
        '[{"sku_id":"1003","oper_time":"20241202","is_add_buy":"1","frist_order_type":"零食","second_order_type":"薯片"}]',
        '[{"sku_id":"1004","oper_time":"20241202","is_add_buy":"0","frist_order_type":"食品","second_order_type":"豆腐干"}]',
        '[{"sku_id":"1005","oper_time":"20241203","is_add_buy":"1","frist_order_type":"饮料","second_order_type":"茶饮"}]',
        '[{"sku_id":"1006","oper_time":"20241203","is_add_buy":"0","frist_order_type":"零食","second_order_type":"糖果"}]',
        '[{"sku_id":"1007","oper_time":"20241204","is_add_buy":"1","frist_order_type":"食品","second_order_type":"方便面"}]',
        '[{"sku_id":"1008","oper_time":"20241204","is_add_buy":"0","frist_order_type":"饮料","second_order_type":"碳酸饮料"}]',
        '[{"sku_id":"1009","oper_time":"20241205","is_add_buy":"1","frist_order_type":"零食","second_order_type":"饼干"}]',
        '[{"sku_id":"1010","oper_time":"20241205","is_add_buy":"0","frist_order_type":"食品","second_order_type":"肉脯"}]'
    ],
    
    'user_purchase_seq': [
        '[{"sku_id":"1001","oper_time":"20241201","is_add_buy":"1","frist_order_type":"食品","second_order_type":"速溶咖啡"}]',
        '[{"sku_id":"1002","oper_time":"20241201","is_add_buy":"0","frist_order_type":"饮料","second_order_type":"果汁"}]',
        '[{"sku_id":"1003","oper_time":"20241202","is_add_buy":"1","frist_order_type":"零食","second_order_type":"薯片"}]',
        '[{"sku_id":"1004","oper_time":"20241202","is_add_buy":"0","frist_order_type":"食品","second_order_type":"豆腐干"}]',
        '[{"sku_id":"1005","oper_time":"20241203","is_add_buy":"1","frist_order_type":"饮料","second_order_type":"茶饮"}]',
        '[{"sku_id":"1006","oper_time":"20241203","is_add_buy":"0","frist_order_type":"零食","second_order_type":"糖果"}]',
        '[{"sku_id":"1007","oper_time":"20241204","is_add_buy":"1","frist_order_type":"食品","second_order_type":"方便面"}]',
        '[{"sku_id":"1008","oper_time":"20241204","is_add_buy":"0","frist_order_type":"饮料","second_order_type":"碳酸饮料"}]',
        '[{"sku_id":"1009","oper_time":"20241205","is_add_buy":"1","frist_order_type":"零食","second_order_type":"饼干"}]',
        '[{"sku_id":"1010","oper_time":"20241205","is_add_buy":"0","frist_order_type":"食品","second_order_type":"肉脯"}]'
    ]
})

df = get_cross_feature(data, sequence_type='view')
print(df.shape)
df.to_csv('dbdt_debug.csv', index=False)
