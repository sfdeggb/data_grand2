import jieba
import jieba.analyse
import pandas as pd

def extract_keywords(df, column_name='extract_text', top_k=20):
    """
    使用TextRank算法从DataFrame的指定列中提取关键词
    
    参数:
    df: pandas DataFrame
    column_name: 包含文本的列名
    top_k: 每条文本提取的关键词数量
    
    返回:
    带有关键词列的DataFrame
    """
    
    # 确保输入的列存在
    if column_name not in df.columns:
        raise ValueError(f"列 '{column_name}' 不存在于DataFrame中")
    
    # 初始化一个列表来存储每条文本的关键词
    all_keywords = []
    
    # 对每条文本进行关键词提取
    for text in df[column_name]:
        if pd.isna(text):  # 处理空值
            all_keywords.append([])
            continue
            
        # 使用TextRank算法提取关键词
        keywords = jieba.analyse.textrank(
            text,
            topK=top_k,  # 提取前k个关键词
            withWeight=True,  # 返回每个关键词的权重
            allowPOS=('n', 'nr', 'ns', 'nt', 'nw', 'vn', 'v')  # 允许的词性
        )
        
        # 将(关键词,权重)对转换为列表
        keywords_list = [(word, weight) for word, weight in keywords]
        all_keywords.append(keywords_list)
    
    # 将关键词添加到DataFrame中
    df['keywords'] = all_keywords
    
    return df

# 使用示例:
# df = pd.DataFrame({'extract_text': ['这是一段示例文本，用于测试关键词提取算法。', '另一段示例文本']})
# result_df = extract_keywords(df)