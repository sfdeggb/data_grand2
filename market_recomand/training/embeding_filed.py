import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import jieba

def embedding_text_columns(text: str, word2vec_model, max_length: int = 100) -> np.ndarray:
    """
    将文本转换为词向量序列
    """
    # 分词
    words = jieba.lcut(text)
    # 截断或补齐到指定长度
    words = words[:max_length] + ['PAD'] * (max_length - len(words)) if len(words) > max_length \
        else words + ['PAD'] * (max_length - len(words))
    # 转换为词向量
    word_vectors = np.array([word2vec_model.wv[word] if word in word2vec_model.wv \
                             else word2vec_model.wv['PAD'] for word in words])
    # 返回平均向量
    return np.mean(word_vectors, axis=0)

def embedding_categorical_columns(df: pd.DataFrame, columns: list[str], text_columns: list[str] = None, embedding_dim: int = 8) -> pd.DataFrame:
    """
    将分类列和文本列转换为嵌入向量
    
    参数:
        df: 输入的DataFrame
        columns: 需要进行分类嵌入的列名列表
        text_columns: 需要进行文本嵌入的列名列表
        embedding_dim: 分类嵌入向量的维度，默认为8
    """
    result_df = df.copy()
    
    # 处理分类列
    for col in columns:
        if col not in df.columns:
            continue
            
        # 使用LabelEncoder将分类值转换为数字
        le = LabelEncoder()
        values = le.fit_transform(df[col].values)
        
        # 获取唯一类别数
        n_categories = len(le.classes_)
        
        # 创建随机嵌入矩阵
        np.random.seed(42)  # 设置随机种子以保证结果可复现
        embedding_matrix = np.random.normal(0, 1, (n_categories, embedding_dim))
        
        # 将原始列转换为嵌入向量
        embedded_values = embedding_matrix[values]
        
        # 为嵌入向量的每个维度创建新列
        for i in range(embedding_dim):
            result_df[f'{col}_embed_{i}'] = embedded_values[:, i]
            
        # 删除原始列
        result_df = result_df.drop(columns=[col])
        
    # 处理文本列
    if text_columns:
        # 训练 Word2Vec 模型
        texts = []
        for col in text_columns:
            if col not in df.columns:
                continue
            texts.extend([jieba.lcut(text) for text in df[col].fillna('')])
        
        word2vec_model = Word2Vec(texts, vector_size=embedding_dim, window=5, min_count=1, workers=4)
        word2vec_model.wv['PAD'] = np.zeros(embedding_dim)  # 添加填充向量
        
        # 转换文本列
        for col in text_columns:
            if col not in df.columns:
                continue
                
            # 将每个文本转换为向量
            text_vectors = df[col].fillna('').apply(lambda x: embedding_text_columns(x, word2vec_model))
            
            # 为文本向量的每个维度创建新列
            for i in range(embedding_dim):
                result_df[f'{col}_embed_{i}'] = text_vectors.apply(lambda x: x[i])
                
            # 删除原始列
            result_df = result_df.drop(columns=[col])
    
    return result_df

# 使用示例:
if __name__ == "__main__":
    # 创建示例数据
    df = pd.DataFrame({
        'product_category': ['电子产品', '服装', '食品', '电子产品', '服装'],
        'brand': ['品牌A', '品牌B', '品牌C', '品牌A', '品牌B'],
        'description': ['这是一款高性能手机', '时尚休闲服装', '美味零食小吃', '智能平板电脑', '舒适运动服'],
        'price': [100, 200, 50, 150, 250]
    })
    
    # 指定需要嵌入的列
    categorical_columns = ['product_category', 'brand']
    text_columns = ['description']
    
    # 进行嵌入转换
    result = embedding_categorical_columns(df, categorical_columns, text_columns, embedding_dim=4)
    print(result)
