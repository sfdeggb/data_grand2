import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def prepare_features(df):
    """
    特征工程函数
    """
    df = df.copy()
    
    # 时间特征
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # 用户特征
    user_watch_counts = df.groupby('user_id').size().reset_index(name='user_historical_counts')
    df = df.merge(user_watch_counts, on='user_id', how='left')
    
    # 视频特征
    video_watch_counts = df.groupby('video_id').size().reset_index(name='video_popularity')
    df = df.merge(video_watch_counts, on='video_id', how='left')
    
    # 对类别特征进行编码
    le = LabelEncoder()
    df['user_id_encoded'] = le.fit_transform(df['user_id'])
    df['video_id_encoded'] = le.fit_transform(df['video_id'])
    
    return df

def train_gbdt_model(df):
    """
    训练GBDT模型
    """
    # 准备特征
    feature_cols = [
        'hour', 'minute', 'dayofweek',
        'user_id_encoded', 'video_id_encoded',
        'user_historical_counts', 'video_popularity'
    ]
    
    X = df[feature_cols]
    y = df['watch_count']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 设置模型参数
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # 预测和评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    
    # 特征重要性可视化
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=10)
    plt.title("特征重要性")
    plt.show()
    
    return model

if __name__ == "__main__":
    # 创建示例数据
    data = {
        'user_id': [1, 1, 1, 2, 2] * 20,  # 扩展数据集
        'video_id': [101, 102, 103, 201, 202] * 20,
        'timestamp': [
            '2024-12-24 15:30:42',
            '2024-12-24 16:20:00',
            '2024-12-25 10:00:00',
            '2024-12-24 09:00:00',
            '2024-12-24 18:00:00'
        ] * 20
    }
    df = pd.DataFrame(data)
    
    # 生成观看次数标签
    from sample_label import generate_watch_count_labels
    labeled_df = generate_watch_count_labels(df)
    
    # 特征工程
    featured_df = prepare_features(labeled_df)
    
    # 训练模型
    model = train_gbdt_model(featured_df)
