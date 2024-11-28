def plot_metrics_trend(df):
    """
    绘制指标随时间变化的趋势图
    
    参数:
    df : pandas.DataFrame
        包含以下列的数据框:
        - model_type: 模型类型
        - date: 日期
        - cover_rate: 覆盖率指标
        - mrr: MRR指标
        - top1: Top1指标
        - top3: Top3指标
        - top5: Top5指标
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # 设置图表风格
    plt.style.use('seaborn')
    sns.set_palette("husl")

    # 定义要绘制的指标
    metrics = ['cover_rate', 'mrr', 'top1', 'top3', 'top5']

    # 创建5个子图
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
    fig.tight_layout(pad=3.0)

    # 为每个指标创建一个图
    for i, metric in enumerate(metrics):
        # 获取所有不同的model_type
        for model in df['model_type'].unique():
            # 筛选当前model_type的数据
            model_data = df[df['model_type'] == model]
            
            # 将date转换为datetime类型以便正确排序
            model_data['date'] = pd.to_datetime(model_data['date'])
            model_data = model_data.sort_values('date')
            
            # 绘制线图
            axes[i].plot(model_data['date'], 
                        model_data[metric], 
                        marker='o', 
                        label=model)
            
            # 设置图表标题和标签
            axes[i].set_title(f'{metric} Trend Over Time')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel(metric)
            axes[i].legend()
            
            # 旋转x轴日期标签以防重叠
            axes[i].tick_params(axis='x', rotation=45)

    # 调整布局
    plt.tight_layout()
    plt.show()
