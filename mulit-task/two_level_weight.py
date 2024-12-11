def objective_function(duration, probability, max_duration=3600):
    """
    优化目标函数
    
    参数:
    duration: 停留时长
    probability: 概率值 (应在0.5-1之间)
    max_duration: 最大停留时长，用于归一化 (默认1小时=3600秒)
    
    返回:
    归一化后的得分 (0-1之间)
    """
    if probability < 0.5 or probability > 1:
        return 0  # 如果概率值不在有效范围内，返回0分
    
    # 停留时长归一化到0-1之间
    duration_normalized = min(duration / max_duration, 1.0)
    
    # 概率值评分：将((p-0.5)**2)归一化到0-1之间
    # 当p=0.5时得分为1，p=1时得分为0
    probability_score = 1 - (4 * (probability - 0.5) ** 2)
    
    # 将两部分组合
    w1 = 0.5  # 停留时长的权重
    w2 = 0.5  # 概率值的权重
    
    final_score = w1 * duration_normalized + w2 * probability_score
    
    return final_score