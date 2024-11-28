import numpy as np

def pegasos(X, y, lambda_=0.01, max_iter=1000, r=1e-3):
    """
    Pegasos SVM训练函数
    :param X: 二维NumPy数组，每行是一个样本，每列是一个特征
    :param y: 一维NumPy数组，样本的标签（+1或-1）
    :param lambda_: 正则化参数
    :param max_iter: 最大迭代次数
    :param r: 收敛阈值
    :return: 训练好的权重向量w
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # 初始化权重向量
    bias = 0  # 初始化偏置项
    eta = 1.0 / (lambda_ * n_samples)  # 学习率

    for t in range(max_iter):
        for i in range(n_samples):
            # 随机选择一个样本
            xi = X[i]
            yi = y[i]
            
            # 计算当前样本的预测值
            prediction = np.dot(w, xi) + bias
            
            # 计算合页损失
            hinge_loss = max(0, 1 - yi * prediction)
            
            # 如果样本被误分类或损失大于0，则更新权重和偏置
            if hinge_loss > 0:
                # 更新权重向量w
                w = (1 - lambda_ * eta) * w + eta * yi * xi
                # 更新偏置项
                bias += eta * yi
            else:
                # 如果样本被正确分类，则只更新权重向量的正则化部分
                w = (1 - lambda_ * eta) * w
        
        # 检查收敛性
        if t > 0 and np.linalg.norm(w - old_w, ord=1) < r:
            break
        old_w = w.copy()  # 保存旧的权重向量以检查收敛性

    return w, bias

# 示例数据
X = np.array([[2, 3], [3, 3], [4, 5], [1, 1]])
y = np.array([1, 1, -1, -1])

# 训练SVM
w, bias = pegasos(X, y)

print("训练得到的权重向量w:", w)
print("训练得到的偏置项bias:", bias)