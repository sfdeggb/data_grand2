from itertools import combinations
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from types import MethodType

""" 使用swing来构建共现矩阵 
改进点： 使用numpy来加快计算
		相似度计算规则自定义 可以通过设置共有用户数来限制大小
		使用生成表达式来减少内存使用并提到性能
		使用多进程来提高并行计算的能力
"""
alpha = 0.5
top_k = 10
sim_gateway=1
parallel=4

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap"""
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))
    
    task_batches = Pool._get_tasks(func, iterable, chunksize)
    result = Pool._IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                        task_batches),
            result._set_length
        ))
    return result

# Modified Pool class implementation
Pool.istarmap = istarmap

def load_data(train_path):
    train_data = pd.read_csv(train_path, sep="\t", engine="python", names=["userid", "itemid"], 
                             dtype={"userid": int, "itemid": int})#提取用户交互记录数据并设置数据类型为整数
    print(train_data.head(3))
    return train_data.to_numpy()  # 转换为NumPy数组
    
def get_uitems_iusers(train):
    u_items = dict()
    i_users = dict()
    
    # 使用NumPy数组进行处理
    user_ids = train[:, 0]  # 第一列为userid
    item_ids = train[:, 1]  # 第二列为itemid
    
    for user_id, item_id in zip(user_ids, item_ids):  # 使用zip直接遍历
        u_items.setdefault(user_id, set())
        i_users.setdefault(item_id, set())
        u_items[user_id].add(item_id)  # 得到user交互过的所有item
        i_users[item_id].add(user_id)  # 得到item交互过的所有user
    
    print("使用的用户个数为：{}".format(len(u_items)))
    print("使用的item个数为：{}".format(len(i_users)))
    return u_items, i_users 

def calculate_similarity(pair, u_items, i_users, alpha, sim_gateway):
    i, j = pair
    common_users = np.intersect1d(np.array(list(i_users[i])), np.array(list(i_users[j])))  # 使用NumPy的交集
    if len(common_users) < sim_gateway:  # 如果共同用户少于sim_gateway，跳过
        return None
    
    user_pairs = combinations(common_users, 2)  # item_i和item_j对应的user取交集后全排列 得到user对
    result = sum(1 / (alpha + len(np.intersect1d(u_items[u], u_items[v]))) for (u, v) in user_pairs)  # 使用NumPy的交集
    
    if result != 0:
        return (i, j, format(result, '.6f'))
    return None

def swing_model(u_items, i_users, sim_gateway=2, parallel=4):
    item_ids = list(i_users.keys())
    item_pairs = list(combinations(item_ids, 2))
    total_pairs = len(item_pairs)
    print("item pairs length：{}".format(total_pairs))
    
    with Pool(processes=parallel) as pool:  # Use Pool instead of MyPool
        results = list(tqdm(
            pool.istarmap(calculate_similarity, 
                         [(pair, u_items, i_users, alpha, sim_gateway) for pair in item_pairs],
                         chunksize=1000),
            total=total_pairs,
            desc="计算物品相似度"
        ))
    
    item_sim_dict = {}
    for result in results:
        if result is not None:
            i, j, similarity = result
            item_sim_dict.setdefault(i, dict())
            item_sim_dict[i][j] = similarity
            
    return item_sim_dict

def save_item_sims(item_sim_dict, top_k, path):
    new_item_sim_dict = dict()
    try:
        writer = open(path, 'w', encoding='utf-8')
        for item, sim_items in item_sim_dict.items():
            new_item_sim_dict.setdefault(item, dict())
            new_item_sim_dict[item] = dict(sorted(sim_items.items(), key = lambda k:k[1], reverse=True)[:top_k])#排序取出 top_k个相似的item
            writer.write('item_id:%d\t%s\n' % (item, new_item_sim_dict[item]))
        print("SUCCESS: top_{} item saved".format(top_k))
    except Exception as e:
        print(e.args)

if __name__ == "__main__":
    train_data_path = "./dataset/swing_data.csv"
    item_sim_save_path = "./item_sim_dict.txt"
    top_k = 10 #与item相似的前 k 个item
    sim_gateway=1
    parallel=4
    train = load_data(train_data_path)
    u_items, i_users = get_uitems_iusers(train)
    item_sim_dict = swing_model(u_items, i_users,sim_gateway,parallel)
    save_item_sims(item_sim_dict, top_k, item_sim_save_path)
