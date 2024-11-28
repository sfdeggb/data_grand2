from itertools import combinations
import pandas as pd
alpha = 0.5
top_k = 10

"""
 swing 算法
 根据用户对物品的评分，计算物品之间的相似度
 进行推荐
"""
def load_data(train_path):
    train_data = pd.read_csv(train_path, sep="\t", engine="python", names=["userid", "itemid", "rate"])#提取用户交互记录数据
    print(train_data.head(3))
    return train_data

def get_uitems_iusers(train):
    u_items = dict()
    i_users = dict()
    for index, row in train.iterrows():#处理用户交互记录 
        u_items.setdefault(row["userid"], set())
        i_users.setdefault(row["itemid"], set())
        u_items[row["userid"]].add(row["itemid"])#得到user交互过的所有item
        i_users[row["itemid"]].add(row["userid"])#得到item交互过的所有user
    print("使用的用户个数为：{}".format(len(u_items)))
    print("使用的item个数为：{}".format(len(i_users)))
    return u_items, i_users 

def swing_model(u_items, i_users):
    """
    计算物品之间的相似度
    Args:
        u_items: dict, key为用户id, value为该用户交互过的物品集合
        i_users: dict, key为物品id, value为与该物品交互过的用户集合
    Returns:
        item_sim_dict: dict, key为物品id, value为与该物品相似的其他物品及其相似度
    """
    # 获取所有可能的物品对组合
    item_pairs = list(combinations(i_users.keys(), 2))  # 例如: [(1,2), (1,3), (2,3)...]
    print("item pairs length：{}".format(len(item_pairs)))
    
    # 存储物品相似度的字典
    item_sim_dict = dict()
    
    # 遍历每一个物品对
    for (i, j) in item_pairs:
        # 找到同时对物品i和j都有行为的用户对
        common_users = i_users[i] & i_users[j]  # 交互过物品i和j的用户交集
        user_pairs = list(combinations(common_users, 2))  # 这些用户的两两组合
        
        result = 0
        # 遍历每一个用户对
        for (u, v) in user_pairs:
            # 计算用户对(u,v)对物品对(i,j)的贡献度
            # 分母为: alpha + 用户u和v共同交互过的物品数量
            common_items_count = len(u_items[u] & u_items[v])
            result += 1 / (alpha + common_items_count)
            
        # 如果两个物品间存在相似度,则保存
        if result != 0:
            item_sim_dict.setdefault(i, dict())
            item_sim_dict[i][j] = format(result, '.6f')  # 保留6位小数
            
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
    train_data_path = "./ratings_final.txt"
    item_sim_save_path = "./item_sim_dict.txt"
    top_k = 10 #与item相似的前 k 个item
    train = load_data(train_data_path)
    u_items, i_users = get_uitems_iusers(train)
    item_sim_dict = swing_model(u_items, i_users)
    save_item_sims(item_sim_dict, top_k, item_sim_save_path)