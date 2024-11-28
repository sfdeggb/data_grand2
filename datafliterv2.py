import os
import json
from typing import List, Any

def write_cold_result(cold_res: List[Any], prod_code, actv_id='') -> bool:
    """
    冷启动快查结果写入

    Args:
    cold_res (List[Any]): 冷启动查找结果
    prod_code (str or List[str]): 产品id
    actv_id (str): 活动id，默认为空字符串

    Returns:
    bool: 写入是否成功
    """
    try:
        # 如果 prod_code 是列表，则确保列表中的所有元素都是字符串，并用逗号连接
        if isinstance(prod_code, list):
            prod_code_key = ','.join(str(code) for code in prod_code)
        else:
            prod_code_key = str(prod_code)

        # 创建输出目录
        output_dir = ROOT
        os.makedirs(output_dir, exist_ok=True)

        # 构建文件路径
        filename = os.path.join(output_dir, f"coldStartFliter.json")

        # 创建元组键并转换为字符串
        tuple_key = str((prod_code_key, actv_id))

        # 读取现有数据并转换键格式
        existing_data = {}
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                temp_data = json.load(f)
                # 确保所有键都是元组格式
                for k, v in temp_data.items():
                    # 将字符串形式的元组转换回真实的元组
                    clean_k = k.strip('()').split(', ')
                    existing_data[str((clean_k[0], clean_k[1].strip("'\"")))] = v

        # 更新数据
        existing_data[tuple_key] = cold_res

        # 写入更新后的数据
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        print(f"Successfully wrote result for key {tuple_key}")
        return True

    except Exception as e:
        print(f"Error writing result: {e}")
        return False