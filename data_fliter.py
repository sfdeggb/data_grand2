import json
import os
from typing import List, Dict, Any, Union
from datetime import datetime

def write_cold_result(cold_s_type: str, cold_res: List[Any], prod_code: Union[str, List[str]], actv_id: Union[str, None] = None) -> bool:
    """
    根据cold_s_type将prod_code和cold_res的对应关系写入不同的文件
    
    Args:
        cold_s_type (str): 结果类型，必须是 'type1', 'type2', 或 'type3'
        cold_res (List[Any]): 需要写入的结果数据列表
        prod_code (Union[str, List[str]]): 产品代码或产品代码列表
        actv_id (Union[str, None]): 活动ID或活动ID列表（仅适用于type1）
    
    Returns:
        bool: 写入是否成功
    """
    try:
        # 验证cold_s_type
        if cold_s_type not in ['type1', 'type2', 'type3']:
            raise ValueError(f"Invalid cold_s_type: {cold_s_type}")
        
        # 将prod_code转换为字符串key
        prod_code_key = ','.join(prod_code) if isinstance(prod_code, list) else str(prod_code)
        
        # 创建输出目录
        output_dir = "cold_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建文件路径
        filename = os.path.join(output_dir, f"{cold_s_type}.json")
        
        # 读取现有数据（如果存在）
        existing_data = {}
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        
        # 添加新数据
        existing_data[prod_code_key] = cold_res
        
        # 写入更新后的数据
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully wrote {cold_s_type} result for prod_code {prod_code_key}")
        return True
        
    except Exception as e:
        print(f"Error writing {cold_s_type} result for prod_code {prod_code}: {e}")
        return False

def merge_cold_results() -> bool:
    """
    合并所有类型的cold结果文件为一个文件
    - type1的重复判断基于prod_code和actv_id的组合
    - type2和type3的重复判断仅基于prod_code
    
    Returns:
        bool: 合并是否成功
    """
    try:
        output_dir = ROOT
        merged_data = []
        merged_filename = os.path.join(output_dir, "coldStartFliter.json")
        
        # 如果合并文件已存在，先读取现有数据
        if os.path.exists(merged_filename):
            with open(merged_filename, 'r', encoding='utf-8') as f:
                merged_data = json.load(f)
        
        # 创建用于判重的集合
        # 对于type1，使用(prod_code, actv_id)元组
        # 对于type2和type3，仅使用prod_code
        existing_keys = set()
        for item in merged_data:
            if 'actv_id' in item:  # type1格式的数据
                existing_keys.add((item['prod_code'], item['actv_id']))
            else:  # type2或type3格式的数据
                existing_keys.add(item['prod_code'])
        
        # 读取所有类型的文件
        for cold_s_type in ['0_1_actv', '0_2_rule', '0_3_name']:
            filename = os.path.join(output_dir, f"{cold_s_type}.json")
            if not os.path.exists(filename):
                print(f"Warning: {filename} does not exist")
                continue
                
            with open(filename, 'r', encoding='utf-8') as f:
                new_data = json.load(f)
                
                # 根据不同类型处理数据
                for item in new_data:
                    if cold_s_type == '0_1_actv':
                        # type1需要检查prod_code和actv_id组合
                        key = (item['prod_code'], item['actv_id'])
                        if key not in existing_keys:
                            merged_data.append(item)
                            existing_keys.add(key)
                    else:
                        # type2和type3只检查prod_code
                        if item['prod_code'] not in existing_keys:
                            merged_data.append(item)
                            existing_keys.add(item['prod_code'])
        
        # 写入合并后的数据
        with open(merged_filename, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully merged results to {merged_filename}")
        print(f"Total records after merge: {len(merged_data)}")
        return True
        
    except Exception as e:
        print(f"Error merging results: {e}")
        return False

# 使用示例
def main():
    # 示例数据 - 包含列表类型的prod_code
    test_data = [
        ("type1", ["data1", "data2"], ["49030611", "89029945"], "126336"),
        ("type2", ["data3", "data4"], ["49030611", "89029945"]),
        ("type3", ["data5", "data6"], ["49030611"]),
    ]
    
    # 写入各个文件
    for cold_s_type, cold_res, prod_code, actv_id in test_data:
        write_cold_result(cold_s_type, cold_res, prod_code, actv_id)
    
    # 合并文件
    merge_cold_results()

if __name__ == "__main__":
    main()