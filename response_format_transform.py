import pandas as pd
import json
from typing import Dict, Any, List
import uuid

def create_instance_from_response(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    从response中的item创建实例数据
    """
    try:
        actv_id = item.get('actv_id', '')
        prod_code = str(item.get('prod_code', ''))
        score = item.get('score', 0)
        
        # 创建标准格式的id
        instance_id = f"220@{actv_id}@{prod_code}@0"
        
        # 将score转换为0-1之间的值
        normalized_score = score / 100.0 if score else 0
        
        return {
            "id": instance_id,
            "scores": [normalized_score]
        }
    except Exception as e:
        print(f"Error creating instance: {e}")
        return None

def transform_response_format(row: pd.Series) -> Dict[str, Any]:
    """
    转换单行数据为目标格式
    """
    try:
        # 解析response数据
        response_data = json.loads(row['model_response'])
        
        # 提取items数据
        items = response_data.get('data', {}).get('items', [])
        
        # 创建实例列表
        instances = []
        for item in items:
            instance = create_instance_from_response(item)
            if instance:
                instances.append(instance)
        
        # 创建完整的响应格式
        transformed_data = {
            "requestId": str(uuid.uuid4()),
            "status": "OK",
            "instances": instances
        }
        
        return transformed_data
    
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error processing row: {e}")
        return None

def process_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    处理整个DataFrame并返回转换后的数据列表
    """
    transformed_data = []
    error_records = []
    
    for idx, row in df.iterrows():
        try:
            result = transform_response_format(row)
            if result:
                transformed_data.append(result)
        except Exception as e:
            error_records.append({
                'index': idx,
                'error': str(e),
                'raw_data': row['model_response']
            })
    
    # 保存错误记录
    if error_records:
        error_df = pd.DataFrame(error_records)
        error_df.to_csv('response_transform_errors.csv', index=False)
        print(f"发现 {len(error_records)} 条错误记录，已保存到 response_transform_errors.csv")
    
    return transformed_data

def main():
    try:
        # 读取数据
        df = pd.read_csv('your_data.csv')
        
        # 转换数据
        transformed_data = process_dataframe(df)
        
        # 保存转换后的数据
        with open('transformed_response_data.json', 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=2)
        
        print(f"成功转换 {len(transformed_data)} 条数据，已保存到 transformed_response_data.json")
        
        # 显示第一条转换后的数据作为样例
        if transformed_data:
            print("\n转换后的数据样例:")
            print(json.dumps(transformed_data[0], indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"处理数据时发生错误: {e}")

if __name__ == "__main__":
    main() 