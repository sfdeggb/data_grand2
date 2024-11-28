import pandas as pd
import json
from typing import Dict, Any, List
import uuid

def create_instance(actv_id: str, prod_code: str) -> Dict[str, Any]:
    """
    创建单个实例数据
    """
    instance_id = f"220@{actv_id}@{prod_code}@0"
    return {
        "id": instance_id,
        "rawFeatures": {
            "item_ID": instance_id,
            "actv_id": actv_id,
            "prod_code": prod_code,
            "product_type": 0.0,
            "product_price": 15800.0,
            "product_flow": 71680.0,
            "product_call_time": 1000.0,
            "product_discount_price": 1440.0
        }
    }

def transform_request_format(row: pd.Series) -> Dict[str, Any]:
    """
    转换单行数据为目标格式
    """
    try:
        # 解析request数据
        request_data = json.loads(row['model_request'])
        
        # 提取products数据
        request_dict = request_data.get('request', {})
        products = request_dict.get('proucts', [])
        
        # 创建实例列表
        instances = []
        for product in products:
            actv_id = product.get('actv_id', '')
            prod_code = str(product.get('prod_code', ''))
            if actv_id and prod_code:
                instances.append(create_instance(actv_id, prod_code))
        
        # 从request中提取commonFeatures
        # 移除不需要放入commonFeatures的字段
        common_features = request_dict.copy()
        common_features.pop('proucts', None)  # 移除products数组
        
        # 创建完整的请求格式
        transformed_data = {
            "requestId": str(uuid.uuid4()),
            "accessToken": "mib",
            "requestTime": 0,
            "resultLimit": 2,
            "commonFeatures": common_features,  # 使用从request中提取的特征
            "rawInstances": instances
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
            result = transform_request_format(row)
            if result:
                transformed_data.append(result)
        except Exception as e:
            error_records.append({
                'index': idx,
                'error': str(e),
                'raw_data': row['model_request']
            })
    
    # 保存错误记录
    if error_records:
        error_df = pd.DataFrame(error_records)
        error_df.to_csv('transform_errors.csv', index=False)
        print(f"发现 {len(error_records)} 条错误记录，已保存到 transform_errors.csv")
    
    return transformed_data

def main():
    try:
        # 读取数据
        df = pd.read_csv('your_data.csv')
        
        # 转换数据
        transformed_data = process_dataframe(df)
        
        # 保存转换后的数据
        with open('transformed_data.json', 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=2)
        
        print(f"成功转换 {len(transformed_data)} 条数据，已保存到 transformed_data.json")
        
    except Exception as e:
        print(f"处理数据时发生错误: {e}")

if __name__ == "__main__":
    main() 