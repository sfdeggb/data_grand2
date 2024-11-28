import pandas as pd
import json
from typing import Set, Tuple
from datetime import datetime

def extract_prod_ids(row, error_records: list) -> Tuple[Set[str], Set[str]]:
    """
    从单行数据中提取request和response中的prod_id
    返回两个集合：request_ids 和 response_ids
    异常数据会被记录到error_records列表中
    """
    request_ids = set()
    response_ids = set()
    
    try:
        # 解析request中的prod_id
        request_data = json.loads(row['model_request'])
        if 'request' in request_data and 'proucts' in request_data['request']:
            for product in request_data['request']['proucts']:
                if 'prod_code' in product:
                    request_ids.add(str(product['prod_code']))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        error_records.append({
            'date': row['statis_date'],
            'error_type': 'request_error',
            'error_message': str(e),
            'raw_data': row['model_request']
        })
    
    try:
        # 解析response中的prod_id
        response_data = json.loads(row['model_response'])
        if 'data' in response_data and 'items' in response_data['data']:
            for item in response_data['data']['items']:
                if 'prod_id' in item:
                    response_ids.add(str(item['prod_id']))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        error_records.append({
            'date': row['statis_date'],
            'error_type': 'response_error',
            'error_message': str(e),
            'raw_data': row['model_response']
        })
    
    return request_ids, response_ids

def calculate_coverage_rate(df: pd.DataFrame) -> dict:
    """
    计算覆盖率
    返回包含每日覆盖率、总体覆盖率和错误记录的字典
    """
    daily_coverage = {}
    total_request_ids = set()
    total_response_ids = set()
    error_records = []
    
    # 按日期分组计算
    for date, group in df.groupby('statis_date'):
        date_request_ids = set()
        date_response_ids = set()
        
        for _, row in group.iterrows():
            req_ids, resp_ids = extract_prod_ids(row, error_records)
            date_request_ids.update(req_ids)
            date_response_ids.update(resp_ids)
        
        # 计算当日覆盖率
        if date_request_ids:
            coverage_rate = len(date_response_ids & date_request_ids) / len(date_request_ids)
            daily_coverage[date] = coverage_rate
        else:
            daily_coverage[date] = 0.0
            
        # 更新总体集合
        total_request_ids.update(date_request_ids)
        total_response_ids.update(date_response_ids)
    
    # 计算总体覆盖率
    total_coverage = (
        len(total_response_ids & total_request_ids) / len(total_request_ids)
        if total_request_ids else 0.0
    )
    
    return {
        'daily_coverage': daily_coverage,
        'total_coverage': total_coverage,
        'error_records': error_records
    }

def save_error_records(error_records: list, output_path: str):
    """
    将错误记录保存到CSV文件
    """
    if error_records:
        error_df = pd.DataFrame(error_records)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_path}/error_records_{timestamp}.csv"
        error_df.to_csv(filename, index=False, encoding='utf-8')
        print(f"错误记录已保存到: {filename}")
    else:
        print("没有发现错误记录")

def main():
    # 读取数据
    try:
        df = pd.read_csv('your_data.csv')  # 替换为实际的数据文件路径
        
        # 确保必要的列存在
        required_columns = ['statis_date', 'model_request', 'model_response']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in DataFrame")
        
        # 计算覆盖率
        results = calculate_coverage_rate(df)
        
        # 输出结果
        print("\n每日覆盖率:")
        for date, rate in results['daily_coverage'].items():
            print(f"日期 {date}: {rate:.2%}")
        
        print(f"\n总体覆盖率: {results['total_coverage']:.2%}")
        
        # 保存错误记录
        save_error_records(results['error_records'], 'error_logs')  # 替换为实际的输出目录
        
    except Exception as e:
        print(f"处理数据时发生错误: {e}")

if __name__ == "__main__":
    main()
