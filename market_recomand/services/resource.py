import os
import time
import pickle
import socket
from datetime import datetime

import pandas as pd
import numpy as np
from flask import request, current_app
from flask.views import MethodView
from flask.blueprints import Blueprint
from tqdm import tqdm

from .config import *
from .config import *

class MarketRecommendView(MethodView):
    """
    商超推荐
    """
    init_every_request = False

    def __init__(self):
        self._view_weight = 0.6
        self._price_weight = 0.4
        self._max_views = 1296
        self._min_views = 0
        self._max_price = 9998
        self._min_price = 0
        self._pad_length = 472
        self._load_model()
        self._load_pkl()
        self._get_labels()

    def _min_max(self, val, min, max):
        return (val - min) / (max - min)

    def _load_model(self):
        # 加载模型
        with open('market_gbdt_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
    def _load_pkl(self):
        with open(label_encoder_path, 'rb') as f:
            self.label_encoders = pickle.load(f)
        with open(standard_scaler_path, 'rb') as f:
            self.min_max_scaler = pickle.load(f)

    def _get_labels(self):
        for k, v in self.label_encoders.items():
            setattr(self, f'_{k}_label_mapper', dict(zip(v.classes_, range(1, len(v.classes_) + 1))))

    def _pad_end(self, sequence, length, pad_value):
        if isinstance(sequence, list):
            pad_value = [pad_value]
        elif isinstance(sequence, tuple):
            pad_value = (pad_value,)
        else:
            pad_value = str(pad_value)

        return sequence + (pad_value * (length - len(sequence))) if len(sequence) < length else sequence
    
    def _get_cross_feature(self,df, user_id='mobile', item_id='skuid', sequence_type='view', cycle=7, batch_size=10000):
        """计算用户-商品交叉特征的优化版本
        """
        # 获取对应的行为序列列名
        seq_col_map = {
            'view': 'qysc_view_seq',
            'click': 'qysc_clk_seq',
            'purchase': 'qysc_order_seq'
        }
        sequence_col = seq_col_map[sequence_type]

        def process_batch(batch_df):
            """处理单个批次的数据"""
            # current_time = pd.Timestamp('20241205')

            # 将序列字符串转换为列表（批量处理）
            sequences = batch_df[sequence_col].fillna('[]').apply(eval)

            # 预分配结果数组
            result_arrays = {
                f'u2i_{cycle}days_{sequence_type}_count': np.zeros(len(batch_df)),
                f'u2i_type1_{cycle}days_{sequence_type}_count': np.zeros(len(batch_df)),
                f'u2i_type2_{cycle}days_{sequence_type}_count': np.zeros(len(batch_df))
            }

            # 向量化处理序列
            for idx, (seq, current_time, current_item, type1, type2) in enumerate(zip(
                sequences,
                batch_df['statis_date'],
                batch_df[item_id],
                batch_df['goods_class_name'],
                batch_df['class_name']
            )):
                current_time = pd.Timestamp(current_time)
                try:
                    # 过滤时间范围内的记录
                    filtered_seq = [
                        s for s in seq 
                        if (current_time - pd.Timestamp(s['oper_time'])).days <= cycle
                    ]
                except Exception as e:
                    print(f"Error processing sequence for row {idx}: {e}")
                    filtered_seq = []
                # print(filtered_seq)
                if filtered_seq:
                    # 使用numpy数组操作进行统计
                    seq_array = np.array([
                        (s['sku_id'], s['first_class_name'], s['second_class_name'])
                        for s in filtered_seq
                    ], dtype=object)
                    result_arrays[f'u2i_{cycle}days_{sequence_type}_count'][idx] = np.sum(seq_array[:, 0] == current_item)
                    result_arrays[f'u2i_type1_{cycle}days_{sequence_type}_count'][idx] = np.sum(seq_array[:, 1] == type1)
                    result_arrays[f'u2i_type2_{cycle}days_{sequence_type}_count'][idx] = np.sum(seq_array[:, 2] == type2)
                else:
                    result_arrays[f'u2i_{cycle}days_{sequence_type}_count'][idx] = 0
                    result_arrays[f'u2i_type1_{cycle}days_{sequence_type}_count'][idx] = 0
                    result_arrays[f'u2i_type2_{cycle}days_{sequence_type}_count'][idx] = 0
                    return pd.DataFrame(result_arrays)
                # 批量处理数据
        result_dfs = []

        for start_idx in tqdm(range(0, len(df), batch_size), desc=f"处理 {sequence_type} 特征"):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            result_df = process_batch(batch_df)
            result_dfs.append(result_df)

        # 合并结果
        final_result = pd.concat(result_dfs, axis=0)
        final_result.index = df.index
        res = pd.concat([df, final_result], axis=1)
        return res
    
    def _process_features(self,df, cycle=7, batch_size=10000):
        """处理所有类型的交叉特征"""
        result_df = df.copy()

        # 使用tqdm显示总体进度
        with tqdm(total=1, desc="处理批次...") as pbar:
            # 处理各类行为特征
            for sequence_type in ['view', 'click', 'purchase']:
                result_df = self._get_cross_feature(
                    result_df,
                    sequence_type=sequence_type,
                    batch_size=batch_size,
                    cycle=cycle
                )
            pbar.update(1)

        return result_df
    
    def _process_real_time_features(self,df, cycle=1, batch_size=10000):
        """处理所有类型的交叉特征"""
        result_df = df.copy()
        # 使用tqdm显示总体进度
        with tqdm(total=1, desc="处理批次...") as pbar:
            
            result_df = self._get_cross_feature(
                result_df,
                sequence_type='click',
                batch_size=batch_size,
                cycle=cycle
            )
            pbar.update(1)

        return result_df
    
    def _process_all_features(self,df):
        data = self._process_features(df,cycle=7,batch_size=10)
        data = self._process_features(data,cycle=14,batch_size=10)
        data = self._process_features(data,cycle=30,batch_size=10)
        data = self._process_features(data,cycle=60,batch_size=10)
        data = self._process_real_time_features(data,cycle=1,batch_size=10)
        return data
    
    def post(self):
        """
        解析用户请求，获得推荐的商品
        """
        request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        load1, load5, load15 = os.getloadavg()
        ip = socket.gethostbyname(socket.gethostname())

        t1 = time.time()
        payload = request.json
        packet_parsing_time = time.time() - t1
        current_app.logger.info(f'报文解析耗时：{packet_parsing_time}')

        request_id = payload['requestId']  # 获取请求id
        items = payload['rawInstances']  # 获取待推荐商品实例
        view_seq = payload['commonFeatures']['qy_sc_view_seq']  # 用户浏览序列
        click_seq = payload['commonFeatures']['qy_sc_click_seq']  # 用户点击序列
        buy_seq = payload['commonFeatures']['qy_sc_buy_seq']  # 用户购买序列

        # 冷启动，热度计算
        if not view_seq:
            t_cold_start = time.time()

            # 实现冷启动推荐逻辑
            instances = [{
                'id': item['id'],
                'scores': [
                    self._min_max(item['rawFeatures']['goods_30day_views'],
                                self._min_views, self._max_views) * self._view_weight
                    + (1 - self._min_max(item['rawFeatures']['c_price'],
                                        self._min_price, self._max_price)) * self._price_weight
                ]
            } for item in items]

            instances.sort(key=lambda x: x['scores'][0], reverse=True)

            cold_start_taking_time = time.time() - t_cold_start

            current_app.logger.info(f"冷启动耗时：{cold_start_taking_time}")

            return {
                'requestId': request_id,
                'status': 'OK',
                'instances': instances,
                'type': 'cold_start',
                'logger': {
                    'request_time': request_time,
                    'ip': ip,
                    'load1': load1,
                    'load5': load5,
                    'load15': load15,
                    'packet_parsing_time': packet_parsing_time,
                    'cold_start_taking_time': cold_start_taking_time,
                }
            }
        
        # 模型推荐
        t2 = time.time()

        # 特征处理
        items_features = [item['rawFeatures'] for item in items]
        df = pd.DataFrame(items_features)
        df['qysc_view_seq'] = [view_seq] * len(df)
        df['qysc_clk_seq'] = [click_seq] * len(df)
        df['qysc_order_seq'] = [buy_seq] * len(df)

        df = self._process_all_features(df)

        # 特征归一化
        df[dense_features] = self.min_max_scaler.transform(df[dense_features])

        # 特征编码
        for feat in sparse_features:
            mapper = getattr(self, f'_{feat}_label_mapper')
            df[feat] = df[feat].apply(lambda x: mapper.get(x, 0))

        preprocess_taking_time = time.time() - t2
        current_app.logger.info(f'数据预处理耗时: {preprocess_taking_time}')

        # 模型预测
        t3 = time.time()
        pred = self.model.predict(df)
        predict_taking_time = time.time() - t3
        current_app.logger.info(f'预测耗时: {predict_taking_time}, 序列长度: {len(view_seq)}')

        instances = [{'id': items[i]['id'], 'scores': [pred[i].item()]} for i in range(len(items))]
        instances.sort(key=lambda x: x['scores'][0], reverse=True)
        return {
            'requestId': request_id,
            'status': 'OK',
            'instances': instances,
            'type': 'model',
            'logger': {
                'request_time': request_time,
                'ip': ip,
                'load1': load1,
                'load5': load5,
                'load15': load15,
                'packet_parsing_time': packet_parsing_time,
                'preprocess_taking_time': preprocess_taking_time,
                'predict_taking_time': predict_taking_time,
            }
        }
    
# 设置请求路由
market_blueprint_v2 = Blueprint('market_blueprint_v2', __name__, url_prefix='/api/v2/market')
market_blueprint_v2.add_url_rule('/recommend', view_func=MarketRecommendView.as_view('market_recommend_v2'))