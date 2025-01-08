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
    
    def _get_cross_feature(self,df,user_id='mobile',item_id='skuid', sequence_type='view', cycle=7):
        """计算用户-商品交叉特征
        
        Args:
            df (pd.DataFrame): 输入数据集
            user_id (str): 用户ID列名
            item_id (str): 商品ID列名 
            sequence_type (str): 行为类型 - 'view'/'click'/'purchase'
            cycle (int): 统计周期 - 7/14/30/60天
        """
        # 获取对应的行为序列列名
        seq_col_map = {
            'view': 'user_view_seq',
            'click': 'user_clk_seq', 
            'purchase': 'user_purchase_seq'
        }
        sequence_col = seq_col_map[sequence_type]
        
        def process_sequence(row, days):
            """处理单条记录的行为序列"""
            current_item = row[item_id]
            sequences = eval(row[sequence_col]) # 将字符串转为列表
            
            # 获取当前时间戳
            #current_time = pd.Timestamp.now()
            #current_time = pd.Timestamp(row['static_date'])
            current_time = pd.Timestamp('20241205')
            # 过滤指定天数内的记录
            filtered_seq = [
                s for s in sequences 
                if (current_time - pd.Timestamp(s['oper_time'])).days <= days
            ]
            
            # 计算商品ID维度统计
            item_count = sum(1 for s in filtered_seq if s['sku_id'] == current_item)
            
            # 计算一级类目维度统计
            type1_count = sum(1 for s in filtered_seq 
                            if s['frist_order_type'] == row['goods_class_name'])
            
            # 计算二级类目维度统计
            type2_count = sum(1 for s in filtered_seq 
                            if s['second_order_type'] == row['class_name'])
            
            return pd.Series({
                f'u2i_{days}days_{sequence_type}_count': item_count,
                f'u2i_type1_{days}days_{sequence_type}_count': type1_count,
                f'u2i_type2_{days}days_{sequence_type}_count': type2_count
            })
        
        # 计算不同时间窗口的特征
        result_df = df.copy()
        
        # 对于点击行为额外计算1天的实时特征
        if sequence_type == 'click':
            result_df = pd.concat([
                result_df,
                df.apply(lambda x: process_sequence(x, 1), axis=1)
            ], axis=1)
        
        # 计算常规时间窗口的特征
        for days in [7, 14, 30, 60]:
            result_df = pd.concat([
                result_df,
                df.apply(lambda x: process_sequence(x, days), axis=1)
            ], axis=1)
        
        return result_df
    
    def _process_all_features(self,df):
        """处理所有类型的交叉特征"""
        # 处理浏览行为特征
        df = self._get_cross_feature(df, sequence_type='view')
        # 处理点击行为特征
        df = self._get_cross_feature(df, sequence_type='click')
        # 处理购买行为特征
        df = self._get_cross_feature(df, sequence_type='purchase')
        return df 
    
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
        df['user_view_seq'] = [view_seq] * len(df)
        df['user_clk_seq'] = [click_seq] * len(df)
        df['user_buy_seq'] = [buy_seq] * len(df)

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