import os
import time
import socket
import pickle
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from flask import request, current_app
from flask.views import MethodView
from flask.blueprints import Blueprint

from tensorflow.python.keras.models import load_model
from deepctr.layers import custom_objects

from .config import *

class ShortVideoRecommend(MethodView):
    init_every_request = False

    def __init__(self):
        self._load_model()
        self._load_pkl()
        self._get_labels()
        self._pad_length = 1547
        self._weight_5s = 0.3
        self._weight_avg = 0.2
        self._weight_deal = 0.1
        self._weight_cnt = 0.1
        self._weight_like = 0.1
        self._weight_share = 0.1
        self._weight_pub = 0.1
        self._batch_size = 32
        self._trt_model_path = "path/to/save/trt_model"

    def _load_model(self):
        try:
            # 尝试使用 TensorRT
            if tf.test.is_built_with_cuda():
                from tensorflow.python.compiler.tensorrt import trt_convert as trt
                # ... TensorRT 相关代码 ...
            else:
                raise RuntimeError("TensorRT not available")
        except Exception as e:
            print(f"Warning: Could not use TensorRT ({str(e)})")
            print("Falling back to standard model with optimization...")
            
            # 使用替代优化方案
            self.model = load_model(model_path, custom_objects)
            self.model.make_predict_function()  # 预热模型
            
            # 启用 XLA 优化
            tf.config.optimizer.set_jit(True)
            
            # 设置线程数
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(4)

    def _load_pkl(self):
        with open(label_encoders_path, 'rb') as f:
            self.label_encoders = pickle.load(f)
        with open(min_max_scaler_path, 'rb') as f:
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
    def _min_max(self, _min, _max, val):
        return (val - _min) / (_max - _min)
    
    def post(self):
        # 解析用户请求，获取推荐视频
        request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        load1, load5, load15 = os.getloadavg()
        ip = socket.gethostbyname(socket.gethostname())

        body = request.get_json()
        user_feature = body['inputs']['user_field']
        item_feature = body['inputs']['data_field']

        # 冷启动，热度计算
        if not user_feature['seq']:
            t_cold_start = time.time()

            cur_avg_watch_time = [item['cur_avg_watch_time'] for item in item_feature]  # 当前视频平均观看时长
            cur_play_video_cnt = [item['cur_play_video_cnt'] for item in item_feature]  # 当前视频播放次数
            feed_like_count = [item['feed_like_count'] for item in item_feature]  # 点赞数
            feed_share_count = [item['feed_share_count'] for item in item_feature]  # 分享数

            min_avg, max_avg = min(cur_avg_watch_time), max(cur_avg_watch_time)
            min_cnt, max_cnt = min(cur_play_video_cnt), max(cur_play_video_cnt)
            min_like, max_like = min(feed_like_count), max(feed_like_count)
            min_share, max_share = min(feed_share_count), max(feed_share_count)

            result = [
                {
                    'id': item['video_id'],
                    'score': (
                        item['cur_5s_video_ratio'] * self._weight_5s +
                        item['cur_handle_rate'] * self._weight_deal +
                        self._min_max(min_avg, max_avg, item['cur_avg_watch_time']) * self._weight_avg +
                        self._min_max(min_cnt, max_cnt, item['cur_play_video_cnt']) * self._weight_cnt +
                        self._min_max(min_like, max_like, item['feed_like_count']) * self._weight_like +
                        self._min_max(min_share, max_share, item['feed_share_count']) * self._weight_share +
                        self._weight_pub / ((datetime.now() - datetime.fromtimestamp(item['publishTime'] / 1000)).days + 1)
                    )
                }
                for item in item_feature  # 返回多个物料的id和分数组成的字典
            ]
            result.sort(key=lambda x: -x['score'])

            cold_start_taking_time = time.time() - t_cold_start
            current_app.logger.info(f'冷启动耗时: {cold_start_taking_time}')
            return {
                'result': result,
                'type': 'cold_start',
                'logger': {
                    'request_time': request_time,
                    'ip': ip,
                    'load1': load1,
                    'load5': load5,
                    'load15': load15,
                    'cold_start_taking_time': cold_start_taking_time,
                }
            }

        # 模型推荐
        t1 = time.time()

        # 数据预处理
        id_counter = Counter([int(item['video_id']) for item in user_feature['seq']])
        type1_counter = Counter([item.get('product_type1', '') for item in user_feature['seq']])
        type2_counter = Counter([item.get('product_type2', '') for item in user_feature['seq']])

        env_watch_time_his = [item['env_watch_time'] for item in user_feature['seq']] or [0]
        env_video_time_his = [item['env_video_time'] for item in user_feature['seq']] or [0]

        user_feature['max_watch_time'] = max(env_watch_time_his)  # 历史最大观看时长
        user_feature['min_watch_time'] = min(env_watch_time_his)  # 历史最小观看时长
        user_feature['max_video_time'] = max(env_video_time_his)  # 历史最大视频时长
        user_feature['min_video_time'] = min(env_video_time_his)  # 历史最小视频时长
        user_feature['avg_video_time'] = sum(env_video_time_his) / (len(env_video_time_his) or 1)  # 历史平均视频时长
        # 数据预处理
        # 历史视频id，一二级标签填充，标签化
        label_hist_id = [self._video_id_label_mapper.get(int(i['video_id']), 0) for i in user_feature['seq']]
        label_hist_type1 = [self._product_type1_label_mapper.get(i.get('product_type1', ''), 0) for i in user_feature['seq']]
        label_hist_type2 = [self._product_type2_label_mapper.get(i.get('product_type2', ''), 0) for i in user_feature['seq']]

        user_feature['hist_video_id'] = self._pad_end(label_hist_id, self._pad_length, 0)
        user_feature['hist_product_type1'] = self._pad_end(label_hist_type1, self._pad_length, 0)
        user_feature['hist_product_type2'] = self._pad_end(label_hist_type2, self._pad_length, 0)
        user_feature['seq_length'] = len(user_feature['seq'])

        # 目标视频历史观看次数
        # 一级类目历史出现频次
        # 二级类目历史出现频次
        for item in item_feature:
            item['watch_cnt'] = id_counter[int(item['video_id'])]
            item['type1_cnt'] = type1_counter[item.get('product_type1', '')]
            item['type2_cnt'] = type2_counter[item.get('product_type2', '')]

        features = [item | user_feature for item in item_feature]

        df = pd.DataFrame(features)
        df[dense_features] = self.min_max_scaler.transform(df[dense_features])
        units = {k: np.array(v) for k, v in df[sparse_features + dense_features + varlen_features].to_dict('list').items()}
        preprocess_taking_time = time.time() - t1
        current_app.logger.info(f'数据预处理耗时:{preprocess_taking_time}')

        t2 = time.time()
        predictions = []
        for i in range(0, len(features), self._batch_size):
            batch_units = {k: v[i:i+self._batch_size] for k, v in units.items()}
            batch_pred = self.model_predict(**{
                k: tf.convert_to_tensor(v) for k, v in batch_units.items()
            })
            output_key = list(batch_pred.keys())[0]
            predictions.extend(batch_pred[output_key].numpy())
        
        pred = np.array(predictions)
        predict_taking_time = time.time() - t2
        current_app.logger.info(f"预测耗时:{predict_taking_time}, 序列长度:{len(user_feature['seq'])}")

        t3 = time.time()
        result = [{'id': features[i]['video_id'], 'score': pred[i].item()} for i in range(df.shape[0])]
        result.sort(key=lambda x: -x['score'])
        sort_taking_time = time.time() - t3
        current_app.logger.info(f'排序耗时:{sort_taking_time}')

        return {
            'result': result,
            'type': 'model',
            'logger': {
                'request_time': request_time,
                'ip': ip,
                'load1': load1,
                'load5': load5,
                'load15': load15,
                'preprocess_taking_time': preprocess_taking_time,
                'predict_taking_time': predict_taking_time,
                'sort_taking_time': sort_taking_time,
            }
        }
short_video_blueprint_v2 = Blueprint('short_video_blueprint_v2', __name__, url_prefix='/api/v2/short_video')
short_video_blueprint_v2.add_url_rule('/recommend', view_func=ShortVideoRecommend.as_view('short_video_recommend_v2'))