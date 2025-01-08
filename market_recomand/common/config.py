from secrets import token_hex
from collections import defaultdict
from logging.config import dictConfig
import os
class BaseConfig:
   SECRET_KEY = token_hex()

class DevelopmentConfig(BaseConfig):
   DEBUG = True

class ProductionConfig(BaseConfig):
   pass

class TestingConfig(BaseConfig):
   TESTING = True

# flask app config
config = defaultdict(lambda: BaseConfig)
config['dev'] = DevelopmentConfig
config['prod'] = ProductionConfig
config['test'] = TestingConfig

# logs 目录不存在则创建
if not os.path.exists('./logs'):
    os.makedirs('./logs')

# 日志
dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'filename': './logs/app.log',
            'formatter': 'default',
            'encoding': 'utf-8'
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file']
    }
})