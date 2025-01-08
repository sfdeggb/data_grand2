from secrets import token_hex
from collections import defaultdict
from logging.config import dictConfig

class BaseConfig:
   SECRET_KEY = token_hex()

class DevelopmentConfig(BaseConfig):
   DEBUG = True

class ProductionConfig(BaseConfig):
   pass

class TestingConfig(BaseConfig):
   TESTING = True

