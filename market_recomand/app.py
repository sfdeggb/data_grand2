from flask import Flask

from common.config import config
from services import market_blueprint_v2

def create_app(env: str = ''):
    _app = Flask(__name__)
    _app.config.from_object(config[env])
    _app.register_blueprint(market_blueprint_v2)
    return _app

app = create_app()

if __name__ == '__main__':
    app.run('0.0.0.0')