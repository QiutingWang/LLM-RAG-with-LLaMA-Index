import os
import dotenv
from loguru import logger
dotenv.load_dotenv()


# 加载配置项，设定默认项，默认选项先用yyh
DEFAULTS = {
    'API_BASE': 'https://gf.nekoapi.com/v1',
    # support for model
    'MODEL_NAME': 'gpt-4 ,Baichuan-13B-Chat',
    'EMBEDDING_NAME': 'text-embedding-ada-002',
    'API_KEYS': 'sk-aJzbu0F3j7bstWlR3e4cA9Db59Ac4f669a9f471aFa66C458',
}


def get_env(key):
    return os.environ.get(key, DEFAULTS.get(key))

class Config:
    """ Configuration class. """
    def __init__(self):
        self.API_BASE = get_env('API_BASE')
        
        self.MODEL_NAME = get_env('MODEL_NAME').split(',')
       
        self.EMBEDDING_NAME = get_env('EMBEDDING_NAME').split(',') #if get_env('EMBEDDING_NAME') else None
       
        self.API_KEYS = get_env('API_KEYS').split(',')[0] #if get_env('API_KEYS') else None

        
config = Config()
logger.debug(f"Config: {config.__dict__}")

