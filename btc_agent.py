from finrl.train import train
from finrl.test import test
# from finrl.apps.config import DOW_30_TICKER
# from finrl.apps.config import TECHNICAL_INDICATORS_LIST
from finrl.finrl_meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv
# from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.finrl_meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.finrl_meta.data_processor import DataProcessor
# from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import numpy as np
import pandas as pd


API_KEY = "PKP8FC8CUTXU9G031NEW"
API_SECRET = "XYuqkxIiPcTJEzBO8nVjz9AnjBz4CinNgKiKIFgL"
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'


CRYPTO_TICKERS = ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','XRPUSDT',
                'SOLUSDT']
env = CryptoEnv

TRAIN_START_DATE = '2021-09-01'
TRAIN_END_DATE = '2021-09-20'

TEST_START_DATE = '2021-09-21'
TEST_END_DATE = '2021-09-30'
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

train(start_date = TRAIN_START_DATE, 
      end_date = TRAIN_END_DATE,
      ticker_list = CRYPTO_TICKERS, 
    #   data_source = 'alpaca',
      time_interval= '1Min', 
      technical_indicator_list= TECHNICAL_INDICATORS_LIST,
    #   drl_lib='stable_baselines3', 
      env=env,
      model_name='ppo', 
      API_KEY = API_KEY, 
      API_SECRET = API_SECRET, 
      APCA_API_BASE_URL = APCA_API_BASE_URL,
      erl_params=SAC_PARAMS,
      cwd='./trained_models/agent_ppo', #current_working_dir
      break_step=1e5)