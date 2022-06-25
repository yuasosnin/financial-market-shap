import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime, math, os
from tqdm import tqdm
from models.utils import *

from finam import Exporter, Market, Timeframe
from enum import IntEnum
import investpy

class Market(IntEnum):

    """
    Markets mapped to ids used by finam.ru export
    List is incomplete, extend it when needed
    """

    SHARES = 1
    BONDS = 2
    COMMODITIES = 24
    CURRENCIES_WORLD = 5
    CURRENCIES = 45
    ETF = 28
    ETF_MOEX = 515
    FUTURES = 14
    FUTURES_ARCHIVE = 17
    FUTURES_USA = 7
    INDEXES_WORLD = 6
    INDEXES = 91
    SPB = 517
    USA = 25
    CRYPTO_CURRENCIES = 520
    
    
class FinamDataLoader():
    def __init__(self):
        self.exporter = Exporter()

    def load(self, name, market=Market.SHARES, timeframe=Timeframe.DAILY, start_date=None, end_date=None):
        if start_date is None:
            start_date = datetime.date(2017,1,1)
        if end_date is None:
            end_date = datetime.date(2020,1,1)
            
        if market == Market.CURRENCIES:
            idx = self.exporter.lookup(name=name, market=market).index[0]
        else:
            idx = self.exporter.lookup(code=name, market=market).index[0]
        
        data = self.exporter.download(
            idx, 
            market=market,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe)
        
        data = data.reset_index(drop=True)
        data.columns = map(lambda x: x.replace('<', '').replace('>', '').lower(), data.columns)
        data = data.rename({'vol': 'volume'}, axis=1)
        data['date'] = pd.to_datetime(data['date'].astype(str)+data['time'], format='%Y%m%d%H:%M:%S')
        data = data.drop('time', axis=1)
        data = data.drop_duplicates('date')
        data['return'] = logreturn(data['close'])
        data['highlow'] = (data['high'] - data['low']) / data['low']
        data['volume'] = np.log(data['volume'] + 1)
        data.columns = [column+('_'+name.lower())*bool(column not in ['date_time', 'date', 'time']) for column in data.columns]
        return data
    
    def load_multiple(self, names, market=Market.SHARES, timeframe=Timeframe.DAILY, start_date=None, end_date=None):
        df_list = [self.load(n, market=market, timeframe=timeframe, start_date=start_date, end_date=end_date) for n in tqdm(names)]
        return multi_merge(df_list, on='date').sort_values('date')

    
def make_daily(data):
    idx = {c.split('_')[-1] for c in data.columns}-{'date'}
    op = {f'open_{d}': 'first' for d in idx}
    cl = {f'close_{d}': 'last' for d in idx}
    hi = {f'high_{d}': 'max' for d in idx}
    lo = {f'low_{d}': 'min' for d in idx}
    vo = {f'volume_{d}': 'sum' for d in idx}

    groupdict = op
    for i in (cl, hi, lo, vo):
        groupdict.update(i)
    
    data_day = data.groupby(data.date.dt.date).aggregate(groupdict)
    for i in idx:
        data_day[f'return_{i}'] = logreturn(data_day[f'close_{i}'])
        data_day[f'volatility_{i}'] = data.groupby(data.date.dt.date).aggregate({f'return_{i}': lambda x: np.sqrt((x**2).sum())})
        data_day[f'highlow_{i}'] = (data_day[f'high_{i}'] - data_day[f'low_{i}']) / data_day[f'low_{i}']
        data_day[f'volume_{i}'] = np.log(data_day[f'volume_{i}'] + 1)
    data_day.index = pd.DatetimeIndex(data_day.index)
    
    return data_day.dropna().reset_index()

def add_columns_suffix(df, name):
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.set_index('date', drop=True)
    df.columns = [f'{c}_{name}' for c in df.columns]
    df = df.reset_index()
    return df

def logreturn(s):
    return np.log(s) - np.log(s.shift(1))
   

indexes = ['IMOEX', 'RTSI']
commodities = ['GC', 'NG', 'BZ']
shares = ['GAZP', 'SBER', 'LKOH', 'GMKN', 'NVTK', 'MGNT', 'ROSN', 'TATN', 'MTSS']
sectors = ['MOEXOG', 'MOEXEU', 'MOEXTL', 'MOEXMM', 'MOEXFN', 'MOEXCN', 'MOEXCH']
foreign = ['UKX', 'INX', 'NDX']
bonds = ['1W', '1M', '6M', '1Y', '3Y', '5Y', '10Y', '20Y']


if __name__ == '__main__':
    os.makedirs('./data', exist_ok=True)

    data_indexes = FinamDataLoader().load_multiple(names=indexes, market=Market.INDEXES, timeframe=Timeframe.MINUTES5)
    make_daily(data_indexes).to_csv('data/indexes_day.csv', index=False)
    print('indexes loaded')

    data_commodities = FinamDataLoader().load_multiple(names=commodities, market=Market.COMMODITIES, timeframe=Timeframe.MINUTES5)
    make_daily(data_commodities).to_csv('data/commodities_day.csv', index=False)
    print('commodities loaded')

    data_foreign = FinamDataLoader().load_multiple(names=foreign, market=Market.INDEXES_WORLD, timeframe=Timeframe.MINUTES5)
    make_daily(data_foreign).to_csv('data/foreign_day.csv', index=False)
    print('foreign loaded')

    data_shares = FinamDataLoader().load_multiple(names=shares, market=Market.SHARES, timeframe=Timeframe.MINUTES5)
    make_daily(data_shares).to_csv('data/shares_day.csv', index=False)
    print('shares loaded')

    data_sectors = FinamDataLoader().load_multiple(names=sectors, market=Market.INDEXES, timeframe=Timeframe.DAILY)
    data_sectors.to_csv('data/sectors_day.csv', index=False)
    print('sectors loaded')

    curr_list = []
    for i in ['usd', 'eur']:
        data_curr = investpy.currency_crosses.get_currency_cross_historical_data(
            f'{i.upper()}/RUB', from_date='01/01/2017', to_date='01/01/2020')
        data_curr = data_curr.drop('Currency', axis=1)
        data_curr['return'] = logreturn(data_curr['Close'])
        data_curr['highlow'] = (data_curr['High'] - data_curr['Low']) / data_curr['Low']
        data_curr = add_columns_suffix(data_curr, i)
        curr_list.append(data_curr)
    multi_merge(curr_list, on='date').sort_values('date').to_csv('data/currencies_day.csv', index=False)
    print('currencies loaded')
    
    bonds_list = []
    for i in bonds:
        data_bond = investpy.bonds.get_bond_historical_data(
            f'Russia {i}', from_date='01/01/2017', to_date='01/01/2020', order='ascending', interval='Daily')
        data_bond = data_bond[['Close']]
        data_bond['yield'] = data_bond['Close'].pct_change()
        data_bond = add_columns_suffix(data_bond, i.lower())
        bonds_list.append(data_bond)
    multi_merge(bonds_list, on='date').sort_values('date').to_csv('data/bonds_day.csv', index=False)
    print('bonds loaded')
