import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import datetime, math, json
from .data import *


def _load_df(name, date, path=None):
    if path is not None:
        path = path.rstrip('/') + '/'
    else:
        path = ''
    data = pd.read_csv(f'{path}{name}_{date}.csv', sep=',', parse_dates=[['<DATE>', '<TIME>']])
    data.columns = map(lambda x: x.replace('<', '').replace('>', '').lower(), data.columns)
    if name == 'USD000000TOD':
        name = 'USD'
    if name == 'EUR_RUB__TOD':
        name = 'EUR'
    if name == 'ICE.BRN':
        name = 'BRENT'
    if name == 'SANDP-500':
        name = 'SNP'
    if name == 'comex.GC':
        name = 'GOLD'
    if name == 'FUTSEE-100':
        name = 'FTSE'
    if name == 'RI.RTSI':
        name = 'RTSI'
    if name == 'SPFB.MIX':
        name = 'MIX'
    for i in ['ticker', 'per']:
        try:
            data = data.drop(i, axis=1)
        except KeyError:
            pass
    data.columns = [column+('_'+name.lower())*bool(column not in ['date_time', 'date', 'time']) for column in data.columns]
    data = data.rename({'date_time': 'date', f'vol_{name.lower()}': f'volume_{name.lower()}'}, axis=1)
    #data = data.set_index('date')
    return data

def load_data(names, dates, path=''):
    from functools import reduce
    df_list = [pd.concat([_load_df(n, d, path) for d in dates]) for n in tqdm(names)]
    return reduce(lambda df1, df2: pd.merge(df1, df2, how='outer'), df_list)

def load_currency(name, path='data/'):
    data = pd.read_csv(f'{path}/{name}_RUB.csv', parse_dates=[0], dayfirst=True, decimal=',').iloc[::-1]
    name = name.lower()
    data.columns = ['date', 'close', 'open', 'high', 'low', 'change']
    data = data[['date', 'open', 'close', 'high', 'low', 'change']]
    data = data.drop(['change'], axis=1)
    data = data.set_index('date', drop=True)
    data['return'] = data['close'].pct_change()
    data['highlow'] = (data['high'] - data['low']) / data['low']
    data = data[1:]
    data.columns = [f'{c}_{name.lower()}' for c in data.columns]
    return data

def load_yield(names, path='data/'):
    from functools import reduce
    
    dfs = []
    for i in names:
        data = pd.read_csv(f'{path}/DGS{i}.csv')
        data.columns = ['date', f'yield_{i}']
        data = data.loc[data[f'yield_{i}'] != '.', :]
        # data[f'yield_{i}'] = data[f'yield_{i}'].apply(lambda x: x.replace('.','0.'))
        data[f'yield_{i}'] = data[f'yield_{i}'].astype(float)
        dfs.append(data)
    
    data = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), dfs)
    data = data.set_index('date', drop=True)
    data.index = pd.DatetimeIndex(data.index)
    return data


def load_yield_ru(names, path='data/'):
    from functools import reduce
    
    dfs = []
    for name in names:
        data = pd.read_csv(f'{path}/Russia {name} Bond Yield Historical Data.csv', parse_dates=[0], dayfirst=True, decimal=',').iloc[::-1]
        name = name.lower().replace('-', '_')
        data = data[['Date', 'Price']]
        data['Price'] = data['Price'].astype(float)
        data.columns = ['date', f'yield_{name}']
        dfs.append(data)
    
    data = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), dfs)
    data = data.set_index('date', drop=True)
    data.index = pd.DatetimeIndex(data.index)
    return data


def prepare_data(data, idx):
    for i in idx:
        data[f'return_{i}'] = (data[f'close_{i}'] - data.shift()[f'close_{i}']) / data.shift()[f'close_{i}']
        data[f'log_return_{i}'] = np.log(data[f'close_{i}'] / data.shift()[f'close_{i}'])

    
    data['time_norm'] = 2 * math.pi * (data.date.dt.hour * 60 + data.date.dt.minute) / (24*60)
    data['time_cos'] = np.cos(data['time_norm'])
    data['time_sin'] = np.sin(data['time_norm'])
    data = data.drop('time_norm', axis=1)
    
    data['dayofweek_norm'] = 2 * math.pi * data.date.dt.dayofweek / 7
    data['dayofweek_cos'] = np.cos(data['dayofweek_norm'])
    data['dayofweek_sin'] = np.sin(data['dayofweek_norm'])
    data = data.drop('dayofweek_norm', axis=1)
    
    data['dayofweek'] = data.date.dt.dayofweek
    data = pd.get_dummies(data, columns=['dayofweek'])
    return data


def make_daily_data(data, idx):
    op = {f'open_{d}': 'first' for d in idx}
    cl = {f'close_{d}': 'last' for d in idx}
    hi = {f'high_{d}': 'max' for d in idx}
    lo = {f'low_{d}': 'min' for d in idx}
    vo = {f'volume_{d}': 'mean' for d in idx}
    re = {f'return_{d}': lambda x: np.prod(x+1)-1 for d in idx}
    # lr = {f'log_return_{d}': 'sum' for d in idx}

    groupdict = op
    for i in (cl, hi, lo, vo, re):
        groupdict.update(i)
    
    data_day = data.groupby(data.date.dt.date).aggregate(groupdict)
    for i in idx:
        data_day[f'volatility_{i}'] = data.groupby(data.date.dt.date).aggregate({f'return_{i}': lambda x: np.sqrt((x**2).sum())})
        data_day[f'spread_{i}'] = (data_day[f'high_{i}'] - data_day[f'low_{i}']) / data_day[f'low_{i}']
    data_day.index = pd.DatetimeIndex(data_day.index)
    
    data_day['dayofweek'] = data_day.index.dayofweek
    data_day = pd.get_dummies(data_day, columns=['dayofweek'])
    
    return data_day

def make_hourly_data(data, idx):
    op = {f'open_{d}': 'first' for d in idx}
    cl = {f'close_{d}': 'last' for d in idx}
    hi = {f'high_{d}': 'max' for d in idx}
    lo = {f'low_{d}': 'min' for d in idx}
    vo = {f'volume_{d}': 'mean' for d in idx}
    re = {f'return_{d}': lambda x: np.prod(x+1)-1 for d in idx}
    # lr = {f'log_return_{d}': 'sum' for d in idx}

    groupdict = op
    for i in (cl, hi, lo, vo, re):
        groupdict.update(i)
    
    data_day = data.groupby([data.date.dt.date, data.date.dt.hour]).aggregate(groupdict)
    for i in idx:
        data_day[f'volatility_{i}'] = data.groupby([data.date.dt.date, data.date.dt.hour]).aggregate({f'return_{i}': lambda x: np.sqrt((x**2).sum())})
        data_day[f'spread_{i}'] = (data_day[f'high_{i}'] - data_day[f'low_{i}']) / data_day[f'low_{i}']
    data_day.index = pd.DatetimeIndex(data_day.index)
    
    data_day['dayofweek'] = data_day.index.dayofweek
    data_day = pd.get_dummies(data_day, columns=['dayofweek'])
    
    return data_day
