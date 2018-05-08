#!/usr/bin/env python

import requests
import numpy as np
import pandas as pd
import json
import pytz


def GET(url):
    req  = requests.get(url)
    data = req.json()

    return data


def get_data(url):
    raw = GET(url)

    return json.dumps(raw['Data'])


def get_prices(data):

    aux = pd.read_json(data, convert_dates=['time'])

    result = pd.DataFrame(aux, columns=['time', 'close', 'open', 'high', 'low', 'volumefrom', 'volumeto'])

    rome_tz = pytz.timezone('Europe/Rome')

    result['time'].dt.tz_localize(pytz.UTC).dt.tz_convert(rome_tz)

    return result


def group(data, step=4):
    data['group_info'] = ['data' if (index+1)%step != 0 else 'target' for index, _ in data.iterrows()]
    data['type'] = data['group_info'].astype('category')

    del(data['group_info'])

    return data

def scale(data_frame):
    data_frame -= data_frame.min()
    data_frame /= data_frame.max()

    return data_frame

def bundle_groups(data, index, group_size):
    return np.concatenate([data.iloc[index + a] for a in range(0, group_size)])


def load():
    """ Returns `X` and `y` arrays, the former being the training data and the former the targets. """
    # Get data:
    curr = 'BTC'
    fiat = 'USD'
    samples = 24 * 82  # 82 days worth of hour-sized data
    exchange = 'CCCAGG'

    url = ('https://min-api.cryptocompare.com/data/histohour?'
        +  'fsym={0}&tsym={1}'
        +  '&limit={2}'
        +  '&e={3}'
        +  '&aggregate=1')

    url = url.format(curr, fiat, samples, exchange)

    data = get_data(url)
    prices = get_prices(data)


    group_items  = 4
    usable_items = group_items - 1


    semi_grouped = group(prices, step=group_items)

    grouped_data    = semi_grouped.loc[semi_grouped['type'] == 'data']
    grouped_targets = semi_grouped.loc[semi_grouped['type'] == 'target']

    # Make them their own DataFrame to avoid operating on copies of `semi_grouped` one:
    grouped_data = grouped_data.copy()
    grouped_targets = grouped_targets.copy()

    grouped_data['day_of_week'] = grouped_data['time'].dt.dayofweek
    grouped_data['day_of_month'] = grouped_data['time'].dt.day / grouped_data['time'].dt.days_in_month
    grouped_data['month'] = grouped_data['time'].dt.month
    grouped_data['time_of_day'] = grouped_data['time'].dt.time.apply(lambda time: str(time).split(':')[0]).astype(int)

    del(grouped_data['time'])
    del(grouped_data['type'])
    del(grouped_data['open'])

    # Cut trailing data (remember that we are grouping by triplets):
    while len(grouped_data) % usable_items > 0:
        grouped_data = grouped_data[:-1]

    grouped_data = scale(grouped_data)

    X = [bundle_groups(grouped_data, index, usable_items) for index in range(0, len(grouped_data), usable_items)]
    y = grouped_targets['close'].values.tolist()


    return X, y
