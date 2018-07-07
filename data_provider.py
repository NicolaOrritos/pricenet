#!/usr/bin/env python

import requests
import numpy as np
import pandas as pd
import json
import pytz


def _GET(url):
    req  = requests.get(url)
    data = req.json()

    return data


def _get_data(url):
    raw = _GET(url)

    return json.dumps(raw['Data'])


def _get_prices(data):

    aux = pd.read_json(data, convert_dates=['time'])

    result = pd.DataFrame(aux, columns=['time', 'close', 'open', 'high', 'low', 'volumefrom', 'volumeto'])

    rome_tz = pytz.timezone('Europe/Rome')

    result['time'].dt.tz_localize(pytz.UTC).dt.tz_convert(rome_tz)

    return result


def _group(data, step=4):
    data['group_info'] = ['data' if (index+1)%step != 0 else 'target' for index, _ in data.iterrows()]
    data['type'] = data['group_info'].astype('category')

    del(data['group_info'])

    return data

def _bundle_groups(data, index, group_size):
    return np.concatenate([data.iloc[index + a] for a in range(0, group_size)])

def scale(data_frame):
    min_values =  data_frame.min()
    max_values =  data_frame.max()

    data_frame -= data_frame.min()
    data_frame /= data_frame.max()

    return data_frame, min_values, max_values

def remove_fields(data, fields):
    for field in fields:
        del(data[field])

    return data

def split_to_X_y(data, groups_size, fields_to_remove=[]):
    semi_grouped = _group(data, step=groups_size)

    grouped_data    = semi_grouped.loc[semi_grouped['type'] == 'data']
    grouped_targets = semi_grouped.loc[semi_grouped['type'] == 'target']

    del(grouped_data['type'])
    del(grouped_targets['type'])

    # Make them their own DataFrame to avoid operating on copies of `semi_grouped` one:
    grouped_data = grouped_data.copy()
    grouped_targets = grouped_targets.copy()

    usable_items = groups_size - 1

    X = [_bundle_groups(grouped_data, index, usable_items) for index in range(0, len(grouped_data), usable_items)]
    y = grouped_targets['close'].values.tolist()

    return X, y


def cut_trailing(data, groups_size=4):
    # Cut trailing data (remember that we are grouping by 'groups_size'):
    while len(data) % groups_size > 0:
        data = data[:-1]

    return data


def load(groups_size=4, resolution='hour'):
    """ Returns a dataset containing the prices for the time-period specified. """
    # Get data:
    curr = 'BTC'
    fiat = 'USD'
    exchange = 'CCCAGG'

    if resolution == 'hour':
        samples = 24 * 82  # 82 days worth of hour-sized data
    elif resolution == 'day':
        samples = 5 * 365 + 1  # Five years worth of data
    else:
        # Assume hours:
        resolution = 'hour'
        samples = 24 * 82  # 82 days worth of hour-sized data

    url = ('https://min-api.cryptocompare.com/data/histo{0}?'
        +  'fsym={1}&tsym={2}'
        +  '&limit={3}'
        +  '&e={4}'
        +  '&aggregate=1')

    url = url.format(resolution, curr, fiat, samples, exchange)

    data = _get_data(url)
    prices = _get_prices(data)

    print('Got {0} samples...'.format(len(prices.index)))

    prices['day_of_week'] = prices['time'].dt.dayofweek
    prices['day_of_month'] = prices['time'].dt.day
    prices['day_of_month_scaled'] = prices['time'].dt.day / prices['time'].dt.days_in_month
    prices['month'] = prices['time'].dt.month

    # Do not consider if resolution is 'day':
    if resolution == 'hour':
        prices['time_of_day'] = prices['time'].dt.time.apply(lambda time: str(time).split(':')[0]).astype(int)


    return prices
