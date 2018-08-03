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


def _get_prices(data, prefix='usd_', prefixed_cols=['close', 'open', 'high', 'low', 'volumefrom', 'volumeto']):

    aux = pd.read_json(data, convert_dates=['time'])

    result = pd.DataFrame(aux, columns=['time', 'close', 'open', 'high', 'low', 'volumefrom', 'volumeto'])


    new_names = {}

    for col in prefixed_cols:
        new_names[col] = prefix + col

    result = result.rename(columns=new_names)


    rome_tz = pytz.timezone('Europe/Rome')

    result['time'].dt.tz_localize(pytz.UTC).dt.tz_convert(rome_tz)

    return result

def min_max_scale(data_frame):
    min_values =  data_frame.min()
    max_values =  data_frame.max()

    max_min_diff = max_values - min_values

    data_frame = (data_frame - min_values) / max_min_diff

    return data_frame, min_values, max_values

def min_max_rescale(num, min_value, max_value):
    """ Performs the inverse of a min-max scaler (see method above),
        applied to a single number """
    num = num * (max_value - min_value) + min_value

    return num

def remove_fields(data, fields):
    for field in fields:
        del(data[field])

    return data

def split_to_X_y(data, groups_size, fields_to_remove=[]):
    usable_items = groups_size - 1

    y = data.iloc[usable_items:]['usd_close'].values.tolist()
    split_X = [data.iloc[a:(a + usable_items)].values.tolist() for a in range(0, len(data) - usable_items)]

    # print('Splits for each row of X:   ', len(split_X[0]))
    # print('Length of every split in X: ', len(split_X[0][0]))

    X = [np.concatenate(split) for split in split_X]

    return X, y


def cut_trailing(data, groups_size=4):
    # Cut trailing data (remember that we are grouping by 'groups_size'):
    while len(data) % groups_size > 0:
        data = data[:-1]

    return data


def get_usd_data_url(resolution, curr, samples, exchange):
    fiat = 'USD'
    url = ('https://min-api.cryptocompare.com/data/histo{0}?'
        +  'fsym={1}&tsym={2}'
        +  '&limit={3}'
        +  '&e={4}'
        +  '&aggregate=1')

    url = url.format(resolution, curr, fiat, samples, exchange)
    return url

def get_eur_data_url(resolution, curr, samples, exchange):
    fiat = 'EUR'
    url = ('https://min-api.cryptocompare.com/data/histo{0}?'
        +  'fsym={1}&tsym={2}'
        +  '&limit={3}'
        +  '&e={4}'
        +  '&aggregate=1')

    url = url.format(resolution, curr, fiat, samples, exchange)
    return url

def get_jpy_data_url(resolution, curr, samples, exchange):
    fiat = 'JPY'
    url = ('https://min-api.cryptocompare.com/data/histo{0}?'
        +  'fsym={1}&tsym={2}'
        +  '&limit={3}'
        +  '&e={4}'
        +  '&aggregate=1')

    url = url.format(resolution, curr, fiat, samples, exchange)
    return url

def get_krw_data_url(resolution, curr, samples, exchange):
    fiat = 'KRW'
    url = ('https://min-api.cryptocompare.com/data/histo{0}?'
        +  'fsym={1}&tsym={2}'
        +  '&limit={3}'
        +  '&e={4}'
        +  '&aggregate=1')

    url = url.format(resolution, curr, fiat, samples, exchange)
    return url


def merge(data_frames, on_column='time'):

    merged = data_frames[0]

    for df in data_frames[1:]:
        merged = merged.merge(df, on=on_column)

    return merged


def load(groups_size=4, resolution='hour'):
    """ Returns a dataset containing the prices for the time-period specified. """
    # Get data:
    curr = 'BTC'
    exchange = 'CCCAGG'

    if resolution == 'hour':
        samples = 24 * 83  # 83 days worth of hour-sized data
    elif resolution == 'day':
        samples = 5 * 365 + 1  # Five years worth of data
    else:
        # Assume hours:
        resolution = 'hour'
        samples = 24 * 83  # 83 days worth of hour-sized data

    usd_data = _get_data(get_usd_data_url(resolution, curr, samples, exchange))
    eur_data = _get_data(get_eur_data_url(resolution, curr, samples, exchange))
    krw_data = _get_data(get_krw_data_url(resolution, curr, samples, exchange))
    jpy_data = _get_data(get_jpy_data_url(resolution, curr, samples, exchange))
    usd_prices = _get_prices(usd_data, prefix='usd_', prefixed_cols=['close', 'open', 'high', 'low', 'volumefrom', 'volumeto'])
    eur_prices = _get_prices(eur_data, prefix='eur_', prefixed_cols=['close', 'open', 'high', 'low', 'volumefrom', 'volumeto'])
    krw_prices = _get_prices(krw_data, prefix='krw_', prefixed_cols=['close', 'open', 'high', 'low', 'volumefrom', 'volumeto'])
    jpy_prices = _get_prices(jpy_data, prefix='jpy_', prefixed_cols=['close', 'open', 'high', 'low', 'volumefrom', 'volumeto'])

    prices = merge([usd_prices, eur_prices, krw_prices, jpy_prices], on_column='time')

    print('Got {0} samples...'.format(len(prices.index)))

    prices['day_of_week'] = prices['time'].dt.dayofweek
    prices['day_of_month'] = prices['time'].dt.day
    prices['day_of_month_scaled'] = prices['time'].dt.day / prices['time'].dt.days_in_month
    prices['month'] = prices['time'].dt.month

    # Do not consider if resolution is 'day':
    if resolution == 'hour':
        prices['time_of_day'] = prices['time'].dt.time.apply(lambda time: str(time).split(':')[0]).astype(int)


    return prices
