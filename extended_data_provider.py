#!/usr/bin/env python

import numpy as np
import pandas as pd
import json
import pytz


def _get_data(file):
    return pd.read_csv(file)


def _get_prices(data):

    df = data

    rome_tz = pytz.timezone('Europe/Rome')

    df['time'] = pd.to_datetime(df['Timestamp'], unit='s')
    df['time'].dt.tz_localize(pytz.UTC).dt.tz_convert(rome_tz)

    del(df['Timestamp'])
    del(df['Weighted_Price'])
    df = df.rename(columns={'Volume_(BTC)': 'volume_btc', 'Volume_(Currency)': 'volume_fiat'})
    df = df.rename(columns={'Open': 'open', 'Close': 'close'})
    df = df.rename(columns={'Low': 'low', 'High': 'high'})

    return df


def _group(data, step=4):
    data['group_info'] = ['data' if (index+1)%step != 0 else 'target' for index, _ in data.iterrows()]
    data['type'] = data['group_info'].astype('category')

    del(data['group_info'])

    return data

def _bundle_groups(data, index, group_size):
    return np.concatenate([data.iloc[index + a] for a in range(0, group_size)])

def scale(data_frame):
    data_frame -= data_frame.min()
    data_frame /= data_frame.max()

    return data_frame

def remove_fields(data, fields):
    for field in fields:
        del(data[field])

    return data

def split_to_X_y(data, groups_size):
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
        data = data.drop(len(data) - 1)

    return data


def load():
    """ Returns `X` and `y` arrays, the former being the training data and the former the targets. """
    # Get data:
    data = _get_data('coinbaseUSD_1-min_data_2014-12-01_to_2018-03-27.csv')
    prices = _get_prices(data)

    prices['day_of_week'] = prices['time'].dt.dayofweek
    prices['day_of_month'] = prices['time'].dt.day
    prices['day_of_month_scaled'] = prices['time'].dt.day / prices['time'].dt.days_in_month
    prices['month'] = prices['time'].dt.month
    prices['time_of_day'] = prices['time'].dt.time.apply(lambda time: str(time).split(':')[0]).astype(int)


    return prices
