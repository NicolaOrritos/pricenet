#!/usr/bin/env python

import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np


def GET(url):
    req  = requests.get(url)
    data = req.json()

    return data


def get_data(url):
    raw = GET(url)

    return raw['Data']


def get_prices(data):
    return [item['close'] for item in data]


def get_volumes(data):
    return [item['volumeto'] - item['volumefrom'] for item in data]


def group(data, step=4):
    return [data[a:(a + step)] for a in range(0, len(data), step)]


def build_inputs(data, inputs_count=3, outputs_count=1):
    result = {}
    result['input'] = data[0:(inputs_count - 1)]

    if outputs_count > 0:
        result['output'] = data[-outputs_count:]

    return result


def normalize(data):
    mx   = max(data)
    mn   = min(data)
    diff = mx - mn

    return [(item - mn) / diff for item in data]


def combine_data(prices, volumes):
    return [prices[item]['input'].extend(volumes[item]) for item in range(len(prices))]


def baseline_model():
    model = Sequential()
    model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(optimizer='rmsprop', loss='mse')

    return model


# Get data:
curr = 'BTC'
fiat = 'USD'
samples = 12 * 80
exchange = 'CCCAGG'

url = ('https://min-api.cryptocompare.com/data/histohour?'
    +  'fsym={0}&tsym={1}'
    +  '&limit={2}'
    +  '&e={3}'
    +  '&aggregate=1')

url = url.format(curr, fiat, samples, exchange)

data = get_data(url)
prices = get_prices(data)
volumes = get_volumes(data)

grouped_prices  = group(prices, 4)
grouped_volumes = group(volumes, 4)

# Remove 4th item from each group for the prices
# and discard it for the volumes.
# Keep the 4th price of each group as "y".
three_items_prices  = [group[0:3] for group in grouped_prices if len(group) == 4]
three_items_volumes = [group[0:3] for group in grouped_volumes if len(group) == 4]
target_prices       = [group[3] for group in grouped_prices if len(group) == 4]

# Put the three prices and volumes together:
prices_and_volumes = [prices + volumes for prices, volumes in zip(three_items_prices, three_items_volumes)]


# Let's obtain "X" and "y" sets:
X_train, X_test, y_train, y_test = train_test_split(prices_and_volumes, target_prices, test_size=0.3)


X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)


clf = RandomForestRegressor()

clf.fit(X_train, y_train)

# clf.score(X_test, y_test)


predicted = clf.predict(X_test[0].reshape(-1, 6))
actual = y_test[0]


difference = (actual/predicted - 1) * 100

print('Predicted:', predicted[0])
print('Actual:   ', actual)
print('Error:    {}%'.format(round(difference[0], 2)))
