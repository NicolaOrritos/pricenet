#!/usr/bin/env python

from time import sleep
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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


def run_and_score():
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

    semi_grouped = group(prices, step=4)

    grouped_data    = semi_grouped[semi_grouped['type'] == 'data']
    grouped_targets = semi_grouped[semi_grouped['type'] == 'target']

    grouped_data['day_of_week'] = grouped_data['time'].dt.dayofweek
    grouped_data['month'] = grouped_data['time'].dt.month
    grouped_data['time_of_day'] = grouped_data['time'].dt.time.apply(lambda time: str(time).split(':')[0]).astype(int)

    del(grouped_data['time'])
    del(grouped_data['type'])

    # del(grouped_data['open'])
    del(grouped_data['volumefrom'])

    # Cut trailing data (remember that we need to group by triplets):
    while len(grouped_data) & 3 > 0:
        grouped_data = grouped_data[:-1]

    grouped_data = scale(grouped_data)


    X = [np.concatenate((grouped_data.iloc[a], grouped_data.iloc[a + 1], grouped_data.iloc[a + 2])) for a in range(0, len(grouped_data), 3)]
    y = grouped_targets['close'].values.tolist()


    # # Let's obtain "X" and "y" training and test sets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)


    clf = RandomForestRegressor()

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    print('Classifier score is: ', score)


    # # Let's use these last values to try predicting:
    # X_last = X.pop(-1)
    # y_last = y.pop(-1)
    #
    # X_last = np.array(X_last)
    # features_number = len(X_last)
    #
    # predicted = clf.predict(X_last.reshape(-1, features_number))
    # actual    = y_last
    #
    #
    # difference = (1 - actual/predicted) * 100
    #
    # print('Predicted:', predicted[0])
    # print('Actual:   ', actual)
    # print('Error:    {}%'.format(round(difference[0], 2)))

    return score



# Run multiple times and evaluate the average score:
scores = []
for i in range(10):
    score = run_and_score()
    scores.append(score)
    sleep(2)

scores = np.array(scores)

print('Average score: {}%'.format(round(scores.mean() * 100, 4)))
