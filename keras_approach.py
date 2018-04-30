#!/usr/bin/env python

import requests
import json
import pytz
import tensorflow
import pandas as pd
import numpy as np
from time import time
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


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

del(grouped_data['time'])
del(grouped_data['type'])

# Cut trailing data (remember that we need to group by triplets):
while len(grouped_data) & 3 > 0:
    grouped_data = grouped_data[:-1]


X = [np.concatenate((grouped_data.iloc[a], grouped_data.iloc[a + 1], grouped_data.iloc[a + 2])) for a in range(0, len(grouped_data), 3)]
y = grouped_targets['close'].values.tolist()

X = np.array(X)
y = np.array(y)


features_number = len(X[0])


# Now let's use a neural network:
def baseline_model():
    model = Sequential()
    model.add(Dense(features_number * 2,   input_dim=features_number,   activation='relu'))
    model.add(Dense(256,                                                activation='relu'))
    model.add(Dense(512,                                                activation='relu'))
    model.add(Dense(1024,                                               activation='relu'))
    model.add(Dense(512,                                                activation='relu'))
    model.add(Dense(256,                                                activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model


# evaluate model with standardized dataset
estimators = []
scaler = MinMaxScaler()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

estimators.append(('standardizer', scaler))
estimators.append(('regressor',    KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=10)))

pipeline = Pipeline(estimators)

pipeline.fit(X, y, regressor__validation_split=0.3, regressor__callbacks=[tensorboard])
