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

def scale(data_frame):
    data_frame -= data_frame.min()
    data_frame /= data_frame.max()

    return data_frame

def validate_model(model, test_data, test_targets):
    predictions = []
    for data in test_data:
        predictions.append(model.predict(data.reshape(-1, len(data))))

    predictions = np.array(predictions)

    print(test_targets)
    print(predictions)

    return ((test_targets - predictions)**2).mean(axis=None)



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


# Scale data before grouping them:
grouped_data = scale(grouped_data)


X = [np.concatenate((grouped_data.iloc[a], grouped_data.iloc[a + 1], grouped_data.iloc[a + 2])) for a in range(0, len(grouped_data), 3)]
y = grouped_targets['close'].values.tolist()

X_last = np.array(X.pop())
y_last = np.array(y.pop())

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


features_number = len(X[0])


# Now let's use a neural network:
def baseline_model():
    model = Sequential()
    model.add(Dense(features_number,    input_dim=features_number,    activation='relu'))
    model.add(Dense(1024,               kernel_initializer='normal',  activation='tanh'))
    model.add(Dense(512,                kernel_initializer='normal',  activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


# evaluate model with standardized dataset
estimators = []

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

estimators.append(('regressor', KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=10, verbose=0)))

pipeline = Pipeline(estimators)

pipeline.fit(X_train, y_train, regressor__callbacks=[tensorboard])

score = pipeline.score(X_test, y_test)
mse = validate_model(pipeline, X_test, y_test)

print('Score: ', score)
print('MSE:   ', mse)

# # Fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)
#
# # Evaluate using 10-fold cross validation
# kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
# results = cross_val_score(pipeline, X, y, cv=kfold)
# print('Results: ', results.mean())

predicted = pipeline.predict(X_last.reshape(-1, features_number))
actual    = y_last

difference = (1 - actual/predicted) * 100

print('Predicted:', predicted)
print('Actual:   ', actual)
print('Error:    {}%'.format(round(difference, 2)))


# pipeline.named_steps['regressor'].model.save('model.h5')
