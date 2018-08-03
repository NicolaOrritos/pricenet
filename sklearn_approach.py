#!/usr/bin/env python

from random import randint
from time import sleep
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
# import extended_data_provider as dp
import data_provider as dp


def run_and_score(data):
    # Cut the first ones until len(data) % 4 == 0:
    # print('Cutting trailing data off...')
    # data = dp.cut_trailing(data, groups_size=4)

    # Remove some useless fields:
    # print('Removing useless fields...')
    data = dp.remove_fields(data, ['time'])

    data.head()

    # Scale them all:
    # print('Scaling data...')
    data, min_values, max_values = dp.min_max_scale(data)

    # print('min values: ', min_values)
    # print('MAX values: ', max_values)

    # Cut a slice of the data,
    # because otherwise it would take ages
    # to run against the whole dataset:
    # print('Slicing data...')
    # data = data[-(4 * 20000):].copy()

    # Split into X and y:
    # print('Splitting data to "X" and "y" sets...')
    X, y = dp.split_to_X_y(data, groups_size=4)


    # Saved for later usage:
    features_number = len(X[0])
    pick = randint(0, (len(X) - 1))
    X_last = np.array(X[pick])
    y_last = np.array(y[pick])


    # # Let's obtain "X" and "y" training and test sets:
    # print('Splitting into training and test sets...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # print('Sizes: training-set X => {0}, y => {1}; test-set X => {2}, y => {3}'.format(len(X_train), len(y_train), len(X_test), len(y_test)))

    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)


    clf = RandomForestRegressor()

    # print('Fitting model to data...')
    clf.fit(X_train, y_train)

    # print('Scoring model...')
    score = clf.score(X_test, y_test)


    # Try predicting latest prices:
    print('##########################################')
    print('This iteration performance:')
    predicted = clf.predict(X_last.reshape(-1, features_number))[0]
    actual    = y_last

    difference = (1 - actual / predicted) * 100

    predicted = dp.min_max_rescale(predicted, min_values['usd_close'], max_values['usd_close'])
    actual    = dp.min_max_rescale(actual,    min_values['usd_close'], max_values['usd_close'])

    print('Predicted:', predicted)
    print('Actual:   ', actual)
    print('Error:    {}%'.format(round(difference, 2)))
    print('##########################################')


    return score



print('Starting...')

# Get data:
print('Loading data...')
data = dp.load(resolution='hour')

runs = 10

# Run multiple times and evaluate the average score:
print('Performing the linear-regression {0} times...'.format(runs))
scores = []
for i in range(runs):
    score = run_and_score(data.copy())
    scores.append(score)

    # We are kind and don't call the remote API too aggressively:
    sleep(2)

    # print('============================')

scores = np.array(scores)

print('Calculating average score...')
print('Average score: {}%'.format(round(scores.mean() * 100, 4)))
