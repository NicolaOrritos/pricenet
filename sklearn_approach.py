#!/usr/bin/env python

from random import randint
from time import sleep
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import data_provider as dp
import pandas as pd
from joblib import dump


def eval_against_swans(clf, swans):
    X = np.array([np.concatenate(data.iloc[:3].values.tolist()) for data in swans])
    y = np.array([data.iloc[-1]['usd_close'] for data in swans])

    score = clf.score(X, y)

    features_number = len(X[0])
    errors = []

    for X_items, actual in zip(X, y):
        predicted  = clf.predict(X_items.reshape(-1, features_number))[0]

        # Prevent division-by-zero errors:
        if predicted == 0:
            actual = actual + 0.0000000001
            predicted = predicted + 0.0000000001

        difference = 1 - actual / predicted

        errors.append(difference)

    return score, np.array(errors).mean()


def run_and_score(data):

    # Cut the first ones until len(data) % 4 == 0:
    # print('Cutting trailing data off...')
    # data = dp.cut_trailing(data, groups_size=4)

    # Remove some useless fields:
    # print('Removing useless fields...')
    data = dp.remove_fields(data, ['time'])

    # Scale them all:
    # print('Scaling data...')
    data, min_values, max_values = dp.min_max_scale(data)

    # Split into X and y:
    # print('Splitting data to "X" and "y" sets...')
    X, y = dp.split_to_X_y(data, groups_size=4)


    # Saved for later usage:
    features_number = len(X[0])
    pick = randint(0, (len(X) - 1))
    X_picked = np.array(X[pick])
    y_picked = np.array(y[pick])


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
    predicted = clf.predict(X_picked.reshape(-1, features_number))[0]
    actual    = y_picked

    difference = (1 - actual / predicted) * 100

    predicted = dp.min_max_rescale(predicted, min_values['usd_close'], max_values['usd_close'])
    actual    = dp.min_max_rescale(actual,    min_values['usd_close'], max_values['usd_close'])
    error     = round(difference, 2)

    print('This iteration performance:')
    print('Predicted:', predicted)
    print('Actual:   ', actual)
    print('Error:     {}%'.format(abs(error)))

    return score, error, clf




print('Starting...')

resolution = 'day'

# Get data:
print('Loading data...')
data  = dp.load(resolution)
swans, swans_min_values, swans_max_values = dp.find_swans(data, resolution, groups_size=4)

runs = 10

# Run multiple times and evaluate the average score:
print('Performing the linear-regression {0} times...'.format(runs))

scores = []
errors = []

# Track errors as [0,1] real numbers:
last_error = 1

for i in range(runs):
    print('##########################################')
    score, error, clf = run_and_score(data.copy())
    scores.append(score)
    errors.append(abs(error))

    # Evaluate against swans:
    swans_score, swans_error = eval_against_swans(clf, swans)

    print('Score against {0} swans: {1}%'.format(len(swans), round(swans_score * 100, 4)))
    print('Error against {0} swans: {1}%'.format(len(swans), round(abs(swans_error) * 100, 4)))

    if (abs(error) < last_error):
        print('Dumping best model (as of now)...')
        dump(clf, 'best_model_{0}.joblib'.format(resolution))

scores = np.array(scores)
errors = np.array(errors)

print('##########################################')

print('Calculating average score and error...')
print('Average score: {}%'.format(round(scores.mean() * 100, 4)))
print('Average error: {}%'.format(abs(round(errors.mean(), 4))))
print('Swans percentage over samples: {}%'.format(round(len(swans)/len(data) * 100, 4)))
