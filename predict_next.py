#!/usr/bin/env python

import sys
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import data_provider as dp
import pandas as pd
from joblib import load


def prediction(data, clf):
    # Scale them all:
    data, min_values, max_values = dp.min_max_scale(data)
    data = np.concatenate(data.iloc[:3].values.tolist())

    features_number = len(data)

    predicted = clf.predict(data.reshape(-1, features_number))[0]
    predicted = dp.min_max_rescale(predicted, min_values['usd_close'], max_values['usd_close'])

    return predicted


resolution = 'day'

if len(sys.argv) == 3 and sys.argv[1] == '-r':
    resolution = sys.argv[2]

print('Using resolution "{0}"...'.format(resolution))


data = dp.load_latest(resolution)

clf = load('best_model_{0}.joblib'.format(resolution))

y = prediction(data, clf)

print('Predicted [resolution="{0}"]: '.format(resolution))
print(y)
