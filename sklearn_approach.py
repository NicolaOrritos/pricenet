#!/usr/bin/env python

from time import sleep
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import data_provider as dp


def run_and_score():
    # Get data:
    X, y = dp.load()


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

    return score



# Run multiple times and evaluate the average score:
scores = []
for i in range(10):
    score = run_and_score()
    scores.append(score)

    # We are kind and don't call the remote API too aggressively:
    sleep(2)

scores = np.array(scores)

print('Average score: {}%'.format(round(scores.mean() * 100, 4)))
