#!/usr/bin/env python

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
import data_provider as dp


def validate_model(model, test_data, test_targets):
    predictions = []
    for data in test_data:
        predictions.append(model.predict(data.reshape(-1, len(data))))

    predictions = np.array(predictions)

    print(test_targets)
    print(predictions)

    return np.absolute(1 - predictions / test_targets).mean(axis=None)



# Get data:
data = dp.load()

# Remove redundant columns:
del(data['time'])
del(data['type'])
del(data['open'])
del(data['day_of_month_scaled'])

# Scale them:
data = dp.scale(data)

# Split into X and y:
X, y = dp.split_to_X_y(data)

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
    model.add(Dense(256,                kernel_initializer='normal',  activation='relu'))
    model.add(Dense(256,                kernel_initializer='normal',  activation='tanh'))
    model.add(Dense(128,                kernel_initializer='normal',  activation='tanh'))
    model.add(Dense(128,                kernel_initializer='normal',  activation='relu'))
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
error = validate_model(pipeline, X_test, y_test)

print('Score: ', score)
print('Error: {}%'.format(round(error * 100, 4)))

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
