#!/usr/bin/env python

import requests
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


def GET(url):
    req  = requests.get(url)
    data = req.json()

    return data


def get_data(url):
    raw = GET(url)

    return raw['Data']


def prices(data):
    return [item['close'] for item in data]


def volumes(data):
    return [item['volumeto'] - item['volumefrom'] for item in data]


def group(data, step=4):
    return [data[a:(a + step - 1)] for a in xrange(0, len(data), step)]


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

estimator = KerasRegressor(build_fn=baseline_model)
# estimator.fit()








# const network = new syn.Architect.Perceptron(6, 12, 1)
#
# const exchange = 'CCCAGG'
# const curr = 'BTC'
# const fiat = 'USD'
# const samples = 12 * 80
#
# const url = `https://min-api.cryptocompare.com/data/histohour?`
#           + `fsym=${curr}&tsym=${fiat}`
#           + `&limit=${samples}`
#           + `&e=${exchange}`
#           + '&aggregate=1'
#
# console.log(`Calling URL "${url}"...`)
#
# let volumes
#
# getData(url)
# .then( data =>
# {
#     // Save volume triplets to add them later:
#     return new Promise( (resolve, reject) =>
#     {
#         buildVolumeData(data)
#         .then( data => volumes = data )
#         .then(   () => resolve(data) )
#         .catch( err => reject(err) )
#     })
# })
# .then( data => pricesData(data) )
# .then( data => buildData(data) )
# .then( data => normalize(data) )
# .then( data => addVolumes(data, volumes) )
# .then( data => stringify(data) )
# .then( trainData =>
# {
#     const trainer = new syn.Trainer(network)
#
#     const testSamples = 24
#
#     // Get last one to be test-data
#     const testData = trainData.slice(-testSamples)
#     trainData = trainData.slice(0, -testSamples)
#
#     console.log(`Training with ${trainData.length} samples...`)
#
#     const options =
#     {
#         iterations: 200000,
#
#         log: 10000,
#
#         shuffle: true,
#         rate: 0.2,
#         error: 0.005
#     }
#
#     const result = trainer.train(trainData, options)
#
#     console.log(`Training results: ${JSON.stringify(result)}`)
#
#     return Promise.resolve(testData)
# })
# .then( testData =>
# {
#     console.log(`Testing ${testData.length} samples...`)
#
#     const outputs = []
#     let error = 0
#
#     for (let item of testData)
#     {
#         const output = network.activate(item.input)
#
#         outputs.push(output)
#
#         error += Math.pow(item.output - output, 2)
#     }
#
#     error = error / testData.length
#
#     /* console.log(`Predicted: ${utils.inspect(outputs)}`)
#     console.log(`Actual:    ${utils.inspect(testData)}`) */
#     console.log(`Error:     ${error} [MSE]`)
# })
# .catch( err => console.log(err) )
