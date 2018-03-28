
'use strict'

const {URL} = require('url')
const http  = require('http')
const https = require('https')
// const utils = require('util')
const syn   = require('synaptic')


function GET(url, headers = {})
{
    try
    {
        const parsed = new URL(url)

        return new Promise( (resolve, reject) =>
        {
            const options =
            {
                protocol: parsed.protocol.includes(':') ? parsed.protocol : parsed.protocol + ':',
                hostname: parsed.hostname,
                port:     parsed.port,
                path:     parsed.pathname + parsed.search,
                headers
            }

            let client = http

            if (parsed.protocol.includes('https'))
            {
                client = https
            }

            console.time('GET')

            client.get(options, res =>
            {
                let result = ''

                res.on('data',  data => result += data )
                res.on('end',   ()   => console.timeEnd('GET') || resolve(result) )
            })
            .on('error', err => reject(err) )
        })
    }
    catch (err)
    {
        return Promise.reject(err)
    }
}


function getData(url)
{
    return GET(url)
    .then( res =>
    {
        res = JSON.parse(res)

        if (res && res.Data && res.Data.length)
        {
            return Promise.resolve(res.Data)
        }
        else
        {
            console.log('Empty results!')

            return Promise.resolve([])
        }
    })
}

function pricesData(data)
{
    if (data && data.length)
    {
        const result = data.map( item => item.close )

        return Promise.resolve(result)
    }
    else
    {
        console.log('Empty results!')

        return Promise.resolve([])
    }
}

function buildData(data)
{
    let item = 0
    let count = 0
    const result = [{ input: [NaN, NaN, NaN], output: [NaN]}]

    for (let close of data)
    {
        if (item < 3)
        {
            result[count].input[item] = close
        }
        else
        {
            result[count].output[0] = close
        }

        item++

        if (item === 4 && count < data.length / 4 - 1)
        {
            item = 0
            count++
            result.push({ input: [NaN, NaN, NaN], output: [NaN]})
        }
    }

    console.log(`Got #${result.length} of inputs...`)

    return Promise.resolve(result)
}

function normalizeVolumes(data)
{
    const items = data.map( item => item.volumeto - item.volumefrom )

    const min  = Math.min(...items)
    const max  = Math.max(...items)
    const diff = max - min

    const result = []

    items.map( volume => (volume - min) / diff )
    // Cleanup data:
    .forEach( item =>
    {
        if (!isNaN(item))
        {
            result.push(item)
        }
        else
        {
            result.push(0)
        }
    })

    return Promise.resolve(result)
}

function buildVolumeData(data)
{
    return normalizeVolumes(data)
    .then( data =>
    {
        let count = 0
        let index = 0
        const result = [[NaN, NaN, NaN, NaN]]

        for (let item of data)
        {
            result[count][index] = item

            index++

            if (index === 4)
            {
                // Discard last volume:
                result[count] = result[count].slice(0, -1)
            }

            if (index === 4 && count < data.length / 4 - 1)
            {
                index = 0
                count++

                // Prepare the next voulumes quadruplet:
                result.push([NaN, NaN, NaN, NaN])
            }
        }

        return Promise.resolve(result)
    })
}

function normalize(data)
{
    const result = data.map( item =>
    {
        let {input} = item
        input = input.concat([item.output])

        const min  = Math.min(...input)
        const max  = Math.max(...input)
        const diff = max - min

        const normalized = input.map( close => (close - min) / diff )

        return {
            input:  normalized.slice(0, 3),
            output: [normalized[3]]
        }
    })
    // Remove empty or incomplete data:
    .reduce( (result, element) =>
    {
        if (!result || !result.push)
        {
            result = []
        }

        if (   element
            && element.input
            && !isNaN(element.input[0])
            && !isNaN(element.input[1])
            && !isNaN(element.input[2])
            && element.output
            && !isNaN(element.output[0]) )
        {
            result.push(element)
        }

        return result
    })

    return Promise.resolve(result)
}

function addVolumes(data, volumes)
{
    let index = 0

    for (let item of data)
    {
        if (volumes[index] && item.input)
        {
            item.input = item.input.concat(volumes[index])

            index++
        }
    }

    return Promise.resolve(data)
}

function stringify(data)
{
    if (data)
    {
        for (let item of data)
        {
            if (item.input && item.output)
            {
                for (let index in item.input)
                {
                    item.input[index] = '' + item.input[index]
                }

                item.output[0] = '' + item.output[0]
            }
        }

        return Promise.resolve(data)
    }
    else
    {
        return Promise.reject(new Error('No data provided'))
    }
}


function printData(data)
{
    let result = ''

    if (data && data.length)
    {
        for (let item of data)
        {
            result += 'input: ' + item.input + '  -  output: ' + item.output + '\n'
        }
    }

    return result
}

// Shutup jshint:
void printData



const network = new syn.Architect.Perceptron(6, 6, 12, 1)

const exchange = 'CCCAGG'
const curr = 'BTC'
const fiat = 'USD'
const samples = 8 * 80

const url = `https://min-api.cryptocompare.com/data/histohour?`
          + `fsym=${curr}&tsym=${fiat}`
          + `&limit=${samples}`
          + `&e=${exchange}`
          + '&aggregate=1'

console.log(`Calling URL "${url}"...`)

let volumes

getData(url)
.then( data =>
{
    // Save volume triplets to add them later:
    return new Promise( (resolve, reject) =>
    {
        buildVolumeData(data)
        .then( data => volumes = data )
        .then(   () => resolve(data) )
        .catch( err => reject(err) )
    })
})
.then( data => pricesData(data) )
.then( data => buildData(data) )
.then( data => normalize(data) )
.then( data => addVolumes(data, volumes) )
.then( data => stringify(data) )
.then( trainData =>
{
    const trainer = new syn.Trainer(network)

    const testSamples = 16

    // Get last one to be test-data
    const testData = trainData.slice(-testSamples)
    trainData = trainData.slice(0, -testSamples)

    console.log(`Training with ${trainData.length} samples...`)

    const options =
    {
        iterations: 100000,

        log: 2000,

        shuffle: true,
        rate: 0.1,
        error: 0.003
    }

    const result = trainer.train(trainData, options)

    console.log(`Training results: ${JSON.stringify(result)}`)

    return Promise.resolve(testData)
})
.then( testData =>
{
    console.log(`Testing ${testData.length} samples...`)

    const outputs = []
    let error = 0

    for (let item of testData)
    {
        const output = network.activate(item.input)

        outputs.push(output)

        error += Math.pow(item.output - output, 2)
    }

    error = error / testData.length

    /* console.log(`Predicted: ${utils.inspect(outputs)}`)
    console.log(`Actual:    ${utils.inspect(testData)}`) */
    console.log(`Error:     ${error} [MSE]`)
})
.catch( err => console.log(err) )
