
'use strict'

const {URL} = require('url')
const http  = require('http')
const https = require('https')
const brain = require('brain.js')


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
            const result = res.Data.map( item => item.close )

            return Promise.resolve(result)
        }
        else
        {
            console.log('Empty results!')

            return Promise.resolve([])
        }
    })
    .then( data =>
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
        if (!result)
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

    }, [])

    return Promise.resolve(result)
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


const net = new brain.NeuralNetwork()

const exchange = 'CCCAGG'
const curr = 'BTC'
const fiat = 'USD'

const url = `https://min-api.cryptocompare.com/data/histohour?`
          + `fsym=${curr}&tsym=${fiat}`
          + `&limit=240`
          + `&e=${exchange}`
          + '&aggregate=1'

console.log(`Calling URL "${url}"...`)

getData(url)
.then( data => normalize(data) )
.then( trainData =>
{
    // Get last one to be test-data
    const testData = trainData.slice(-1)
    trainData = trainData.slice(0, -1)

    console.log(`Training with data "${printData(trainData)}"...`)

    const options =
    {
        iterations: 40000
    }

    const result = net.train(trainData, options)

    console.log(`Training results: ${JSON.stringify(result)}`)

    return Promise.resolve(testData)
})
.then( testData =>
{
    console.log(`Testing with data "${printData(testData)}"...`)

    const output = net.run(testData[0].input)

    console.log(`Predicted: ${output}`)
    console.log(`Actual:    ${testData[0].output}`)
    console.log(`Error:     ${Math.abs(testData[0].output - output)}`)
})
.catch( err => console.log(err) )
