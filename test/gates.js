import { Value } from '../engine.js'
import { MLP } from '../neuron.js'


let log = console.log

function getTrainedMLP(xs, ys) {
    // Create a new MLP with 2 inputs, 1 hidden layer with 2 neurons, and 1 output
    let n = new MLP(2, [2, 1])

    // Train the MLP
    for (let i = 0; i < 1000; i++) {
        xs.forEach((x, i) => {
            let y = ys[i]
            let ypred = n.forward(x)
            let loss = new Value(y, [], '', 'loss')
            loss = loss.sub(ypred).pow(2)
            for (let p of n.parameters()) {
                p.grad = 0.0
            }
            loss.backward()
            for (let p of n.parameters()) {
                p.data += -0.05 * p.grad
            }
        })
    }

    return n;
}

// Training data
let xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]

let ys = [0.0, 1.0, 1.0, 1.0]
let n = getTrainedMLP(xs, ys)

log ("Test on data", xs)

log("Predictions after training (OR gate)")
for (let x of xs) {
    log(n.forward(x).data)
}

ys = [0.0, 0.0, 0.0, 1.0]
n = getTrainedMLP(xs, ys)

log("Predictions after training (AND gate)")
for (let x of xs) {
    log(n.forward(x).data)
}

log("Predictions after training (XOR gate)")
ys = [0.0, 1.0, 1.0, 0.0]
n = getTrainedMLP(xs, ys)
for (let x of xs) {
    log(n.forward(x).data)
}





