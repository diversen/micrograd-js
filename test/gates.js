import { Value } from '../engine.js'
import { MLP } from '../neuron.js'


let log = console.log

function getTrainedMLP(mlp, xs, ys) {

    // Train the MLP
    for (let i = 0; i < 1000; i++) {
        xs.forEach((x, i) => {
            let y = ys[i]
            let ypred = mlp.forward(x)
            let loss = new Value(y, [], '', 'loss')
            loss = loss.sub(ypred).pow(2)
            for (let p of mlp.parameters()) {
                p.grad = 0.0
            }
            loss.backward()
            for (let p of mlp.parameters()) {
                p.data += -0.05 * p.grad
            }
        })
    }

    return mlp;
}

// Training data
let xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]

let ys = [0.0, 1.0, 1.0, 1.0]

log("Test on data", xs)

// Create a new MLP with 2 inputs, 1 hidden layer with 2 neurons, and 1 output
let mlp = new MLP(2, [2, 1])
let n = getTrainedMLP(mlp, xs, ys)
log("Predictions after training (OR gate)")
for (let x of xs) {
    log(n.forward(x).data)
}

mlp = new MLP(2, [2, 1])
ys = [0.0, 0.0, 0.0, 1.0]
n = getTrainedMLP(mlp, xs, ys)
log("Predictions after training (AND gate)")
for (let x of xs) {
    log(n.forward(x).data)
}

// XOR gate. 4 neurons in the hidden layer
mlp = new MLP(2, [4, 1])
ys = [0.0, 1.0, 1.0, 0.0]
n = getTrainedMLP(mlp, xs, ys)
log("Predictions after training (XOR gate)")
for (let x of xs) {
    log(n.forward(x).data)
}

// NOT gate
mlp = new MLP(1, [2, 1])
xs = [
    [0.0], [1.0]
]
ys = [1.0, 0.0]

log("Test on data", xs)

n = getTrainedMLP(mlp, xs, ys)
log("Predictions after training (NOT gate)")
for (let x of xs) {
    log(n.forward(x).data)
}







