import { Value } from '../engine.js'
import { MLP } from '../neuron.js'

/**
 * This is the main example used in the tutorial
 */
let log = console.log
let n = new MLP(3, [4, 4, 1])

let xs = [
    [2.0, 3.0, -1.0],
    [3.0,-1.0,  0.5],
    [0.5, 1.0,  1.0],
    [1.0, 1.0, -1.0], 
]

let ys = [1.0,-1.0, -1.0, 1.0]
let last_pred = []

function train(round) {

    log("Training round: " + round)

    for (let i = 0; i < 20; i++) {

        let ypred = []
        xs.forEach(x => {
            ypred.push(n.forward(x))
        })

        let loss_total = new Value(0.0, [], '', 'loss_total');    
        for (let i = 0; i < ys.length; i++) {
            let loss = new Value(ys[i], [], '', 'loss');
            loss = loss.sub(ypred[i]).pow(2)
            loss_total = loss_total.add(loss)
        }

        for (let p of n.parameters()) {
            p.grad = 0.0
        }

        loss_total.backward()

        for (let p of n.parameters()) {
            p.data += -0.05 * p.grad
        }

        log(i, loss_total.data)
        last_pred = ypred

    }
}

for (let i = 0; i < 4; i++) {
    train(i)
}

log("Predictions after training")
for (let p of last_pred) {
    log(p.data)
}