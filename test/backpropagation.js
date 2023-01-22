import { Value, logNodes } from '../index.js'

function backpropagation() {

    // Test implementation of the backpropagation
    let x1 = new Value(2.0, [], '', 'x1')
    let x2 = new Value(0.0, [], '', 'x2')

    let w1 = new Value(-3.0, [], '', 'w1')
    let w2 = new Value(1.0, [], '', 'w2')

    let b = new Value(6.8813735870195432, [], '', 'b')

    let x1w1 = x1.mul(w1); x1w1.label = 'x1*w1'
    let x2w2 = x2.mul(w2); x2w2.label = 'x2*w2'

    let x1w1x2w2 = x1w1.add(x2w2); x1w1x2w2.label = 'x1w1 + x2w2'

    let n = x1w1x2w2.add(b); n.label = 'n'
    // let o = n.tanh(); o.label = 'o'

    let nx2 = n.mul(2.0); nx2.label = 'nx2'
    let e = nx2.exp(); e.label = 'e'
    
    let nom = e.sub(1); nom.label = 'nom'
    let den = e.add(1); den.label = 'den'

    let o = nom.truediv(den); o.label = 'o'
    
    o.backward()
    logNodes(o)

}

backpropagation()

