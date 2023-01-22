import { Value } from './engine.js';

function random () {
    return Math.random() * 2 - 1;
}

class Neuron {

    /**
     * @param {Number} nin number of inputs to this neuron
     */
    constructor(nin) {
        
        this.w = [];
        for (let i = 0; i < nin; i++) {
            this.w.push(new Value(random()));
        }

        this.b = new Value(random());
    }

    /**
     * Forward pass through neuron
     * @param {Array} x values to be dotted with weights in neuron 
     * @returns {Value} output of neuron
     */
    forward (x) {

        let act = new Value(0.0);
        // wx + b
        for (let i = 0; i < this.w.length; i++) {
            act = act.add(this.w[i].mul(x[i]));
        }
        return act.add(this.b).tanh();
    }

    /**
     * Get all parameters in neuron
     * @returns {Array} all parameters in neuron
     */
    parameters() {
        return this.w.concat([this.b]);
    }
}

class Layer {

    /**
     * @param {Number} nin number of inputs to this layer 
     * @param {Number} nout number of outputs from this layer
     */
    constructor (nin, nout) {
        this.neurons = [];
        for (let i = 0; i < nout; i++) {
            this.neurons.push(new Neuron(nin));
        }
    }

    /**
     * Forward pass through layer
     * @param {Array} x values to be dotted with weights in layer 
     * @returns {Array} outputs of layer 
     */
    forward (x) {
        let outs = [];
        for (let i = 0; i < this.neurons.length; i++) {
            outs.push(this.neurons[i].forward(x));
        }
        
        // If only one neuron, return it directly
        if (outs.length === 1) {
            return outs[0];
        }
        return outs;
    }

    /**
     * Get all parameters in layer
     * @returns {Array} all parameters in layer
     */
    parameters() {
        let params = [];
        this.neurons.forEach(neuron => {
            params = params.concat(neuron.parameters());
        });
        return params;
    }
}

class MLP {

    /**
     * let n = new MLP(3, [4, 4, 1])
     * @param {Number} Number of inputs 
     * @param {Array} Array of sizes of layers []
     */
    constructor(nin, nouts) {
        let nz = [nin].concat(nouts);
        
        this.layers = [];
        for (let i = 0; i < nouts.length; i++) {
            this.layers.push(new Layer(nz[i], nz[i+1]));
        }
    }

    /**
     * Forward pass through MLP
     * @param {Array} x 
     * @returns {Array} output of MLP
     */
    forward (x) {

        this.layers.forEach(layer => {
            x = layer.forward(x);          
        });
        return x;
    }

    /**
     * Get all parameters in MLP
     * @returns {Array} all parameters in MLP
     */
    parameters() {
        let params = [];
        this.layers.forEach(layer => {
            params = params.concat(layer.parameters());
        });
        return params;
    }
}

export { Neuron, Layer, MLP };