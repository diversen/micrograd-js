class Value {

    constructor(data, _children = [], _op = '', label = '') {
        this.data = data
        this.grad = 0.0
        this._backward = () => {
            return null;
        };
        this._prev = [...new Set(_children)]
        this._op = _op
        this.label = label
    }

    add(other) {

        if (!(other instanceof Value)) {
            other = new Value(other)
        }

        let out = new Value(this.data + other.data, [this, other], '+');
        out._backward = () => {
            this.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        }

        return out
    }

    sub(other) {

        if (!(other instanceof Value)) {
            other = new Value(other)
        }

        return this.add(other.mul(-1.0))
    }

    mul(other) {

        if (!(other instanceof Value)) {
            other = new Value(other)
        }

        let out = new Value(this.data * other.data, [this, other], '*');
        out._backward = () => {
            this.grad += other.data * out.grad
            other.grad += this.data * out.grad
        }

        return out
    }

    pow(other) {

        if (typeof other !== 'number') {
            throw new Error('Can only raise to a number')
        }

        let out = new Value(this.data ** other, [this], `**${other}`);
        out._backward = () => {
            this.grad += other * (this.data ** (other - 1)) * out.grad
        }

        return out
    }


    truediv(other) {

        if (!(other instanceof Value)) {
            other = new Value(other)
        }

        let out = new Value(this.data * other.data ** -1, [this, other], '/');
        out._backward = () => {
            this.grad += 1.0 / other.data * out.grad
            other.grad += -this.data / (other.data ** 2) * out.grad
        }

        return out

    }

    tanh() {
        // Write tanh in terms of e exp
        let x = this.data
        let t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1)
        let out = new Value(t, [this], 'tanh')

        out._backward = () => {
            this.grad += (1 - t ** 2) * out.grad
        }

        return out
    }

    exp() {
        // Write tanh in terms of e exp
        let x = this.data
        let t = Math.exp(x)
        let out = new Value(t, [this], 'exp')

        out._backward = () => {
            this.grad += out.data * out.grad
        }

        return out
    }

    backward() {

        // topological order all of the children in the graph
        let topo = []
        let seen = new Set()

        let build_topo = (node) => {
            if (seen.has(node)) {
                return
            }

            seen.add(node)

            for (let child of node._prev) {
                build_topo(child)
            }

            topo.push(node)
        }

        build_topo(this)
        this.grad = 1.0

        for (let node of topo.reverse()) {
            node._backward()
        }
    }
}

export { Value }