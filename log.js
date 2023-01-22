// Log the graph of a node
function logNodes(node) {

    let log = []
    
    // Format number to 4 decimals
    function format(x) {
        return parseFloat(x).toFixed(4)
    }

    // topological order all of the children in the graph
    let topo = []
    let seen = new Set()

    let build_topo = (node, level) => {
        if (seen.has(node)) {
            return
        }

        seen.add(node)
        let debug = {
            'level': level,
            'label': node.label,
            'data': format(node.data),
            'grad': format(node.grad),
            'op': node._op,
            'prev_length': node._prev.length,
            'prev': node._prev.map(x => format(x.data))
        }
        log.push(debug)

        for (let child of node._prev) {
            build_topo(child, level + 1)
        }

        topo.push(node)
    }

    build_topo(node, 0)
    node.grad = 1.0

    for (let node of topo.reverse()) {
        node._backward()
    }

    // sort array of objects by level
    log.sort((a, b) => a.level - b.level)
    console.table(log)
}

export {logNodes}