# micrograd-js

Made for fun in order to try and understand a deep learning library from scratch.

All the code is written using the "The spelled-out intro to neural networks and backpropagation: building micrograd" by Andrej Karpathy

The tutorial can be found on https://www.youtube.com/watch?v=VMj-3S1tku0

This is written in javascript in order "to type it out" myself - in another language.

The original code is written in python which is a better language for the purpose. 

But I wanted to try it out in javascript.

## Requirements

nodejs >= 14.17.4

I know it won't work with nodejs <= v12.22.4

    nvm use v14.17.4

## Example

This runs the main example from the tutorial:

    node test/mlp.js

[mlp.js](test/mlp.js)

## Install 

    npm install micrograd-js

## Usage

```javascript
import { Value } from 'micrograd-js'
import { MLP } from 'micrograd-js'
```

## License

MIT © [Dennis Iversen](https://github.com/diversen)
