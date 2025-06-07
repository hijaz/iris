import { Neuron } from './neuron.js';
export class Layer {
    constructor(numberOfNeurons, numberOfInputsPerNeuron, activationFn) {
        this.neurons = [];
        for (let i = 0; i < numberOfNeurons; i++) {
            this.neurons.push(new Neuron(numberOfInputsPerNeuron, activationFn));
        }
    }
    calculateOutputs(inputs) {
        const layerActivations = [];
        for (const neuron of this.neurons) {
            const activation = neuron.calculateOutput(inputs);
            layerActivations.push(activation);
        }
        return layerActivations;
    }
}