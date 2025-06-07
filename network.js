import { Layer } from './layer.js';
import { sigmoidDerivative, reluDerivative } from './activations.js';
export class Network {
    constructor(layerConfigurations) {
        this.layers = [];
        this.layerConfigs = layerConfigurations;
        for (const config of layerConfigurations) {
            this.layers.push(new Layer(config.numberOfNeurons, config.numberOfInputsPerNeuron, config.activationFn));
        }
    }
    predict(initialInputs) {
        let currentOutputs = initialInputs;
        const allLayerActivations = [];
        allLayerActivations.push([...initialInputs]);
        for (const layer of this.layers) {
            currentOutputs = layer.calculateOutputs(currentOutputs);
            allLayerActivations.push([...currentOutputs]);
        }
        return {
            finalOutput: currentOutputs,
            allLayerActivations: allLayerActivations
        };
    }
    backpropagate(targets, allLayerActivations) {
        const outputLayerIndex = this.layers.length - 1;
        const outputLayer = this.layers[outputLayerIndex];
        const outputLayerActivations = allLayerActivations[outputLayerIndex + 1];
        const prevLayerActivations = allLayerActivations[outputLayerIndex];
        outputLayer.neurons.forEach((neuron, j) => {
            const prediction = outputLayerActivations[j];
            const target = targets[j];
            const errorDerivative = prediction - target;
            const activationDerivative = sigmoidDerivative(neuron.lastActivation);
            neuron.delta = errorDerivative * activationDerivative;
            neuron.dL_db = neuron.delta;
            for (let k = 0; k < neuron.weights.length; k++) {
                neuron.dL_dw[k] = neuron.delta * prevLayerActivations[k];
            }
        });
        for (let i = this.layers.length - 2; i >= 0; i--) {
            const currentLayer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            const prevLayerActivationsForCurrent = allLayerActivations[i];
            currentLayer.neurons.forEach((neuron, k) => {
                let weightedErrorSum = 0;
                nextLayer.neurons.forEach((nextNeuron, j) => {
                    weightedErrorSum += nextNeuron.weights[k] * nextNeuron.delta;
                });
                let activationDerivative;
                activationDerivative = reluDerivative(neuron.lastZ);
                neuron.delta = weightedErrorSum * activationDerivative;
                neuron.dL_db = neuron.delta;
                for(let prev_k=0; prev_k < neuron.weights.length; prev_k++){
                    neuron.dL_dw[prev_k] = neuron.delta * prevLayerActivationsForCurrent[prev_k];
                }
            });
        }
    }
    updateParameters(learningRate) {
        for (const layer of this.layers) {
            for (const neuron of layer.neurons) {
                neuron.bias -= learningRate * neuron.dL_db;
                for (let i = 0; i < neuron.weights.length; i++) {
                    neuron.weights[i] -= learningRate * neuron.dL_dw[i];
                }
            }
        }
    }
}
