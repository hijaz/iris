export class Neuron {
    constructor(numberOfInputs, activationFn) {
        this.weights = new Array(numberOfInputs)
            .fill(0)
            .map(() => Math.random() * 2 - 1);
        this.bias = Math.random() * 2 - 1;
        this.activationFunction = activationFn;
        this.lastZ = 0;
        this.lastActivation = 0;
        this.dL_db = 0;
        this.dL_dw = new Array(numberOfInputs).fill(0);
        this.delta = 0;
    }
    calculateOutput(inputs) {
        if (inputs.length !== this.weights.length) {
            throw new Error("Number of inputs must match number of weights in Neuron");
        }
        let weightedSum = 0;
        for (let i = 0; i < this.weights.length; i++) {
            weightedSum += this.weights[i] * inputs[i];
        }
        weightedSum += this.bias;
        this.lastZ = weightedSum;
        this.lastActivation = this.activationFunction(this.lastZ);
        return this.lastActivation;
    }
}
