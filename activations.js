export function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
export function relu(x) {
    return Math.max(0, x);
}
export function sigmoidDerivative(outputValue) {
    return outputValue * (1 - outputValue);
}
export function reluDerivative(inputValue) {
    return inputValue > 0 ? 1 : 0;
}