export function meanSquaredError(predictions, targets) {
    if (predictions.length !== targets.length) {
        throw new Error("Predictions and targets must have the same length.");
    }
    if (predictions.length === 0) {
        return 0;
    }
    let sumSquaredError = 0;
    for (let i = 0; i < predictions.length; i++) {
        const error = targets[i] - predictions[i];
        sumSquaredError += error * error;
    }
    return sumSquaredError / predictions.length;
}