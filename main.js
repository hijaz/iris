import { Network } from './network.js';
import { sigmoid, relu } from './activations.js';
import { initVisualization, updateNetworkActivations } from './nnVisualizer.js';
import { irisDataset } from './iris.js'; 
import { meanSquaredError } from './lossFunction.js';
import { initHUD, updateHUD, addHUDEventListeners } from './hud.js';

// --- Global State & Training Config ---
let myNetwork;
let currentDataIndex = 0;
let currentEpoch = 0;
let isTraining = false;
let learningRate;

const INITIAL_LEARNING_RATE = 0.1;
const MAX_EPOCHS = 100;

// --- Network and Data Configuration ---
const inputLayerSize = 4;
const networkArchitectureConfig = [
    { numberOfNeurons: 6, activationFn: relu },
    { numberOfNeurons: 3, activationFn: sigmoid } 
];

// --- Helper Functions ---
const speciesMap = { "Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2 };
const speciesToTarget = (speciesName) => {
    const target = [0, 0, 0];
    const index = speciesMap[speciesName];
    if (index !== undefined) {
        target[index] = 1;
    }
    return target;
};

const argMax = (array) => array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];

// --- Function Declarations ---

function createNetwork() {
    const configuredNetworkArchitecture = [];
    if (networkArchitectureConfig.length > 0) {
        configuredNetworkArchitecture.push({
            numberOfNeurons: networkArchitectureConfig[0].numberOfNeurons,
            activationFn: networkArchitectureConfig[0].activationFn,
            numberOfInputsPerNeuron: inputLayerSize
        });
        for (let i = 1; i < networkArchitectureConfig.length; i++) {
            configuredNetworkArchitecture.push({
                numberOfNeurons: networkArchitectureConfig[i].numberOfNeurons,
                activationFn: networkArchitectureConfig[i].activationFn,
                numberOfInputsPerNeuron: configuredNetworkArchitecture[i - 1].numberOfNeurons
            });
        }
    }
    return new Network(configuredNetworkArchitecture);
}

function runTrainingStep() {
    if (!isTraining) return;

    const dataPoint = irisDataset[currentDataIndex];
    const targetVector = speciesToTarget(dataPoint.species);
    
    const predictionResult = myNetwork.predict(dataPoint.features);
    
    if (predictionResult && predictionResult.allLayerActivations) {
        const loss = meanSquaredError(predictionResult.finalOutput, targetVector);
        updateHUD({ loss: loss, epoch: currentEpoch, learningRate: learningRate });
        myNetwork.backpropagate(targetVector, predictionResult.allLayerActivations);
        myNetwork.updateParameters(learningRate);
        updateNetworkActivations(myNetwork, inputLayerSize, predictionResult.allLayerActivations);
    }
    
    currentDataIndex++;
    if (currentDataIndex >= irisDataset.length) {
        currentDataIndex = 0;
        currentEpoch++;
        if (currentEpoch % 10 === 0) {
            learningRate *= 0.9;
        }
    }

    if (currentEpoch < MAX_EPOCHS) {
        requestAnimationFrame(runTrainingStep);
    } else {
        stopTrainingLoop("Training Complete!");
    }
}

function startTraining() {
    isTraining = true;
    const trainButton = document.getElementById('train-button');
    trainButton.innerText = 'Pause Training';
    console.log("Training started/resumed.");
    requestAnimationFrame(runTrainingStep);
}

function stopTrainingLoop(message = "Training Paused") {
    isTraining = false;
    const trainButton = document.getElementById('train-button');
    trainButton.innerText = 'Resume Training';
    console.log(message);
}

function handleTrainButtonClick() {
    if (isTraining) {
        stopTrainingLoop();
    } else {
        startTraining();
    }
}

function resetNetwork() {
    isTraining = false; 
    const trainButton = document.getElementById('train-button');
    trainButton.innerText = 'Start Training'; 

    console.log("--- NETWORK RESET ---");
    myNetwork = createNetwork();
    currentDataIndex = 0;
    currentEpoch = 0;
    learningRate = INITIAL_LEARNING_RATE;

    const initialData = irisDataset[0];
    const predictionResult = myNetwork.predict(initialData.features);
    const initialLoss = meanSquaredError(predictionResult.finalOutput, speciesToTarget(initialData.species));
    updateHUD({ loss: initialLoss, epoch: 0, learningRate: learningRate });
    updateNetworkActivations(myNetwork, inputLayerSize, predictionResult.allLayerActivations);
}

function testNetworkOnDataset() {
    console.log("--- RUNNING FULL DATASET TEST ---");
    let correctPredictions = 0;
    let totalLoss = 0;

    for (const dataPoint of irisDataset) {
        const targetVector = speciesToTarget(dataPoint.species);
        const trueIndex = speciesMap[dataPoint.species];

        const predictionResult = myNetwork.predict(dataPoint.features);
        const predictedIndex = argMax(predictionResult.finalOutput);

        if (predictedIndex === trueIndex) {
            correctPredictions++;
        }

        totalLoss += meanSquaredError(predictionResult.finalOutput, targetVector);
    }

    const accuracy = (correctPredictions / irisDataset.length) * 100;
    const averageLoss = totalLoss / irisDataset.length;

    const resultMessage = `Test Complete!\n` +
                          `-----------------\n` +
                          `Accuracy: ${accuracy.toFixed(2)}% (${correctPredictions} / ${irisDataset.length} correct)\n` +
                          `Average Loss: ${averageLoss.toFixed(4)}`;

    console.log(resultMessage);
    alert(resultMessage);
}

function showAboutModal() {
    const modal = document.getElementById('about-modal');
    if (modal) {
        modal.classList.remove('hidden');
    }
}

function hideAboutModal() {
    const modal = document.getElementById('about-modal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

function setupInitialState() {
    myNetwork = createNetwork(); // This will now work correctly
    learningRate = INITIAL_LEARNING_RATE;

    const initialData = irisDataset[0];
    const predictionResult = myNetwork.predict(initialData.features);
    const initialLoss = meanSquaredError(predictionResult.finalOutput, speciesToTarget(initialData.species));
    updateHUD({ loss: initialLoss, epoch: 0, learningRate: learningRate });
    initVisualization(myNetwork, inputLayerSize, predictionResult.allLayerActivations);

    addHUDEventListeners({
        onReset: resetNetwork,
        onTrain: handleTrainButtonClick,
        onTest: testNetworkOnDataset,
        onAbout: showAboutModal
    });

    const modal = document.getElementById('about-modal');
    const modalCloseBtn = document.getElementById('modal-close');
    if (modal) modal.addEventListener('click', hideAboutModal);
    if (modalCloseBtn) modalCloseBtn.addEventListener('click', hideAboutModal);
    
    const modalContent = document.querySelector('.modal-content');
    if(modalContent) modalContent.addEventListener('click', (e) => e.stopPropagation());
}

// --- INITIALIZATION ---
initHUD();
setupInitialState();