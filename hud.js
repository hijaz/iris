let lossValueElement, epochValueElement, lrValueElement;

/**
 * Creates the HUD container and appends it to the document body.
 * Should be called once at initialization.
 */
export function initHUD() {
    if (document.getElementById('hud-container')) return;
    const hudContainer = document.createElement('div');
    hudContainer.id = 'hud-container';

    // --- Loss Display ---
    const lossItem = document.createElement('div');
    lossItem.className = 'hud-item';
    const lossLabel = document.createElement('span');
    lossLabel.className = 'hud-label';
    lossLabel.innerText = 'Loss (MSE):';
    lossValueElement = document.createElement('span'); // Assign to module-scoped variable
    lossValueElement.className = 'hud-value';
    lossValueElement.innerText = '...';
    lossItem.appendChild(lossLabel);
    lossItem.appendChild(lossValueElement);
    hudContainer.appendChild(lossItem);

    // --- Epoch Display ---
    const epochItem = document.createElement('div');
    epochItem.className = 'hud-item';
    const epochLabel = document.createElement('span');
    epochLabel.className = 'hud-label';
    epochLabel.innerText = 'Epoch:';
    epochValueElement = document.createElement('span'); // Assign to module-scoped variable
    epochValueElement.className = 'hud-value';
    epochValueElement.innerText = '0';
    epochItem.appendChild(epochLabel);
    epochItem.appendChild(epochValueElement);
    hudContainer.appendChild(epochItem);
    
    // --- Learning Rate Display ---
    const lrItem = document.createElement('div');
    lrItem.className = 'hud-item';
    const lrLabel = document.createElement('span');
    lrLabel.className = 'hud-label';
    lrLabel.innerText = 'Learn Rate:';
    lrValueElement = document.createElement('span'); // Assign to module-scoped variable
    lrValueElement.className = 'hud-value';
    lrValueElement.innerText = '...';
    lrItem.appendChild(lrLabel);
    lrItem.appendChild(lrValueElement);
    hudContainer.appendChild(lrItem);

    // --- Buttons ---
    const buttonContainer = document.createElement('div');
    buttonContainer.style.marginTop = '15px';
    const resetButton = document.createElement('button');
    resetButton.id = 'reset-button';
    resetButton.innerText = 'Reset Network';
    const trainButton = document.createElement('button');
    trainButton.id = 'train-button';
    trainButton.innerText = 'Start Training';
    trainButton.style.marginLeft = '10px';
    const testButton = document.createElement('button');
    testButton.id = 'test-button';
    testButton.innerText = 'Test Full Dataset';
    testButton.style.marginTop = '5px';
    testButton.style.width = '100%';
    const aboutButton = document.createElement('button');
    aboutButton.id = 'about-button';
    aboutButton.innerText = 'About';
    aboutButton.style.marginTop = '5px';
    aboutButton.style.backgroundColor = '#555';
    buttonContainer.appendChild(resetButton);
    buttonContainer.appendChild(trainButton);
    hudContainer.appendChild(buttonContainer);
    hudContainer.appendChild(testButton); 
    hudContainer.appendChild(aboutButton);
    
    document.body.appendChild(hudContainer);
}

/**
 * Updates the data displayed in the HUD.
 * @param {object} data - An object containing the data to display. e.g., { loss: 0.123 }
 */
export function updateHUD(data) {
    // This now correctly uses the module-scoped variables set in initHUD
    if (data.loss !== undefined && lossValueElement) {
        lossValueElement.innerText = data.loss.toFixed(4);
    }
    if (data.epoch !== undefined && epochValueElement) {
        epochValueElement.innerText = `${data.epoch} / ${100}`; // Assuming MAX_EPOCHS is 100
    }
    if (data.learningRate !== undefined && lrValueElement) {
        lrValueElement.innerText = data.learningRate.toExponential(2);
    }
}

/**
 * Attaches event listeners to the HUD buttons.
 * @param {object} callbacks - An object containing callback functions. e.g., { onReset: func, onTrain: func }
 */
export function addHUDEventListeners(callbacks) {
    // ... (This function remains unchanged)
    const resetButton = document.getElementById('reset-button');
    if (resetButton && callbacks.onReset) resetButton.addEventListener('click', callbacks.onReset);
    
    const trainButton = document.getElementById('train-button');
    if (trainButton && callbacks.onTrain) trainButton.addEventListener('click', callbacks.onTrain);
    
    const testButton = document.getElementById('test-button');
    if (testButton && callbacks.onTest) testButton.addEventListener('click', callbacks.onTest);

    const aboutButton = document.getElementById('about-button');
    if (aboutButton && callbacks.onAbout) {
        aboutButton.addEventListener('click', callbacks.onAbout);
    }
}