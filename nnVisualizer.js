import * as THREE from 'three';

let scene, camera, renderer;
let networkGroup, textLabelGroup;
let cameraAngle = 0;

const NEURON_RADIUS = 0.15;
const LAYER_SPACING_Y = 1.75;
const LAYER_CIRCLE_RADIUS = 1.0;
const CAMERA_ORBIT_RADIUS = 8;

const BIAS_DISPLACEMENT_SCALE = 0.3; 
const WEIGHT_DISPLACEMENT_SCALE = 0.5; 

const TEXT_SIZE = 0.18;
const TEXT_COLOR_BIAS = '#FFFFFF';
const TEXT_COLOR_WEIGHT = '#FFFF99';
const TEXT_COLOR_ACTIVATION = '#99FF99';

const LAYER_ANIMATION_DELAY = 300; 
const LAYER_HIGHLIGHT_DURATION = 250; 
const HIGHLIGHT_EMISSIVE_COLOR = 0xffaa00; 
const DEFAULT_EMISSIVE_INTENSITY = 0.4;

let neuronMeshesByLayer = []; 
let originalNeuronMaterials = new Map(); 

function makeTextSprite(message, x, y, z, size = TEXT_SIZE, color = TEXT_COLOR_BIAS, backColor = null) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const fontface = "Arial";
    const fontsize = 32;
    context.font = "Bold " + fontsize + "px " + fontface;
    const metrics = context.measureText(message);
    const textWidth = metrics.width;
    const actualHeight = Math.max(fontsize, metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent);
    const canvasPaddingX = 20;
    const canvasPaddingY = 10;
    canvas.width = THREE.MathUtils.ceilPowerOfTwo(textWidth + canvasPaddingX);
    canvas.height = THREE.MathUtils.ceilPowerOfTwo(actualHeight + canvasPaddingY);
    context.font = "Bold " + fontsize + "px " + fontface;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    if (backColor) {
        context.fillStyle = backColor;
        context.fillRect(0, 0, canvas.width, canvas.height);
    }
    context.fillStyle = color;
    context.fillText(message, canvas.width / 2, canvas.height / 2);
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    const spriteMaterial = new THREE.SpriteMaterial({
        map: texture,
        depthTest: false,
        transparent: true
    });
    const sprite = new THREE.Sprite(spriteMaterial);
    const aspectRatio = canvas.width / canvas.height;
    sprite.scale.set(size * aspectRatio, size, 1);
    sprite.position.set(x, y, z);
    return sprite;
}

function drawNetwork(network, numInputs, allActivations = []) {
    networkGroup.clear();
    textLabelGroup.clear();
    networkGroup.add(textLabelGroup);

    neuronMeshesByLayer = []; 
    originalNeuronMaterials.clear(); 

    let allNeuronPositions = [];

    const neuronGeometry = new THREE.SphereGeometry(NEURON_RADIUS, 16, 16);
    const baseInputMaterial = new THREE.MeshStandardMaterial({ color: 0xffdd44, metalness: 0.3, roughness: 0.5, emissive: 0x000000, emissiveIntensity: DEFAULT_EMISSIVE_INTENSITY });
    const baseHiddenMaterial = new THREE.MeshStandardMaterial({ color: 0x44bbff, metalness: 0.3, roughness: 0.5, emissive: 0x000000, emissiveIntensity: DEFAULT_EMISSIVE_INTENSITY });
    const baseOutputMaterial = new THREE.MeshStandardMaterial({ color: 0xff8888, metalness: 0.3, roughness: 0.5, emissive: 0x000000, emissiveIntensity: DEFAULT_EMISSIVE_INTENSITY });
    
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xaaaaaa, transparent: true, opacity: 0.7 });

    let tempY = 0;
    let layerPositions = [];

    const processLayer = (neuronCount, yPos, baseMat, layerData = null, isInputLayer = false, currentLayerIndexInAllActivations = 0) => {
        if (neuronCount === 0) return;
        let currentLayerNeuronMeshes = [];
        let currentLayerNeuronPositions = [];
        const baseRadius = neuronCount <= 1 ? 0 : LAYER_CIRCLE_RADIUS;
        const angleStep = neuronCount > 1 ? (2 * Math.PI) / neuronCount : 0;

        for (let i = 0; i < neuronCount; i++) {
            const angle = i * angleStep;
            
            let finalX = 0, finalY = yPos, finalZ = 0;
            
            // Base Position
            const baseX = baseRadius * Math.cos(angle);
            const baseZ = baseRadius * Math.sin(angle);

            if (!isInputLayer && layerData && layerData.neurons[i]) {
                const neuron = layerData.neurons[i];
                // Vertical displacement based on bias
                finalY = yPos + (neuron.bias * BIAS_DISPLACEMENT_SCALE);
                
                // Radial displacement based on average absolute weight
                const sumAbsWeights = neuron.weights.reduce((sum, w) => sum + Math.abs(w), 0);
                const avgAbsWeight = neuron.weights.length > 0 ? sumAbsWeights / neuron.weights.length : 0;
                const displacedRadius = baseRadius + (avgAbsWeight * WEIGHT_DISPLACEMENT_SCALE);

                finalX = displacedRadius * Math.cos(angle);
                finalZ = displacedRadius * Math.sin(angle);
            } else {
                finalX = baseX;
                finalZ = baseZ;
            }

            const neuronMaterial = baseMat.clone();
            let activationValue = 0;

            if (allActivations[currentLayerIndexInAllActivations] && allActivations[currentLayerIndexInAllActivations][i] !== undefined) {
                activationValue = allActivations[currentLayerIndexInAllActivations][i];
                const intensity = Math.min(1, Math.max(0, activationValue)) * 0.7; 
                neuronMaterial.emissive.set(neuronMaterial.color); 
                neuronMaterial.emissiveIntensity = intensity + 0.1; 
            }
            
            const neuronMesh = new THREE.Mesh(neuronGeometry, neuronMaterial);
            neuronMesh.position.set(finalX, finalY, finalZ); // Use final positions
            networkGroup.add(neuronMesh);
            currentLayerNeuronMeshes.push(neuronMesh);
            currentLayerNeuronPositions.push(new THREE.Vector3(finalX, finalY, finalZ)); // Store final positions for lines
            originalNeuronMaterials.set(neuronMesh, neuronMaterial); 

            // Text for Activation
            const actSprite = makeTextSprite(`A: ${activationValue.toFixed(2)}`, finalX, finalY - NEURON_RADIUS - TEXT_SIZE * 0.7, finalZ, TEXT_SIZE * 0.8, TEXT_COLOR_ACTIVATION);
            textLabelGroup.add(actSprite);

            if (!isInputLayer && layerData && layerData.neurons[i]) {
                const bias = layerData.neurons[i].bias;
                const biasSprite = makeTextSprite(`B: ${bias.toFixed(2)}`, finalX, finalY + NEURON_RADIUS + TEXT_SIZE * 0.7, finalZ, TEXT_SIZE * 0.8, TEXT_COLOR_BIAS);
                textLabelGroup.add(biasSprite);
            }
        }
        neuronMeshesByLayer.push(currentLayerNeuronMeshes);
        layerPositions.push(currentLayerNeuronPositions);
    };

    if (numInputs > 0) {
        processLayer(numInputs, tempY, baseInputMaterial, null, true, 0); 
        if (network.layers.length > 0) tempY += LAYER_SPACING_Y;
    }
    network.layers.forEach((layer, layerIndex) => {
        let material = baseHiddenMaterial;
        if (layerIndex === network.layers.length - 1) material = baseOutputMaterial;
        processLayer(layer.neurons.length, tempY, material, layer, false, layerIndex + (numInputs > 0 ? 1 : 0) );
        if (layerIndex < network.layers.length - 1) tempY += LAYER_SPACING_Y;
    });
    allNeuronPositions = layerPositions; 
    if (allNeuronPositions.length > 1) {
        for (let i = 0; i < allNeuronPositions.length - 1; i++) {
            const sourceNeuronPositions = allNeuronPositions[i];
            const targetNeuronPositions = allNeuronPositions[i + 1];
            const targetLayerData = network.layers[i]; 
            sourceNeuronPositions.forEach((startPos, sourceIndex) => {
                targetNeuronPositions.forEach((endPos, targetIndexInLayer) => {
                    const points = [startPos, endPos];
                    const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
                    const line = new THREE.Line(lineGeometry, lineMaterial);
                    networkGroup.add(line);
                    if (targetLayerData && targetLayerData.neurons[targetIndexInLayer]) {
                        const weight = targetLayerData.neurons[targetIndexInLayer].weights[sourceIndex];
                        if (weight !== undefined) {
                            const midPoint = new THREE.Vector3().addVectors(startPos, endPos).multiplyScalar(0.5);
                            const weightSprite = makeTextSprite(`W: ${weight.toFixed(2)}`, midPoint.x, midPoint.y + TEXT_SIZE * 0.4, midPoint.z, TEXT_SIZE * 0.7, TEXT_COLOR_WEIGHT);
                            textLabelGroup.add(weightSprite);
                        }
                    }
                });
            });
        }
    }
    const firstLayerY = 0;
    const lastLayerY = tempY;
    const networkMidY = (firstLayerY + lastLayerY) / 2;
    networkGroup.position.y = (numInputs > 0 || network.layers.length > 0) ? -networkMidY : 0;
}


let animationTimeoutId = null; 

function animateActivationFlow() { 
    if (animationTimeoutId) {
        clearTimeout(animationTimeoutId); 
    }
    let currentLayerToAnimate = 0;
    function highlightNextLayer() {
        if (currentLayerToAnimate > 0 && neuronMeshesByLayer[currentLayerToAnimate - 1]) {
            neuronMeshesByLayer[currentLayerToAnimate - 1].forEach(mesh => {
                mesh.material = originalNeuronMaterials.get(mesh) || mesh.material; 
            });
        }
        if (currentLayerToAnimate < neuronMeshesByLayer.length && neuronMeshesByLayer[currentLayerToAnimate]) {
            neuronMeshesByLayer[currentLayerToAnimate].forEach(mesh => {
                const highlightMaterial = mesh.material.clone();
                highlightMaterial.emissive.setHex(HIGHLIGHT_EMISSIVE_COLOR);
                highlightMaterial.emissiveIntensity = 1.0;
                mesh.material = highlightMaterial;
            });
            animationTimeoutId = setTimeout(() => {
                 if (neuronMeshesByLayer[currentLayerToAnimate]) {
                    neuronMeshesByLayer[currentLayerToAnimate].forEach(mesh => {
                         mesh.material = originalNeuronMaterials.get(mesh) || mesh.material;
                    });
                }
                currentLayerToAnimate++;
                if (currentLayerToAnimate < neuronMeshesByLayer.length) {
                    highlightNextLayer();
                } else {
                    animationTimeoutId = null; 
                }
            }, LAYER_HIGHLIGHT_DURATION); 
        } else {
             animationTimeoutId = null; 
        }
    }
    highlightNextLayer(); 
}

function animate() { 
    requestAnimationFrame(animate);
    cameraAngle += 0.005;
    camera.position.x = CAMERA_ORBIT_RADIUS * Math.sin(cameraAngle);
    camera.position.z = CAMERA_ORBIT_RADIUS * Math.cos(cameraAngle);
    camera.position.y = CAMERA_ORBIT_RADIUS * 0.3;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
}

function onWindowResize() { 
    if (camera && renderer) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

export function initVisualization(networkInstance, numInputs, initialActivations) { 
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1A1A1A);
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, CAMERA_ORBIT_RADIUS * 0.3, CAMERA_ORBIT_RADIUS);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    const existingCanvas = document.querySelector('canvas');
    if (existingCanvas) {
        existingCanvas.remove();
    }
    document.body.appendChild(renderer.domElement);
    networkGroup = new THREE.Group();
    textLabelGroup = new THREE.Group();
    scene.add(networkGroup);
    networkGroup.add(textLabelGroup);
    drawNetwork(networkInstance, numInputs, initialActivations);
    animateActivationFlow(); 
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(3, 5, 4);
    scene.add(directionalLight);
    window.addEventListener('resize', onWindowResize, false);
    onWindowResize(); 
    if (!renderer.info.render.frame) { 
        animate();
    }
}

export function updateNetworkActivations(networkInstance, numInputs, newActivations) { 
    if (!scene || !networkGroup || !textLabelGroup) {
        console.error("Visualization not initialized.");
        return;
    }
    drawNetwork(networkInstance, numInputs, newActivations); 
    animateActivationFlow(); 
}