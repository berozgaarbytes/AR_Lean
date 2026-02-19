const videoElement = document.getElementById('video_input');
const canvasElement = document.getElementById('canvas_output');
const canvasCtx = canvasElement.getContext('2d');
const terminal = document.getElementById('terminal');

let currentFacingMode = "environment"; // Starts with BACK camera
let camera = null;

// Æ UI State
let searchString = "GOOGLE: ";
let isPinching = false;

function onResults(results) {
    canvasElement.width = window.innerWidth;
    canvasElement.height = window.innerHeight;
    
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // 1. Draw Reality (Camera Feed)
    if (currentFacingMode === "user") {
        canvasCtx.translate(canvasElement.width, 0);
        canvasCtx.scale(-1, 1);
    }
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.restore();

    // 2. Render Æ Virtual UI (Floating Search Bar)
    drawVirtualUI();

    // 3. Hand Interaction Logic
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const hand = results.multiHandLandmarks[0];
        const indexTip = hand[8];
        const thumbTip = hand[4];

        // Convert normalized coordinates to pixels
        const x = indexTip.x * canvasElement.width;
        const y = indexTip.y * canvasElement.height;

        // Detect "Pinch" Gesture (Pinch distance < 0.05)
        const distance = Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y);
        
        if (distance < 0.05) {
            if (!isPinching) { // Trigger once per pinch
                handleAetherClick(x, y);
                isPinching = true;
            }
            canvasCtx.fillStyle = "#0ff"; // Glow cyan on pinch
        } else {
            isPinching = false;
            canvasCtx.fillStyle = "#fff";
        }

        // Draw Spatial Cursor
        canvasCtx.beginPath();
        canvasCtx.arc(x, y, 15, 0, 2 * Math.PI);
        canvasCtx.fill();
    }
}

function drawVirtualUI() {
    // Floating Search Box
    canvasCtx.fillStyle = "rgba(0, 0, 0, 0.6)";
    canvasCtx.fillRect(50, 50, canvasElement.width - 100, 60);
    canvasCtx.strokeStyle = "#0ff";
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeRect(50, 50, canvasElement.width - 100, 60);
    
    canvasCtx.fillStyle = "#0ff";
    canvasCtx.font = "20px monospace";
    canvasCtx.fillText(searchString + (Math.floor(Date.now()/500)%2 ? "_" : ""), 70, 88);
}

function handleAetherClick(x, y) {
    terminal.innerText = `Æ Event: Spatial Click at ${Math.round(x)}, ${Math.round(y)}`;
    
    // If click is in the top header, trigger Google
    if (y < 150) {
        const query = prompt("Æ Input: Enter Search Query");
        if (query) {
            window.open(`https://www.google.com/search?q=${encodeURIComponent(query)}`, '_blank');
        }
    }
}

function startCamera() {
    if (camera) camera.stop();
    
    camera = new Camera(videoElement, {
        onFrame: async () => {
            await hands.send({image: videoElement});
        },
        facingMode: currentFacingMode,
        width: 1280,
        height: 720
    });
    camera.start();
    terminal.innerText = `Æ System: ${currentFacingMode.toUpperCase()} Camera Active`;
}

function toggleCamera() {
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    startCamera();
}

const hands = new Hands({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});
hands.onResults(onResults);

startCamera();
