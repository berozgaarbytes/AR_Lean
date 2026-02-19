const videoElement = document.getElementById('video_input');
const canvasElement = document.getElementById('canvas_output');
const canvasCtx = canvasElement.getContext('2d');
const terminal = document.getElementById('terminal');

let currentFacingMode = "environment"; 
let camera = null;
let isPinching = false;
let æInput = "";

// Define Virtual Keys (Spatial Layout)
const keys = [
    { label: "1", x: 0.3, y: 0.4 }, { label: "2", x: 0.5, y: 0.4 }, { label: "3", x: 0.7, y: 0.4 },
    { label: "4", x: 0.3, y: 0.55 }, { label: "5", x: 0.5, y: 0.55 }, { label: "6", x: 0.7, y: 0.55 },
    { label: "7", x: 0.3, y: 0.7 }, { label: "8", x: 0.5, y: 0.7 }, { label: "9", x: 0.7, y: 0.7 },
    { label: "CLR", x: 0.3, y: 0.85 }, { label: "0", x: 0.5, y: 0.85 }, { label: "GO", x: 0.7, y: 0.85 }
];

function drawAetherUI(results) {
    const w = canvasElement.width;
    const h = canvasElement.height;

    // 1. Draw Glass Display (Search Bar)
    canvasCtx.fillStyle = "rgba(0, 40, 40, 0.5)";
    canvasCtx.roundRect(w * 0.2, h * 0.1, w * 0.6, 60, 15);
    canvasCtx.fill();
    canvasCtx.strokeStyle = "#0ff";
    canvasCtx.stroke();
    
    canvasCtx.fillStyle = "#fff";
    canvasCtx.font = "bold 24px monospace";
    canvasCtx.fillText(`DATA: ${æInput}`, w * 0.25, h * 0.16);

    // 2. Draw Floating Keypad
    keys.forEach(key => {
        const kX = key.x * w;
        const kY = key.y * h;
        const size = 60;

        // Visual "Hover" effect (Logic handled in onResults)
        canvasCtx.fillStyle = "rgba(0, 255, 255, 0.15)";
        canvasCtx.strokeStyle = "rgba(0, 255, 255, 0.8)";
        canvasCtx.beginPath();
        canvasCtx.arc(kX, kY, size/2, 0, 2 * Math.PI);
        canvasCtx.fill();
        canvasCtx.stroke();

        canvasCtx.fillStyle = "#0ff";
        canvasCtx.font = "20px monospace";
        canvasCtx.fillText(key.label, kX - 10, kY + 7);
    });
}

function onResults(results) {
    canvasElement.width = window.innerWidth;
    canvasElement.height = window.innerHeight;
    const w = canvasElement.width;
    const h = canvasElement.height;

    canvasCtx.save();
    if (currentFacingMode === "user") {
        canvasCtx.translate(w, 0);
        canvasCtx.scale(-1, 1);
    }
    canvasCtx.drawImage(results.image, 0, 0, w, h);
    canvasCtx.restore();

    drawAetherUI(results);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const hand = results.multiHandLandmarks[0];
        const index = hand[8];
        const thumb = hand[4];

        // Finger Cursor
        canvasCtx.fillStyle = "white";
        canvasCtx.beginPath();
        canvasCtx.arc(index.x * w, index.y * h, 10, 0, Math.PI * 2);
        canvasCtx.fill();

        // Detect Pinch
        const dist = Math.hypot(index.x - thumb.x, index.y - thumb.y);
        if (dist < 0.04) {
            if (!isPinching) {
                checkKeyCollision(index.x, index.y);
                isPinching = true;
                terminal.innerText = "Æ: PINCH DETECTED";
            }
            // Draw connection line
            canvasCtx.strokeStyle = "#0ff";
            canvasCtx.beginPath();
            canvasCtx.moveTo(index.x * w, index.y * h);
            canvasCtx.lineTo(thumb.x * w, thumb.y * h);
            canvasCtx.stroke();
        } else {
            isPinching = false;
        }
    }
}

function checkKeyCollision(ix, iy) {
    keys.forEach(key => {
        const d = Math.hypot(ix - key.x, iy - key.y);
        if (d < 0.08) { // Collision threshold
            if (key.label === "GO") {
                window.open(`https://www.google.com/search?q=${æInput}`, '_blank');
            } else if (key.label === "CLR") {
                æInput = "";
            } else {
                æInput += key.label;
            }
            // Trigger Haptic Feedback (if supported)
            if (navigator.vibrate) navigator.vibrate(50);
        }
    });
}

// ... (Rest of Camera and Mediapipe Setup from previous script) ...
