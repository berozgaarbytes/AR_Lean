const videoElement = document.getElementById('video_input');
const canvasElement = document.getElementById('canvas_output');
const canvasCtx = canvasElement.getContext('2d');
const terminal = document.getElementById('terminal');

let currentFacingMode = "environment"; 
let camera = null;
let isPinching = false;
let æInput = "";

// Spatial Keyboard Config (Centered for Mobile)
const keys = [
    { label: "1", rx: 0.35, ry: 0.45 }, { label: "2", rx: 0.5, ry: 0.45 }, { label: "3", rx: 0.65, ry: 0.45 },
    { label: "4", rx: 0.35, ry: 0.55 }, { label: "5", rx: 0.5, ry: 0.55 }, { label: "6", rx: 0.65, ry: 0.55 },
    { label: "7", rx: 0.35, ry: 0.65 }, { label: "8", rx: 0.5, ry: 0.65 }, { label: "9", rx: 0.65, ry: 0.65 },
    { label: "CLR", rx: 0.35, ry: 0.75 }, { label: "0", rx: 0.5, ry: 0.75 }, { label: "GO", rx: 0.65, ry: 0.75 }
];

function onResults(results) {
    // FIX: Match canvas to visual viewport for mobile
    canvasElement.width = window.innerWidth;
    canvasElement.height = window.innerHeight;
    const w = canvasElement.width;
    const h = canvasElement.height;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, w, h);

    // 1. Render Reality (Camera)
    if (currentFacingMode === "user") {
        canvasCtx.translate(w, 0);
        canvasCtx.scale(-1, 1);
    }
    canvasCtx.drawImage(results.image, 0, 0, w, h);
    canvasCtx.restore();

    // 2. Render Æ Glass Display
    renderGlassDisplay(w, h);

    // 3. Render Floating Keypad
    renderKeypad(w, h);

    // 4. Hand Tracking & Collision
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const hand = results.multiHandLandmarks[0];
        const index = hand[8];
        const thumb = hand[4];

        // Finger Coordinates
        const fx = index.x * w;
        const fy = index.y * h;

        // Detect Depth/Pinch
        const pinchDist = Math.hypot(index.x - thumb.x, index.y - thumb.y);
        const active = pinchDist < 0.045;

        // Draw Spatial Cursor (with Glow)
        canvasCtx.shadowBlur = 15;
        canvasCtx.shadowColor = active ? "#0ff" : "#fff";
        canvasCtx.fillStyle = active ? "#0ff" : "rgba(255, 255, 255, 0.8)";
        canvasCtx.beginPath();
        canvasCtx.arc(fx, fy, 12, 0, Math.PI * 2);
        canvasCtx.fill();
        canvasCtx.shadowBlur = 0;

        if (active) {
            if (!isPinching) {
                processAetherInput(index.x, index.y);
                isPinching = true;
            }
        } else {
            isPinching = false;
        }
    }
}

function renderGlassDisplay(w, h) {
    canvasCtx.fillStyle = "rgba(0, 20, 20, 0.7)";
    canvasCtx.strokeStyle = "#0ff";
    canvasCtx.lineWidth = 3;
    
    // Search Bar
    roundRect(canvasCtx, w * 0.1, h * 0.1, w * 0.8, 70, 15, true, true);
    
    canvasCtx.fillStyle = "#0ff";
    canvasCtx.font = "bold 28px monospace";
    canvasCtx.fillText(`Æ > ${æInput}`, w * 0.15, h * 0.16);
}

function renderKeypad(w, h) {
    keys.forEach(key => {
        const kx = key.rx * w;
        const ky = key.ry * h;
        
        // Button Circle
        canvasCtx.fillStyle = "rgba(0, 255, 255, 0.1)";
        canvasCtx.strokeStyle = "rgba(0, 255, 255, 0.5)";
        canvasCtx.beginPath();
        canvasCtx.arc(kx, ky, 35, 0, Math.PI * 2);
        canvasCtx.fill();
        canvasCtx.stroke();
        
        // Label
        canvasCtx.fillStyle = "#fff";
        canvasCtx.font = "bold 20px sans-serif";
        canvasCtx.textAlign = "center";
        canvasCtx.fillText(key.label, kx, ky + 8);
    });
}

function processAetherInput(ix, iy) {
    keys.forEach(key => {
        const dist = Math.hypot(ix - key.rx, iy - key.ry);
        if (dist < 0.07) { // Interaction Radius
            if (key.label === "GO") {
                window.open(`https://www.google.com/search?q=${encodeURIComponent(æInput)}`, '_blank');
            } else if (key.label === "CLR") {
                æInput = "";
            } else {
                æInput += key.label;
            }
            if (navigator.vibrate) navigator.vibrate(40);
        }
    });
}

// Utility for Rounded Rectangles
function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    if (fill) ctx.fill();
    if (stroke) ctx.stroke();
}

// ... Camera start logic from previous responses ...
