const videoElement = document.getElementById('video_input');
const canvasElement = document.getElementById('canvas_output');
const canvasCtx = canvasElement.getContext('2d');
const hud = document.getElementById('hud_overlay');

let currentFacingMode = "environment";
let camera = null;
let isPinching = false;
let qrScanActive = false;
let æInput = "";

// Spatial Desktop Configuration
const layout = {
    browser: { label: "ACUMATICA_ERP_PORTAL", rx: 0.1, ry: 0.1, rw: 0.8, rh: 0.35, color: "#0ff" },
    terminal: { label: "SYSTEM_LOGS", rx: 0.1, ry: 0.5, rw: 0.35, rh: 0.4, color: "#0f0" },
    keypad: { label: "SPATIAL_INPUT", rx: 0.55, ry: 0.5, rw: 0.35, rh: 0.4, color: "#ff0" }
};

const keys = [
    { label: "1", x: 0.62, y: 0.6 }, { label: "2", x: 0.72, y: 0.6 }, { label: "3", x: 0.82, y: 0.6 },
    { label: "4", x: 0.62, y: 0.7 }, { label: "5", x: 0.72, y: 0.7 }, { label: "6", x: 0.82, y: 0.7 },
    { label: "7", x: 0.62, y: 0.8 }, { label: "8", x: 0.72, y: 0.8 }, { label: "9", x: 0.82, y: 0.8 },
    { label: "ERP", x: 0.72, y: 0.88, special: true }
];

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

    // 1. Process QR Scanning if active
    if (qrScanActive) scanForQR(results.image);

    // 2. Render Æ Spatial Desktop
    renderWindows(w, h);
    renderKeys(w, h);

    // 3. Hand Interaction Logic
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const hand = results.multiHandLandmarks[0];
        const index = hand[8];
        const thumb = hand[4];

        const pinchDist = Math.hypot(index.x - thumb.x, index.y - thumb.y);
        const active = pinchDist < 0.045;

        // Draw Spatial Pointer
        canvasCtx.fillStyle = active ? "#0ff" : "white";
        canvasCtx.beginPath();
        canvasCtx.arc(index.x * w, index.y * h, 10, 0, Math.PI * 2);
        canvasCtx.fill();

        if (active && !isPinching) {
            handleSpatialClick(index.x, index.y);
            isPinching = true;
        } else if (!active) {
            isPinching = false;
        }
    }
}

function renderWindows(w, h) {
    Object.values(layout).forEach(win => {
        const x = win.rx * w; const y = win.ry * h;
        const width = win.rw * w; const height = win.rh * h;

        canvasCtx.fillStyle = "rgba(0, 10, 10, 0.85)";
        canvasCtx.strokeStyle = win.color;
        canvasCtx.lineWidth = 2;
        canvasCtx.fillRect(x, y, width, height);
        canvasCtx.strokeRect(x, y, width, height);

        // Header Bar
        canvasCtx.fillStyle = win.color;
        canvasCtx.fillRect(x, y - 20, width, 20);
        canvasCtx.fillStyle = "#000";
        canvasCtx.font = "bold 10px monospace";
        canvasCtx.fillText(win.label, x + 5, y - 7);
    });

    // ERP Content Simulation
    canvasCtx.fillStyle = "#fff";
    canvasCtx.font = "12px monospace";
    canvasCtx.fillText(`ACTIVE_SESSION: ADMIN_01`, w*0.12, h*0.18);
    canvasCtx.fillText(`INPUT_BUFFER: ${æInput}`, w*0.12, h*0.22);
}

function renderKeys(w, h) {
    keys.forEach(k => {
        canvasCtx.strokeStyle = "#ff0";
        canvasCtx.fillStyle = "rgba(255, 255, 0, 0.1)";
        canvasCtx.beginPath();
        canvasCtx.arc(k.x * w, k.y * h, 25, 0, Math.PI * 2);
        canvasCtx.stroke();
        canvasCtx.fill();
        canvasCtx.fillStyle = "#ff0";
        canvasCtx.fillText(k.label, k.x * w - 8, k.y * h + 5);
    });
}

function handleSpatialClick(ix, iy) {
    keys.forEach(k => {
        const dist = Math.hypot(ix - k.x, iy - k.y);
        if (dist < 0.06) {
            if (k.label === "ERP") {
                window.open(`https://www.google.com/search?q=Acumatica+Entry+${æInput}`, '_blank');
            } else {
                æInput += k.label;
            }
            if (navigator.vibrate) navigator.vibrate(30);
        }
    });
}

function scanForQR(image) {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = image.width;
    tempCanvas.height = image.height;
    tempCtx.drawImage(image, 0, 0);
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const code = jsQR(imageData.data, imageData.width, imageData.height);
    
    if (code) {
        qrScanActive = false;
        hud.innerText = `Æ_SCAN_SUCCESS: ${code.data}`;
        if (confirm(`Open ERP Work Order: ${code.data}?`)) {
            window.open(code.data, '_blank');
        }
    }
}

function toggleCamera() {
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    startCamera();
}

function toggleQR() {
    qrScanActive = !qrScanActive;
    hud.innerText = qrScanActive ? "Æ_STATUS: SCANNING_FOR_QR..." : "Æ_STATUS: STANDBY";
}

function startCamera() {
    if (camera) camera.stop();
    camera = new Camera(videoElement, {
        onFrame: async () => { await hands.send({image: videoElement}); },
        facingMode: currentFacingMode,
        width: 1280, height: 720
    });
    camera.start();
}

const hands = new Hands({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.7 });
hands.onResults(onResults);
startCamera();
