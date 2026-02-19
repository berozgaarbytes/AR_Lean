const videoElement = document.getElementById('video_input');
const canvasElement = document.getElementById('canvas_output');
const canvasCtx = canvasElement.getContext('2d');

// --- Commercial Configuration ---
let lastPos = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
const SMOOTHING = 0.15; // The "Golden Ratio" for smooth AR (0.1 - 0.2)

function onResults(results) {
    // Match Canvas to Viewport
    canvasElement.width = window.innerWidth;
    canvasElement.height = window.innerHeight;
    const w = canvasElement.width;
    const h = canvasElement.height;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, w, h);

    // 1. Render Reality (Back Camera)
    canvasCtx.drawImage(results.image, 0, 0, w, h);
    canvasCtx.restore();

    // 2. The Logic: Smoothing and Anchoring
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        // We track Landmark 9 (The Middle Finger MCP - the "Anchor" of the hand)
        const target = results.multiHandLandmarks[0][9];
        const tx = target.x * w;
        const ty = target.y * h;

        // Apply LERP (Linear Interpolation) to remove jitter
        lastPos.x += (tx - lastPos.x) * SMOOTHING;
        lastPos.y += (ty - lastPos.y) * SMOOTHING;

        drawSpatialHUD(lastPos.x, lastPos.y);
    }
}

function drawSpatialHUD(x, y) {
    // 1. Draw the "Power Core" (The visual anchor point)
    const glow = canvasCtx.createRadialGradient(x, y, 0, x, y, 50);
    glow.addColorStop(0, 'rgba(0, 255, 255, 0.8)');
    glow.addColorStop(1, 'rgba(0, 255, 255, 0)');
    
    canvasCtx.fillStyle = glow;
    canvasCtx.beginPath();
    canvasCtx.arc(x, y, 50, 0, Math.PI * 2);
    canvasCtx.fill();

    // 2. Draw the Data Bracket
    canvasCtx.strokeStyle = "#0ff";
    canvasCtx.lineWidth = 2;
    canvasCtx.beginPath();
    canvasCtx.moveTo(x + 40, y - 40);
    canvasCtx.lineTo(x + 60, y - 60);
    canvasCtx.lineTo(x + 200, y - 60);
    canvasCtx.stroke();

    // 3. Render Industrial Data
    canvasCtx.fillStyle = "#fff";
    canvasCtx.font = "bold 16px 'Courier New'";
    canvasCtx.fillText("ASSET_ID: PUMP_V4", x + 65, y - 75);
    
    canvasCtx.fillStyle = "#0ff";
    canvasCtx.font = "12px 'Courier New'";
    canvasCtx.fillText("RPM: 1,450 [STABLE]", x + 65, y - 45);
    canvasCtx.fillText("TEMP: 42°C [OPTIMAL]", x + 65, y - 30);

    // 4. Subtle Pulse Effect
    const pulse = Math.sin(Date.now() / 300) * 5;
    canvasCtx.strokeStyle = "rgba(0, 255, 255, 0.5)";
    canvasCtx.beginPath();
    canvasCtx.arc(x, y, 30 + pulse, 0, Math.PI * 2);
    canvasCtx.stroke();
}

// --- Initialize MediaPipe (Safe for Commercial Use) ---
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1, // Higher complexity = rock solid tracking
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    },
    facingMode: "environment", // BACK camera for shop floor
    width: 1280,
    height: 720
});

camera.start();
