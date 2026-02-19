const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');
const uiLayer = document.getElementById('ui_layer');
const statusText = document.getElementById('status_text');
const hud = document.getElementById('hud');

let session = null;

async function bootSystem() {
    statusText.innerText = "SYSTEM: ALLOCATING WASM RESOURCES...";
    
    try {
        // 1. Point to the WASM worker files on the CDN
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        // 2. Load Model (Make sure yolox_nano.onnx is in your GitHub folder!)
        session = await ort.InferenceSession.create('./yolox_nano.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        statusText.innerText = "SYSTEM: ACQUIRING OPTICAL FEED...";
        
        // 3. Trigger Camera (Must be done inside a user-click event!)
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                facingMode: 'environment', // BACK CAMERA
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            // Hide the boot screen and show the HUD
            uiLayer.style.display = 'none';
            hud.style.display = 'block';
            renderLoop();
        };

    } catch (err) {
        statusText.innerHTML = `<span style="color:red">BOOT_FAILURE: ${err.message}</span><br>Check if HTTPS is enabled.`;
        console.error(err);
    }
}

function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Draw Camera Feed
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // --- AR OVERLAY ---
    drawSentryHUD();

    requestAnimationFrame(renderLoop);
}

function drawSentryHUD() {
    const w = canvas.width;
    const h = canvas.height;

    // Scanline effect
    const scanY = (Date.now() / 15) % h;
    ctx.strokeStyle = "rgba(0, 255, 255, 0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, scanY);
    ctx.lineTo(w, scanY);
    ctx.stroke();

    // Corner brackets
    ctx.strokeStyle = "#0ff";
    ctx.lineWidth = 3;
    const bSize = 40;
    // Top Left
    ctx.beginPath(); ctx.moveTo(50, 50+bSize); ctx.lineTo(50, 50); ctx.lineTo(50+bSize, 50); ctx.stroke();
    // Top Right
    ctx.beginPath(); ctx.moveTo(w-50-bSize, 50); ctx.lineTo(w-50, 50); ctx.lineTo(w-50, 50+bSize); ctx.stroke();
}
