const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');
const uiLayer = document.getElementById('ui_layer');
const statusText = document.getElementById('status_text');

let session = null;

async function bootSystem() {
    statusText.innerText = "LOADING AI ENGINE...";
    
    try {
        // 1. FORCE THE PATH (This fixes the "Initializing" hang)
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.1/dist/";

        // 2. LOAD MODEL (Make sure the name is exactly yolox_nano.onnx)
        session = await ort.InferenceSession.create('./yolox_nano.onnx', {
            executionProviders: ['wasm']
        });

        statusText.innerText = "ACQUIRING BACK CAMERA...";
        
        // 3. START CAMERA
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });

        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            uiLayer.style.display = 'none'; // Hide the black screen
            renderLoop();
        };

    } catch (err) {
        statusText.innerHTML = `<b style="color:red">ERROR:</b> ${err.message}<br><small>Ensure you are using HTTPS</small>`;
        console.error("ORT Error:", err);
    }
}

function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Draw Camera feed to screen
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Simple HUD overlay to prove it's working
    drawHUD();

    requestAnimationFrame(renderLoop);
}

function drawHUD() {
    ctx.strokeStyle = "#0ff";
    ctx.lineWidth = 2;
    ctx.strokeRect(20, 20, canvas.width - 40, canvas.height - 40);
    ctx.fillStyle = "#0ff";
    ctx.fillText("Æ-SENTRY LIVE // AI ACTIVE", 30, 40);
}
