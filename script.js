const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');

let session;
const MODEL_SIZE = 416; // Change to 640 if your model is 640

async function initÆ() {
    status.innerText = "Æ-SENTRY: LOADING WASM CORE...";
    try {
        // Force the WASM path to the CDN to avoid local file errors
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        
        // Load the model from your GitHub repo folder
        session = await ort.InferenceSession.create('./yolox_nano.onnx', {
            executionProviders: ['wasm']
        });
        
        status.innerText = "Æ-SENTRY: ONLINE. ACCESSING LENS...";
        startCamera();
    } catch (e) {
        status.innerText = "INITIALIZATION ERROR: " + e.message;
        console.error(e);
    }
}

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        .then(stream => {
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                video.play();
                renderLoop();
            };
        });
}

async function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Draw reality
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Prepare YOLOX Input
    const tensor = await prepareInput(video);

    try {
        const output = await session.run({ images: tensor });
        // The 'output' contains raw numbers. For a demo, let's pulse the HUD 
        // to show the AI is "thinking" in real-time.
        drawIndustrialOverlay();
    } catch (e) {
        console.warn("Inference skipped: ", e.message);
    }

    requestAnimationFrame(renderLoop);
}

async function prepareInput(source) {
    // Create an offscreen canvas to resize the image to 416x416
    const offscreen = new OffscreenCanvas(MODEL_SIZE, MODEL_SIZE);
    const oCtx = offscreen.getContext('2d');
    oCtx.drawImage(source, 0, 0, MODEL_SIZE, MODEL_SIZE);
    
    const imgData = oCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const { data } = imgData;
    
    // Float32Array for [1, 3, 416, 416] (N, C, H, W)
    const floatData = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
    
    // Normalize and Transpose (RGBRGB -> RRR...GGG...BBB...)
    for (let i = 0; i < data.length / 4; i++) {
        floatData[i] = data[i * 4] / 255.0; // R
        floatData[i + MODEL_SIZE * MODEL_SIZE] = data[i * 4 + 1] / 255.0; // G
        floatData[i + 2 * MODEL_SIZE * MODEL_SIZE] = data[i * 4 + 2] / 255.0; // B
    }
    
    return new ort.Tensor('float32', floatData, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

function drawIndustrialOverlay() {
    ctx.strokeStyle = "#0ff";
    ctx.lineWidth = 2;
    ctx.strokeRect(50, 50, canvas.width - 100, canvas.height - 100);
    
    // Scanline Effect
    const scanlineY = (Date.now() / 10) % canvas.height;
    ctx.fillStyle = "rgba(0, 255, 255, 0.1)";
    ctx.fillRect(0, scanlineY, canvas.width, 2);
}
