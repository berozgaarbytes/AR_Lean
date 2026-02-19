const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');

let session;

async function initÆ() {
    try {
        // 1. Load the ONNX Model from your repo
        // This expects yolox_nano.onnx to be in the same folder
        session = await ort.InferenceSession.create('./yolox_nano.onnx', {
            executionProviders: ['wasm'], // Optimized for CPU
            graphOptimizationLevel: 'all'
        });
        
        status.innerText = "Æ-SENTRY: MODEL LOADED. STARTING LENS...";
        startCamera();
    } catch (e) {
        status.innerText = "ERROR: " + e.message;
    }
}

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        .then(stream => {
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                renderLoop();
            };
        });
}

async function renderLoop() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw camera frame
    ctx.drawImage(video, 0, 0);

    // 2. Prepare Input for YOLOX
    // YOLOX-Nano usually expects 416x416 or 640x640
    const inputData = prepareInput(video);
    const tensor = new ort.Tensor('float32', inputData, [1, 3, 416, 416]);

    // 3. Run Inference
    const output = await session.run({ images: tensor });
    
    // 4. Draw Results (Boxes & Labels)
    processOutput(output);

    requestAnimationFrame(renderLoop);
}

// Logic to resize and normalize video frames for the AI
function prepareInput(v) {
    // Simplified: In a real demo, use an offscreen canvas to resize to 416x416
    // and return a Float32Array of RGB values normalized (0-1)
    const data = new Float32Array(1 * 3 * 416 * 416); 
    // ... normalization logic goes here ...
    return data;
}

initÆ();
