const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');
const bootLayer = document.getElementById('boot_layer');
const status = document.getElementById('status');

let session = null;
const MODEL_DIM = 416; // Standard YOLOX-Nano dimension

async function bootSystem() {
    status.innerText = "LOADING WASM CORE...";
    try {
        // 1. Force CDN path for engine files
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

        // 2. Create Inference Session
        session = await ort.InferenceSession.create('./yolox_nano.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        status.innerText = "STARTING OPTICAL LENS...";
        
        // 3. Request Camera
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: 1280, height: 720 }
        });

        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            bootLayer.style.display = 'none';
            renderLoop();
        };
    } catch (e) {
        status.innerText = "BOOT ERROR: " + e.message;
        console.error(e);
    }
}

async function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // AI INFERENCE STEP
    const tensor = await prepareInput(video);
    try {
        const result = await session.run({ images: tensor });
        const output = result[Object.keys(result)[0]].data;
        
        // DECODE YOLOX OUTPUT
        const detections = decodeYOLOX(output, canvas.width, canvas.height);
        drawDetections(detections);
    } catch (e) {
        console.warn("AI skipped frame");
    }

    requestAnimationFrame(renderLoop);
}

async function prepareInput(source) {
    const off = new OffscreenCanvas(MODEL_DIM, MODEL_DIM);
    const oCtx = off.getContext('2d');
    oCtx.drawImage(source, 0, 0, MODEL_DIM, MODEL_DIM);
    const imgData = oCtx.getImageData(0, 0, MODEL_DIM, MODEL_DIM).data;

    const floatData = new Float32Array(3 * MODEL_DIM * MODEL_DIM);
    for (let i = 0; i < imgData.length / 4; i++) {
        floatData[i] = imgData[i * 4] / 255.0; // R
        floatData[i + MODEL_DIM * MODEL_DIM] = imgData[i * 4 + 1] / 255.0; // G
        floatData[i + 2 * MODEL_DIM * MODEL_DIM] = imgData[i * 4 + 2] / 255.0; // B
    }
    return new ort.Tensor('float32', floatData, [1, 3, MODEL_DIM, MODEL_DIM]);
}

function decodeYOLOX(data, viewW, viewH) {
    const detections = [];
    const threshold = 0.4;
    // YOLOX outputs [1, 3549, 85] for 416 resolution
    // 3549 is the number of anchor-free prediction points
    for (let i = 0; i < data.length; i += 85) {
        const objScore = data[i + 4];
        const clsScore = data[i + 5]; // Class 0 = Person
        const confidence = objScore * clsScore;

        if (confidence > threshold) {
            // YOLOX coordinates are relative to the 416x416 input
            let cx = data[i] * (viewW / MODEL_DIM);
            let cy = data[i + 1] * (viewH / MODEL_DIM);
            let w = data[i + 2] * (viewW / MODEL_DIM);
            let h = data[i + 3] * (viewH / MODEL_DIM);

            detections.push({
                x: cx - w/2,
                y: cy - h/2,
                w: w,
                h: h,
                score: confidence
            });
        }
    }
    return detections;
}

function drawDetections(detections) {
    detections.forEach(d => {
        ctx.strokeStyle = "#ff00ff";
        ctx.lineWidth = 3;
        ctx.strokeRect(d.x, d.y, d.w, d.h);
        
        ctx.fillStyle = "#ff00ff";
        ctx.font = "bold 14px monospace";
        ctx.fillText(`HUMAN_${Math.round(d.score*100)}%`, d.x, d.y - 10);
    });
}
