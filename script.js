const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');
const bootLayer = document.getElementById('boot_layer');
const status = document.getElementById('status');

let session = null;
const MODEL_DIM = 416;

async function bootSystem() {
    status.innerText = "LOADING ENGINE...";
    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";
        session = await ort.InferenceSession.create('./yolox_nano.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

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
        status.innerText = "ERROR: " + e.message;
    }
}

async function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const tensor = await prepareInput(video);
    try {
        const result = await session.run({ images: tensor });
        const output = result[Object.keys(result)[0]].data;
        
        // This is the specific YOLOX decoding logic
        const detections = decodeYOLOX(output, canvas.width, canvas.height);
        drawDetections(detections);
    } catch (e) { console.warn("Inference skipped"); }

    requestAnimationFrame(renderLoop);
}

async function prepareInput(source) {
    const off = new OffscreenCanvas(MODEL_DIM, MODEL_DIM);
    const oCtx = off.getContext('2d');
    oCtx.drawImage(source, 0, 0, MODEL_DIM, MODEL_DIM);
    const imgData = oCtx.getImageData(0, 0, MODEL_DIM, MODEL_DIM).data;

    const floatData = new Float32Array(3 * MODEL_DIM * MODEL_DIM);
    for (let i = 0; i < imgData.length / 4; i++) {
        floatData[i] = imgData[i * 4]; // R
        floatData[i + MODEL_DIM * MODEL_DIM] = imgData[i * 4 + 1]; // G
        floatData[i + 2 * MODEL_DIM * MODEL_DIM] = imgData[i * 4 + 2]; // B
    }
    return new ort.Tensor('float32', floatData, [1, 3, MODEL_DIM, MODEL_DIM]);
}

function decodeYOLOX(data, viewW, viewH) {
    const detections = [];
    const threshold = 0.3; // Lowered for the demo
    const strides = [8, 16, 32];
    let offset = 0;

    strides.forEach(stride => {
        const gridH = MODEL_DIM / stride;
        const gridW = MODEL_DIM / stride;

        for (let gY = 0; gY < gridH; gY++) {
            for (let gX = 0; gX < gridW; gX++) {
                const idx = (offset + gY * gridW + gX) * 85;
                const objScore = data[idx + 4];
                const clsScore = data[idx + 5]; // Class 0: Person
                const confidence = objScore * clsScore;

                if (confidence > threshold) {
                    // YOLOX Grid Decoding
                    let cx = (data[idx] + gX) * stride;
                    let cy = (data[idx + 1] + gY) * stride;
                    let w = Math.exp(data[idx + 2]) * stride;
                    let h = Math.exp(data[idx + 3]) * stride;

                    detections.push({
                        x: (cx - w / 2) * (viewW / MODEL_DIM),
                        y: (cy - h / 2) * (viewH / MODEL_DIM),
                        w: w * (viewW / MODEL_DIM),
                        h: h * (viewH / MODEL_DIM),
                        score: confidence
                    });
                }
            }
        }
        offset += gridH * gridW;
    });
    return detections;
}

function drawDetections(detections) {
    detections.forEach(d => {
        ctx.strokeStyle = "#ff00ff";
        ctx.lineWidth = 4;
        ctx.strokeRect(d.x, d.y, d.w, d.h);
        ctx.fillStyle = "#ff00ff";
        ctx.fillText(`HUMAN ${Math.round(d.score * 100)}%`, d.x, d.y - 10);
    });
}
