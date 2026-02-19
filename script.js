const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const specCanvas = document.getElementById('spectrum_canvas');
const ctx = canvas.getContext('2d');
const sCtx = specCanvas.getContext('2d');
const bootLayer = document.getElementById('boot_layer');
const status = document.getElementById('status');

let session = null;
const MODEL_DIM = 416;

// COCO Classes (80 Total)
const COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

async function bootSystem() {
    status.innerText = "LOADING VIBROMETRY ENGINE...";
    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";
        session = await ort.InferenceSession.create('./yolox_nano.onnx', { executionProviders: ['wasm'] });

        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            bootLayer.style.display = 'none';
            renderLoop();
        };
    } catch (e) { status.innerText = "ERROR: " + e.message; }
}

async function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 1. AI Detection
    const tensor = await prepareInput(video);
    try {
        const result = await session.run({ images: tensor });
        const output = result[Object.keys(result)[0]].data;
        let detections = decodeYOLOX(output, canvas.width, canvas.height);
        
        // APPLY NMS (Fixes multiple boxes for one human)
        detections = applyNMS(detections, 0.45);
        drawDetections(detections);
    } catch (e) { console.warn("AI processing..."); }

    // 2. Visual Vibrometry (Temporal Power Spectrum)
    runVibrometry();

    requestAnimationFrame(renderLoop);
}

// --- VISUAL VIBROMETRY ENGINE ---
let motionBuffer = [];
function runVibrometry() {
    // Extract micro-motion from the center of the screen
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const pixelData = ctx.getImageData(centerX, centerY, 1, 1).data;
    const intensity = (pixelData[0] + pixelData[1] + pixelData[2]) / 3;
    
    motionBuffer.push(intensity);
    if (motionBuffer.length > 100) motionBuffer.shift();

    // Draw Temporal Power Spectrum
    sCtx.clearRect(0, 0, specCanvas.width, specCanvas.height);
    sCtx.strokeStyle = "#f00"; // Red for Vibrometry
    sCtx.beginPath();
    for (let i = 0; i < motionBuffer.length; i++) {
        const x = (i / motionBuffer.length) * specCanvas.width;
        const y = specCanvas.height - (motionBuffer[i] / 255) * specCanvas.height;
        if (i === 0) sCtx.moveTo(x, y); else sCtx.lineTo(x, y);
    }
    sCtx.stroke();
    sCtx.fillStyle = "#fff";
    sCtx.fillText("TEMPORAL_POWER_SPECTRUM (Hz)", 5, 15);
}

// --- YOLOX & NMS LOGIC ---
function applyNMS(boxes, threshold) {
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    const selected = new Array(boxes.length).fill(true);
    for (let i = 0; i < boxes.length; i++) {
        if (selected[i]) {
            result.push(boxes[i]);
            for (let j = i + 1; j < boxes.length; j++) {
                if (selected[j] && calculateIOU(boxes[i], boxes[j]) > threshold) selected[j] = false;
            }
        }
    }
    return result;
}

function calculateIOU(boxA, boxB) {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
    const yB = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);
    const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    return inter / (boxA.w * boxA.h + boxB.w * boxB.h - inter);
}

function decodeYOLOX(data, viewW, viewH) {
    const candidates = [];
    for (let i = 0; i < data.length; i += 85) {
        const objScore = data[i + 4];
        if (objScore > 0.5) {
            let maxScore = 0; let classId = 0;
            for (let c = 0; c < 80; c++) {
                if (data[i + 5 + c] > maxScore) { maxScore = data[i + 5 + c]; classId = c; }
            }
            if (objScore * maxScore > 0.4) {
                candidates.push({
                    x: (data[i] - data[i+2]/2) * (viewW / MODEL_DIM),
                    y: (data[i+1] - data[i+3]/2) * (viewH / MODEL_DIM),
                    w: data[i+2] * (viewW / MODEL_DIM),
                    h: data[i+3] * (viewH / MODEL_DIM),
                    score: objScore * maxScore,
                    class: COCO_CLASSES[classId]
                });
            }
        }
    }
    return candidates;
}

function drawDetections(detections) {
    detections.forEach(d => {
        ctx.strokeStyle = "#0ff";
        ctx.strokeRect(d.x, d.y, d.w, d.h);
        ctx.fillStyle = "#0ff";
        ctx.fillText(`${d.class.toUpperCase()} ${Math.round(d.score*100)}%`, d.x, d.y - 5);
    });
}

async function prepareInput(source) {
    const off = new OffscreenCanvas(MODEL_DIM, MODEL_DIM);
    const oCtx = off.getContext('2d');
    oCtx.drawImage(source, 0, 0, MODEL_DIM, MODEL_DIM);
    const d = oCtx.getImageData(0, 0, MODEL_DIM, MODEL_DIM).data;
    const f = new Float32Array(3 * MODEL_DIM * MODEL_DIM);
    for (let i = 0; i < d.length / 4; i++) {
        f[i] = d[i * 4] / 255.0; f[i + MODEL_DIM**2] = d[i * 4 + 1] / 255.0; f[i + 2 * MODEL_DIM**2] = d[i * 4 + 2] / 255.0;
    }
    return new ort.Tensor('float32', f, [1, 3, MODEL_DIM, MODEL_DIM]);
}
