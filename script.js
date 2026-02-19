const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');
const bootLayer = document.getElementById('boot_layer');
const status = document.getElementById('status');

let session = null;
const MODEL_DIM = 416;

async function bootSystem() {
    status.innerText = "TUNING AI PRECISION...";
    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";
        session = await ort.InferenceSession.create('./yolox_nano.onnx', {
            executionProviders: ['wasm']
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
        status.innerText = "INIT ERROR: " + e.message;
    }
}

async function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const tensor = await prepareInput(video);
    try {
        const result = await session.run({ images: tensor });
        const outputName = session.outputNames[0];
        const outputData = result[outputName].data;
        
        let detections = decodeYOLOX(outputData, canvas.width, canvas.height);
        
        // --- THE FIX: APPLY NON-MAXIMUM SUPPRESSION ---
        detections = applyNMS(detections, 0.45); // 0.45 IOU threshold
        
        drawDetections(detections);
    } catch (e) { console.warn("Processing..."); }

    requestAnimationFrame(renderLoop);
}

async function prepareInput(source) {
    const off = new OffscreenCanvas(MODEL_DIM, MODEL_DIM);
    const oCtx = off.getContext('2d');
    oCtx.drawImage(source, 0, 0, MODEL_DIM, MODEL_DIM);
    const imgData = oCtx.getImageData(0, 0, MODEL_DIM, MODEL_DIM).data;

    const floatData = new Float32Array(3 * MODEL_DIM * MODEL_DIM);
    for (let i = 0; i < imgData.length / 4; i++) {
        floatData[i] = imgData[i * 4] / 255.0; 
        floatData[i + MODEL_DIM * MODEL_DIM] = imgData[i * 4 + 1] / 255.0;
        floatData[i + 2 * MODEL_DIM * MODEL_DIM] = imgData[i * 4 + 2] / 255.0;
    }
    return new ort.Tensor('float32', floatData, [1, 3, MODEL_DIM, MODEL_DIM]);
}

function decodeYOLOX(data, viewW, viewH) {
    const candidates = [];
    const CONF_THRESHOLD = 0.5; // Increased to kill false positives

    for (let i = 0; i < data.length; i += 85) {
        const objScore = data[i + 4];
        const clsScore = data[i + 5]; // 0 is Person
        const finalScore = objScore * clsScore;

        if (finalScore > CONF_THRESHOLD) {
            let cx = data[i] * (viewW / MODEL_DIM);
            let cy = data[i + 1] * (viewH / MODEL_DIM);
            let w = data[i + 2] * (viewW / MODEL_DIM);
            let h = data[i + 3] * (viewH / MODEL_DIM);

            candidates.push({
                x: cx - w / 2,
                y: cy - h / 2,
                w: w,
                h: h,
                score: finalScore
            });
        }
    }
    return candidates;
}

// --- NMS ALGORITHM: REMOVES DUPLICATE BOXES ---
function applyNMS(boxes, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    const selected = new Array(boxes.length).fill(true);

    for (let i = 0; i < boxes.length; i++) {
        if (selected[i]) {
            result.push(boxes[i]);
            for (let j = i + 1; j < boxes.length; j++) {
                if (selected[j] && calculateIOU(boxes[i], boxes[j]) > iouThreshold) {
                    selected[j] = false;
                }
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

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const areaA = boxA.w * boxA.h;
    const areaB = boxB.w * boxB.h;
    return interArea / (areaA + areaB - interArea);
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
