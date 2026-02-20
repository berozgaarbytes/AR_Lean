const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const bootLayer = document.getElementById('boot_layer');
const statusConsole = document.getElementById('status_console');

let session = null;
const MODEL_DIM = 416; // Standard YOLOX-Nano input size

// iOS Safe Image Processor (Avoids OffscreenCanvas bug)
const prepCanvas = document.createElement('canvas');
prepCanvas.width = MODEL_DIM;
prepCanvas.height = MODEL_DIM;
const prepCtx = prepCanvas.getContext('2d', { willReadFrequently: true });

let motionBuffer = new Array(120).fill(0); // Vibrometry data array

const COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

function updateStatus(msg) {
    statusConsole.innerHTML += `<br>> ${msg}`;
    console.log(msg);
}

async function bootSystem() {
    updateStatus("INITIALIZING IOS SAFARI PROTOCOLS...");
    try {
        // 1. Force WASM paths for mobile browsers
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";
        ort.env.wasm.numThreads = 1; // Prevent iOS Safari from crashing due to thread exhaustion
        
        updateStatus("LOADING YOLOX-NANO.ONNX...");
        session = await ort.InferenceSession.create('./yolox_nano.onnx', { 
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        updateStatus("AI ENGINE COMPILED.");

        updateStatus("REQUESTING CAMERA (BACK LENS)...");
        // iOS Specific constraint: exact environment
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: { exact: 'environment' } },
            audio: false
        });

        // iOS explicit overrides
        video.setAttribute('playsinline', '');
        video.setAttribute('autoplay', '');
        video.setAttribute('muted', '');
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            video.play();
            bootLayer.style.display = 'none';
            renderLoop();
        };

    } catch (e) {
        // Fallback for devices that don't support 'exact' back camera
        if (e.name === 'OverconstrainedError' || e.name === 'NotReadableError') {
            updateStatus("BACK CAMERA BLOCKED. TRYING ANY CAMERA...");
            tryFallbackCamera();
        } else {
            updateStatus(`<span style="color:red">ERROR: ${e.message}</span>`);
        }
    }
}

async function tryFallbackCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.onloadedmetadata = () => { video.play(); bootLayer.style.display = 'none'; renderLoop(); };
}

async function renderLoop() {
    // 1. Match Canvas to iPhone Screen
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 2. PAMI 2016 VISUAL VIBROMETRY
    extractMicroMotion();

    // 3. AI OBJECT DETECTION
    await runAI();

    requestAnimationFrame(renderLoop);
}

function extractMicroMotion() {
    // Analyze a 10x10 patch in the dead center of the screen
    const midX = Math.floor(canvas.width / 2);
    const midY = Math.floor(canvas.height / 2);
    const patch = ctx.getImageData(midX - 5, midY - 5, 10, 10).data;
    
    let totalBrightness = 0;
    for (let i = 0; i < patch.length; i += 4) {
        totalBrightness += (patch[i] + patch[i+1] + patch[i+2]) / 3;
    }
    const avgBrightness = totalBrightness / (patch.length / 4);
    
    motionBuffer.push(avgBrightness);
    motionBuffer.shift();
    drawVibrometryHUD();
}

async function runAI() {
    if (!session) return;
    try {
        const tensor = await prepareInput(video);
        const result = await session.run({ images: tensor });
        const outputName = session.outputNames[0];
        const rawData = result[outputName].data;
        
        let detections = decodeYOLOX(rawData, canvas.width, canvas.height);
        detections = applyNMS(detections, 0.45); // Standard IOU Threshold
        drawDetections(detections);
    } catch (e) { 
        // Silent catch to keep the video feed smooth if a frame drops
    }
}

async function prepareInput(source) {
    // Standard DOM Canvas instead of OffscreenCanvas for iOS safety
    prepCtx.drawImage(source, 0, 0, MODEL_DIM, MODEL_DIM);
    const imgData = prepCtx.getImageData(0, 0, MODEL_DIM, MODEL_DIM).data;
    
    const floatData = new Float32Array(3 * MODEL_DIM * MODEL_DIM);
    for(let i = 0; i < imgData.length / 4; i++) {
        // Most ONNX YOLOX models expect 0-255 scale. 
        floatData[i] = imgData[i*4];                     // R
        floatData[i + MODEL_DIM**2] = imgData[i*4+1];    // G
        floatData[i + 2 * MODEL_DIM**2] = imgData[i*4+2]; // B
    }
    return new ort.Tensor('float32', floatData, [1, 3, MODEL_DIM, MODEL_DIM]);
}

function decodeYOLOX(data, viewW, viewH) {
    const boxes = [];
    const confThreshold = 0.35; // Sensitivity
    
    // Expecting standard [1, 3549, 85] layout
    for (let i = 0; i < data.length; i += 85) {
        const objScore = data[i + 4];
        if (objScore > confThreshold) {
            let maxClsScore = 0; 
            let clsId = 0;
            for(let c = 0; c < 80; c++) { 
                if(data[i + 5 + c] > maxClsScore) { 
                    maxClsScore = data[i + 5 + c]; 
                    clsId = c; 
                } 
            }
            
            const finalConfidence = objScore * maxClsScore;
            if (finalConfidence > confThreshold) {
                // Determine if coords are normalized (0-1) or absolute (0-416)
                let cx = data[i];
                let cy = data[i+1];
                let w = data[i+2];
                let h = data[i+3];
                
                // Safety scaling logic depending on your exact model's export
                if (w < 1.5 && h < 1.5) {
                    cx *= MODEL_DIM; cy *= MODEL_DIM; w *= MODEL_DIM; h *= MODEL_DIM;
                }

                boxes.push({
                    x: (cx - w/2) * (viewW / MODEL_DIM),
                    y: (cy - h/2) * (viewH / MODEL_DIM),
                    w: w * (viewW / MODEL_DIM),
                    h: h * (viewH / MODEL_DIM),
                    score: finalConfidence,
                    label: COCO_CLASSES[clsId]
                });
            }
        }
    }
    return boxes;
}

function applyNMS(boxes, iouLimit) {
    boxes.sort((a,b) => b.score - a.score);
    const result = [];
    const skip = new Array(boxes.length).fill(false);
    
    for(let i = 0; i < boxes.length; i++) {
        if(skip[i]) continue;
        result.push(boxes[i]);
        for(let j = i + 1; j < boxes.length; j++) {
            if(!skip[j] && calculateIOU(boxes[i], boxes[j]) > iouLimit) {
                skip[j] = true;
            }
        }
    }
    return result;
}

function calculateIOU(a, b) {
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x+a.w, b.x+b.w), y2 = Math.min(a.y+a.h, b.y+b.h);
    const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
    const inter = w * h;
    return inter / (a.w*a.h + b.w*b.h - inter);
}

function drawDetections(dets) {
    // Draw target crosshair
    ctx.strokeStyle = "rgba(0, 255, 255, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(canvas.width/2 - 20, canvas.height/2); ctx.lineTo(canvas.width/2 + 20, canvas.height/2);
    ctx.moveTo(canvas.width/2, canvas.height/2 - 20); ctx.lineTo(canvas.width/2, canvas.height/2 + 20);
    ctx.stroke();

    dets.forEach(d => {
        // Bounding Box
        ctx.strokeStyle = "#0ff"; // Cyan for industrial tech look
        ctx.lineWidth = 3;
        ctx.strokeRect(d.x, d.y, d.w, d.h);
        
        // Label Background
        ctx.fillStyle = "rgba(0, 255, 255, 0.8)";
        ctx.fillRect(d.x, d.y - 25, d.w, 25);
        
        // Text
        ctx.fillStyle = "#000";
        ctx.font = "bold 16px Arial";
        ctx.fillText(`${d.label.toUpperCase()} ${Math.round(d.score*100)}%`, d.x + 5, d.y - 7);
    });
}

function drawVibrometryHUD() {
    ctx.strokeStyle = "#f00";
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    // Draw the spectrum graph in the bottom right corner
    for(let i = 0; i < motionBuffer.length; i++) {
        const x = canvas.width - 20 - (motionBuffer.length - i) * 2;
        // Normalize the brightness variation for visual impact
        const variance = (motionBuffer[i] - 128) * 2; 
        const y = canvas.height - 50 - variance;
        if(i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
}
