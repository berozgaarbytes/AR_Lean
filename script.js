const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const bootLayer = document.getElementById('boot_layer');
const statusConsole = document.getElementById('status_console');
const btnActivate = document.getElementById('btn_activate');

let session = null;
const MODEL_DIM = 416;

const prepCanvas = document.createElement('canvas');
prepCanvas.width = MODEL_DIM;
prepCanvas.height = MODEL_DIM;
const prepCtx = prepCanvas.getContext('2d', { willReadFrequently: true });

let motionBuffer = new Array(120).fill(0);
const COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

// 1. GLOBAL ERROR CATCHER (Prints Safari errors to your screen)
window.onerror = function(msg, source, lineno, colno, error) {
    statusConsole.innerHTML += `<br><span style="color:red">ERR: ${msg} (Line ${lineno})</span>`;
    return true; 
};

function updateStatus(msg) {
    statusConsole.innerHTML += `<br>> ${msg}`;
    console.log(msg);
}

// 2. BIND NATIVE IOS TOUCH EVENTS
btnActivate.addEventListener('click', bootSystem);
btnActivate.addEventListener('touchstart', function(e) {
    e.preventDefault(); // Stop double-firing on iOS
    bootSystem();
}, {passive: false});

async function bootSystem() {
    // Disable button to prevent spamming
    btnActivate.style.pointerEvents = "none";
    btnActivate.innerText = "STARTING...";

    // STEP 1: GET CAMERA INSTANTLY (Bypasses Safari Timeout)
    try {
        updateStatus("ACCESSING BACK LENS...");
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' },
            audio: false
        });
        video.srcObject = stream;
        
        // Wait for video to be ready before playing
        await new Promise(resolve => {
            video.onloadedmetadata = () => {
                video.play();
                resolve();
            };
        });
        updateStatus("CAMERA SECURED.");
    } catch (e) {
        updateStatus(`<span style="color:red">CAMERA BLOCKED: ${e.message}</span>`);
        return; // Stop here if camera fails
    }

    // STEP 2: LOAD THE HEAVY AI (Now safe because camera is running)
    try {
        updateStatus("DOWNLOADING AI BRAIN (WASM)...");
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";
        ort.env.wasm.numThreads = 1; 

        session = await ort.InferenceSession.create('./yolox_nano.onnx', { 
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        updateStatus("AI READY.");

        // STEP 3: START HUD
        bootLayer.style.display = 'none';
        renderLoop();
    } catch (e) {
        updateStatus(`<span style="color:red">AI LOAD FAILED: ${e.message}</span>`);
    }
}

async function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // VIBROMETRY
    extractMicroMotion();

    // AI
    await runAI();

    requestAnimationFrame(renderLoop);
}

function extractMicroMotion() {
    const midX = Math.floor(canvas.width / 2);
    const midY = Math.floor(canvas.height / 2);
    const patch = ctx.getImageData(midX - 5, midY - 5, 10, 10).data;
    
    let total = 0;
    for (let i = 0; i < patch.length; i += 4) total += (patch[i] + patch[i+1] + patch[i+2]) / 3;
    motionBuffer.push(total / (patch.length / 4));
    motionBuffer.shift();
    
    // Draw Vibrometry
    ctx.strokeStyle = "#f00"; ctx.lineWidth = 2; ctx.beginPath();
    for(let i = 0; i < motionBuffer.length; i++) {
        const x = canvas.width - 20 - (motionBuffer.length - i) * 2;
        const y = canvas.height - 50 - ((motionBuffer[i] - 128) * 2); 
        if(i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "#f00"; ctx.font="10px Arial"; ctx.fillText("VIBRATION", canvas.width-80, canvas.height-20);
}

async function runAI() {
    if (!session) return;
    try {
        const tensor = await prepareInput(video);
        const result = await session.run({ images: tensor });
        const data = result[session.outputNames[0]].data;
        let detections = decodeYOLOX(data, canvas.width, canvas.height);
        detections = applyNMS(detections, 0.45);
        drawDetections(detections);
    } catch(e) {}
}

async function prepareInput(source) {
    prepCtx.drawImage(source, 0, 0, MODEL_DIM, MODEL_DIM);
    const imgData = prepCtx.getImageData(0, 0, MODEL_DIM, MODEL_DIM).data;
    const floatData = new Float32Array(3 * MODEL_DIM * MODEL_DIM);
    for(let i = 0; i < imgData.length / 4; i++) {
        floatData[i] = imgData[i*4];
        floatData[i + MODEL_DIM**2] = imgData[i*4+1];
        floatData[i + 2 * MODEL_DIM**2] = imgData[i*4+2];
    }
    return new ort.Tensor('float32', floatData, [1, 3, MODEL_DIM, MODEL_DIM]);
}

function decodeYOLOX(data, viewW, viewH) {
    const boxes = [];
    for (let i = 0; i < data.length; i += 85) {
        if (data[i + 4] > 0.35) {
            let maxCls = 0, clsId = 0;
            for(let c = 0; c < 80; c++) if(data[i + 5 + c] > maxCls) { maxCls = data[i + 5 + c]; clsId = c; }
            const conf = data[i + 4] * maxCls;
            if (conf > 0.35) {
                let cx = data[i], cy = data[i+1], w = data[i+2], h = data[i+3];
                if (w < 2 && h < 2) { cx *= MODEL_DIM; cy *= MODEL_DIM; w *= MODEL_DIM; h *= MODEL_DIM; }
                boxes.push({ x: (cx - w/2) * (viewW / MODEL_DIM), y: (cy - h/2) * (viewH / MODEL_DIM), w: w * (viewW / MODEL_DIM), h: h * (viewH / MODEL_DIM), score: conf, label: COCO_CLASSES[clsId] });
            }
        }
    }
    return boxes;
}

function applyNMS(boxes, limit) {
    boxes.sort((a,b) => b.score - a.score);
    const res = []; const skip = new Array(boxes.length).fill(false);
    for(let i=0; i<boxes.length; i++) {
        if(skip[i]) continue; res.push(boxes[i]);
        for(let j=i+1; j<boxes.length; j++) if(!skip[j] && iou(boxes[i], boxes[j]) > limit) skip[j] = true;
    }
    return res;
}

function iou(a, b) {
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y), x2 = Math.min(a.x+a.w, b.x+b.w), y2 = Math.min(a.y+a.h, b.y+b.h);
    const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return i / (a.w*a.h + b.w*b.h - i);
}

function drawDetections(dets) {
    ctx.strokeStyle = "rgba(0, 255, 255, 0.5)"; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(canvas.width/2 - 20, canvas.height/2); ctx.lineTo(canvas.width/2 + 20, canvas.height/2);
    ctx.moveTo(canvas.width/2, canvas.height/2 - 20); ctx.lineTo(canvas.width/2, canvas.height/2 + 20); ctx.stroke();
    
    dets.forEach(d => {
        ctx.strokeStyle = "#0ff"; ctx.lineWidth = 3; ctx.strokeRect(d.x, d.y, d.w, d.h);
        ctx.fillStyle = "rgba(0, 255, 255, 0.8)"; ctx.fillRect(d.x, d.y - 25, d.w, 25);
        ctx.fillStyle = "#000"; ctx.font = "bold 16px Arial"; ctx.fillText(`${d.label.toUpperCase()} ${Math.round(d.score*100)}%`, d.x + 5, d.y - 7);
    });
}
