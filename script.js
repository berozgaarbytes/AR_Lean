const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');
const bootLayer = document.getElementById('boot_layer');
const status = document.getElementById('status');

let session = null;
const MODEL_DIM = 416;
let motionBuffer = new Array(100).fill(0);

const COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

async function bootSystem() {
    status.innerText = "LOADING ENGINE...";
    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";
        session = await ort.InferenceSession.create('./yolox_nano.onnx', { executionProviders: ['wasm'] });

        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        video.play();
        bootLayer.style.display = 'none';
        renderLoop();
    } catch (e) {
        status.innerText = "BOOT ERROR: " + e.message;
    }
}

function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 1. RUN VIBROMETRY (Center Point Analysis)
    const midX = canvas.width / 2;
    const midY = canvas.height / 2;
    const p = ctx.getImageData(midX, midY, 1, 1).data;
    motionBuffer.push((p[0] + p[1] + p[2]) / 3);
    motionBuffer.shift();
    drawVibrometryHUD();

    // 2. RUN AI DETECTION (Attempt every frame)
    runAI();

    requestAnimationFrame(renderLoop);
}

async function runAI() {
    if (!session) return;
    
    try {
        const tensor = await prepareInput(video);
        const result = await session.run({ images: tensor });
        const output = result[Object.keys(result)[0]].data;
        
        let detections = decodeYOLOX(output, canvas.width, canvas.height);
        detections = applyNMS(detections, 0.45);
        drawDetections(detections);
    } catch (e) { 
        // Silently skip if AI frame drops
    }
}

function decodeYOLOX(data, viewW, viewH) {
    const candidates = [];
    const threshold = 0.4;
    for (let i = 0; i < data.length; i += 85) {
        const objScore = data[i + 4];
        if (objScore > threshold) {
            let maxCls = 0; let clsId = 0;
            for(let c=0; c<80; c++) { if(data[i+5+c] > maxCls) { maxCls = data[i+5+c]; clsId = c; } }
            
            if (objScore * maxCls > threshold) {
                candidates.push({
                    x: (data[i] - data[i+2]/2) * (viewW / MODEL_DIM),
                    y: (data[i+1] - data[i+3]/2) * (viewH / MODEL_DIM),
                    w: data[i+2] * (viewW / MODEL_DIM),
                    h: data[i+3] * (viewH / MODEL_DIM),
                    score: objScore * maxCls,
                    label: COCO_CLASSES[clsId]
                });
            }
        }
    }
    return candidates;
}

function applyNMS(boxes, limit) {
    boxes.sort((a,b) => b.score - a.score);
    const result = [];
    const skip = new Array(boxes.length).fill(false);
    for(let i=0; i<boxes.length; i++) {
        if(skip[i]) continue;
        result.push(boxes[i]);
        for(let j=i+1; j<boxes.length; j++) {
            if(!skip[j] && iou(boxes[i], boxes[j]) > limit) skip[j] = true;
        }
    }
    return result;
}

function iou(a, b) {
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x+a.w, b.x+b.w), y2 = Math.min(a.y+a.h, b.y+b.h);
    const width = Math.max(0, x2 - x1), height = Math.max(0, y2 - y1);
    const inter = width * height;
    return inter / (a.w*a.h + b.w*b.h - inter);
}

function drawDetections(dets) {
    dets.forEach(d => {
        ctx.strokeStyle = "#0ff";
        ctx.lineWidth = 3;
        ctx.strokeRect(d.x, d.y, d.w, d.h);
        ctx.fillStyle = "#0ff";
        ctx.fillText(`${d.label.toUpperCase()} ${Math.round(d.score*100)}%`, d.x, d.y - 10);
    });
}

function drawVibrometryHUD() {
    ctx.strokeStyle = "#f00";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for(let i=0; i<motionBuffer.length; i++) {
        const x = canvas.width - (motionBuffer.length - i) * 3;
        const y = canvas.height - 50 - (motionBuffer[i]/255)*100;
        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.fillStyle = "#f00";
    ctx.fillText("VIBROMETRY_POWER_SPECTRUM", canvas.width - 200, canvas.height - 160);
}

async function prepareInput(source) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = MODEL_DIM; tempCanvas.height = MODEL_DIM;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(source, 0, 0, MODEL_DIM, MODEL_DIM);
    const d = tCtx.getImageData(0,0,MODEL_DIM,MODEL_DIM).data;
    const f = new Float32Array(3 * MODEL_DIM * MODEL_DIM);
    for(let i=0; i<d.length/4; i++) {
        f[i] = d[i*4]/255; f[i+MODEL_DIM**2] = d[i*4+1]/255; f[i+2*MODEL_DIM**2] = d[i*4+2]/255;
    }
    return new ort.Tensor('float32', f, [1, 3, MODEL_DIM, MODEL_DIM]);
}
