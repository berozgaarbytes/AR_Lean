const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');
const bootLayer = document.getElementById('boot_layer');
const status = document.getElementById('status');

let session = null;
const MODEL_DIM = 416;
let motionBuffer = new Array(100).fill(0);
let outputShape = null;

const COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

async function bootSystem() {
    status.innerText = "PROBING ONNX TENSORS...";
    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";
        session = await ort.InferenceSession.create('./yolox_nano.onnx', { executionProviders: ['wasm'] });
        
        // --- SELF-HEALING: Detect Output Shape ---
        outputShape = session.outputTypes[session.outputNames[0]].tensorType.shape;
        console.log("Model Output Detected:", outputShape);

        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        video.onloadedmetadata = () => { video.play(); bootLayer.style.display = 'none'; renderLoop(); };
    } catch (e) { status.innerText = "CRITICAL BOOT ERROR: " + e.message; }
}

async function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 1. VISUAL VIBROMETRY
    const p = ctx.getImageData(canvas.width/2, canvas.height/2, 1, 1).data;
    motionBuffer.push((p[0]+p[1]+p[2])/3); motionBuffer.shift();
    drawVibrometryHUD();

    // 2. SELF-HEALING AI INFERENCE
    try {
        const tensor = await prepareInput(video);
        const result = await session.run({ images: tensor });
        const data = result[session.outputNames[0]].data;
        
        // Decide decoding based on shape [1, 3549, 85] vs [1, 85, 3549]
        let detections = (outputShape[1] === 85) ? decodeTransposed(data) : decodeStandard(data);
        
        detections = applyNMS(detections, 0.45);
        drawDetections(detections);
    } catch (e) { /* Skipping Frame */ }

    requestAnimationFrame(renderLoop);
}

function decodeStandard(data) {
    const res = [];
    for (let i = 0; i < data.length; i += 85) {
        if (data[i+4] * data[i+5] > 0.4) {
            res.push({ x: (data[i]-data[i+2]/2)*(canvas.width/MODEL_DIM), y: (data[i+1]-data[i+3]/2)*(canvas.height/MODEL_DIM), w: data[i+2]*(canvas.width/MODEL_DIM), h: data[i+3]*(canvas.height/MODEL_DIM), score: data[i+4]*data[i+5], label: COCO_CLASSES[0] });
        }
    }
    return res;
}

function decodeTransposed(data) {
    const res = [];
    const numBoxes = 3549;
    for (let i = 0; i < numBoxes; i++) {
        const objScore = data[i + 4 * numBoxes];
        const clsScore = data[i + 5 * numBoxes];
        if (objScore * clsScore > 0.4) {
            let w = data[i + 2 * numBoxes] * (canvas.width/MODEL_DIM);
            let h = data[i + 3 * numBoxes] * (canvas.height/MODEL_DIM);
            res.push({ x: (data[i + 0 * numBoxes] * (canvas.width/MODEL_DIM)) - w/2, y: (data[i + 1 * numBoxes] * (canvas.height/MODEL_DIM)) - h/2, w: w, h: h, score: objScore * clsScore, label: COCO_CLASSES[0] });
        }
    }
    return res;
}

function applyNMS(boxes, limit) {
    boxes.sort((a,b) => b.score - a.score);
    const res = []; const skip = new Array(boxes.length).fill(false);
    for(let i=0; i<boxes.length; i++) {
        if(skip[i]) continue; res.push(boxes[i]);
        for(let j=i+1; j<boxes.length; j++) {
            if(!skip[j] && iou(boxes[i], boxes[j]) > limit) skip[j] = true;
        }
    }
    return res;
}

function iou(a,b) {
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y), x2 = Math.min(a.x+a.w, b.x+b.w), y2 = Math.min(a.y+a.h, b.y+b.h);
    const i = Math.max(0, x2-x1) * Math.max(0, y2-y1);
    return i / (a.w*a.h + b.w*b.h - i);
}

function drawDetections(dets) {
    dets.forEach(d => {
        ctx.strokeStyle = "#ff00ff"; ctx.lineWidth = 3;
        ctx.strokeRect(d.x, d.y, d.w, d.h);
        ctx.fillStyle = "#ff00ff"; ctx.fillText(`${d.label.toUpperCase()} ${Math.round(d.score*100)}%`, d.x, d.y-10);
    });
}

function drawVibrometryHUD() {
    ctx.strokeStyle = "#f00"; ctx.beginPath();
    for(let i=0; i<motionBuffer.length; i++) {
        let x = canvas.width - (motionBuffer.length-i)*3, y = canvas.height - 50 - (motionBuffer[i]/255)*100;
        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.fillStyle = "#f00"; ctx.fillText("VIBROMETRY_POWER_SPECTRUM", canvas.width - 200, canvas.height - 160);
}

async function prepareInput(source) {
    const t = document.createElement('canvas'); t.width = MODEL_DIM; t.height = MODEL_DIM;
    t.getContext('2d').drawImage(source, 0, 0, MODEL_DIM, MODEL_DIM);
    const d = t.getContext('2d').getImageData(0,0,MODEL_DIM,MODEL_DIM).data;
    const f = new Float32Array(3 * MODEL_DIM**2);
    for(let i=0; i<d.length/4; i++) {
        f[i] = d[i*4]/255; f[i+MODEL_DIM**2] = d[i*4+1]/255; f[i+2*MODEL_DIM**2] = d[i*4+2]/255;
    }
    return new ort.Tensor('float32', f, [1, 3, MODEL_DIM, MODEL_DIM]);
}
