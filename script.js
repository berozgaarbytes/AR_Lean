const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const bootLayer = document.getElementById('boot_layer');
const status = document.getElementById('status');

let session = null;
const MODEL_DIM = 416;
let vibrationBuffer = new Array(60).fill(0); // 1-second rolling window at 60fps

const COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

async function bootSystem() {
    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";
        session = await ort.InferenceSession.create('./yolox_nano.onnx', { executionProviders: ['wasm'] });
        
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        video.onloadedmetadata = () => { video.play(); bootLayer.style.display = 'none'; renderLoop(); };
    } catch (e) { status.innerText = "ERROR: " + e.message; }
}

function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    runWIPAudit();
    requestAnimationFrame(renderLoop);
}

async function runWIPAudit() {
    if (!session) return;
    try {
        const tensor = await prepareInput(video);
        const result = await session.run({ images: tensor });
        const output = result[session.outputNames[0]].data;
        
        let detections = decodeYOLOX(output, canvas.width, canvas.height);
        detections = applyNMS(detections, 0.45);
        
        detections.forEach(d => {
            // 1. VOLUMETRIC ANALYSIS
            const volume = calculateVolume(d.w, d.h);
            
            // 2. VIBRATION ANALYSIS (PAMI 2016)
            const massRating = analyzeMassThroughVibration(d);
            
            drawAuditHUD(d, volume, massRating);
        });
    } catch (e) {}
}

function calculateVolume(w, h) {
    // Assumption: Industrial pallets/boxes are usually square-based
    const depth = (w + h) / 2;
    return Math.round((w * h * depth) / 10000); // Relative units
}

function analyzeMassThroughVibration(det) {
    // Get pixel intensity from the center of the detected object
    const patch = ctx.getImageData(det.x + det.w/2, det.y + det.h/2, 5, 5).data;
    let brightness = 0;
    for(let i=0; i<patch.length; i+=4) brightness += (patch[i]+patch[i+1]+patch[i+2])/3;
    
    vibrationBuffer.push(brightness / (patch.length/4));
    vibrationBuffer.shift();

    // Calculate Variance (Micro-Motion)
    const avg = vibrationBuffer.reduce((a,b) => a+b) / vibrationBuffer.length;
    const variance = vibrationBuffer.reduce((a,b) => a + Math.pow(b-avg, 2), 0);
    
    // HEAVY MASS = Damped vibration (Low Variance)
    // EMPTY/LIGHT = Rattling/High frequency (High Variance)
    return variance > 10 ? "EMPTY/LIGHT" : "LOADED/HEAVY";
}

function drawAuditHUD(d, vol, mass) {
    ctx.strokeStyle = mass === "LOADED/HEAVY" ? "#0f0" : "#f0f";
    ctx.lineWidth = 4;
    ctx.strokeRect(d.x, d.y, d.w, d.h);
    
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(d.x, d.y - 60, 200, 60);
    
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px monospace";
    ctx.fillText(`ID: ${d.label.toUpperCase()}`, d.x + 5, d.y - 45);
    ctx.fillText(`EST_VOL: ${vol} units`, d.x + 5, d.y - 30);
    ctx.fillStyle = mass === "LOADED/HEAVY" ? "#0f0" : "#f0f";
    ctx.fillText(`MASS_INT: ${mass}`, d.x + 5, d.y - 15);
}

// ... (Use the same prepareInput, decodeYOLOX, applyNMS, and iou functions from our stable version)
