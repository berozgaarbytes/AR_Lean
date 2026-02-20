const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const bootLayer = document.getElementById('boot_layer');
const consoleLog = document.getElementById('console');
const btnInit = document.getElementById('btn_init');

let session = null;
const MODEL_DIM = 416;
let vibroBuffer = new Array(60).fill(0);

const CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

btnInit.addEventListener('click', boot);
btnInit.addEventListener('touchstart', (e) => { e.preventDefault(); boot(); });

async function boot() {
    consoleLog.innerText = "STATUS: ACCESSING CAMERA...";
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        await video.play();

        consoleLog.innerText = "STATUS: LOADING OmniV CORE...";
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";
        session = await ort.InferenceSession.create('./yolox_nano.onnx', { executionProviders: ['wasm'] });

        bootLayer.style.display = 'none';
        renderLoop();
    } catch (e) { consoleLog.innerText = "BOOT_ERR: " + e.message; }
}

function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    runInference();
    requestAnimationFrame(renderLoop);
}

async function runInference() {
    if (!session) return;
    try {
        const tensor = await prepareInput(video);
        const result = await session.run({ images: tensor });
        const output = result[session.outputNames[0]].data;

        let detections = decode(output, canvas.width, canvas.height);
        detections = nms(detections, 0.4); // Stricter NMS to stop multiple boxes
        
        detections.forEach(d => {
            audit(d);
            draw(d);
        });
    } catch (e) {}
}

function decode(data, vW, vH) {
    const boxes = [];
    const threshold = 0.35;
    for (let i = 0; i < data.length; i += 85) {
        const score = data[i+4];
        if (score > threshold) {
            let max=0, id=0; for(let c=0; c<80; c++) if(data[i+5+c]>max){max=data[i+5+c]; id=c;}
            if (score * max > threshold) {
                let cx = data[i], cy = data[i+1], w = data[i+2], h = data[i+3];
                // AUTO-SCALE: If coords are normalized (0-1), scale them to MODEL_DIM
                if (w < 1.0) { cx *= MODEL_DIM; cy *= MODEL_DIM; w *= MODEL_DIM; h *= MODEL_DIM; }
                
                boxes.push({
                    x: (cx - w/2) * (vW / MODEL_DIM),
                    y: (cy - h/2) * (vH / MODEL_DIM),
                    w: w * (vW / MODEL_DIM),
                    h: h * (vH / MODEL_DIM),
                    score: score * max,
                    label: CLASSES[id]
                });
            }
        }
    }
    return boxes;
}

function audit(d) {
    // Vibrometry (PAMI 2016)
    const patch = ctx.getImageData(d.x + d.w/2, d.y + d.h/2, 2, 2).data;
    let b = (patch[0]+patch[1]+patch[2])/3;
    vibroDataPush(b);
    
    const variance = getVariance();
    d.mass = variance > 12 ? "LOW_DENSITY" : "HIGH_DENSITY";
    d.vol = Math.round((d.w * d.h * ((d.w+d.h)/2)) / 15000);
}

function vibroDataPush(v) { vibroBuffer.push(v); vibroBuffer.shift(); }
function getVariance() {
    const avg = vibroBuffer.reduce((a,b)=>a+b)/60;
    return vibroBuffer.reduce((a,b)=>a+Math.pow(b-avg,2),0);
}

function draw(d) {
    ctx.strokeStyle = d.mass.includes("HIGH") ? "#0f0" : "#f0f";
    ctx.lineWidth = 4;
    ctx.strokeRect(d.x, d.y, d.w, d.h);
    
    ctx.fillStyle = "rgba(0,0,0,0.8)";
    ctx.fillRect(d.x, d.y - 50, d.w > 150 ? d.w : 150, 50);
    ctx.fillStyle = "#0ff";
    ctx.font = "bold 12px monospace";
    ctx.fillText(`${d.label.toUpperCase()} | VOL: ${d.vol}`, d.x + 5, d.y - 35);
    ctx.fillStyle = d.mass.includes("HIGH") ? "#0f0" : "#f0f";
    ctx.fillText(`MASS: ${d.mass}`, d.x + 5, d.y - 15);
}

// Stable Prep & NMS
async function prepareInput(src) {
    const c = document.createElement('canvas'); c.width=416; c.height=416;
    c.getContext('2d').drawImage(src,0,0,416,416);
    const d = c.getContext('2d').getImageData(0,0,416,416).data;
    const f = new Float32Array(3 * 416**2);
    for(let i=0; i<d.length/4; i++) {
        f[i]=d[i*4]/255; f[i+416**2]=d[i*4+1]/255; f[i+2*416**2]=d[i*4+2]/255;
    }
    return new ort.Tensor('float32', f, [1, 3, 416, 416]);
}

function nms(b, l) {
    b.sort((a,b)=>b.score-a.score);
    const r = []; const s = new Array(b.length).fill(false);
    for(let i=0; i<b.length; i++) {
        if(s[i]) continue; r.push(b[i]);
        for(let j=i+1; j<b.length; j++) if(!s[j] && iou(b[i], b[j])>l) s[j]=true;
    }
    return r;
}

function iou(a,b) {
    const x1=Math.max(a.x,b.x), y1=Math.max(a.y,b.y), x2=Math.min(a.x+a.w,b.x+b.w), y2=Math.min(a.y+a.h,b.y+b.h);
    const i=Math.max(0,x2-x1)*Math.max(0,y2-y1);
    return i/(a.w*a.h+b.w*b.h-i);
}
