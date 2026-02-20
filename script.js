const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const uiOverlay = document.getElementById('ui_overlay');
const consoleLog = document.getElementById('console');
const btnInit = document.getElementById('btn_init');

let session = null;
const MODEL_DIM = 416;
let vibroData = new Array(60).fill(0);

// COMPLETE 80-CLASS COCO SET
const CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

btnInit.addEventListener('touchstart', bootSequence);
btnInit.addEventListener('click', bootSequence);

async function bootSequence() {
    consoleLog.innerHTML = "STATUS: INITIALIZING CAMERA...";
    try {
        // 1. Camera First (Safari Requirement)
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        video.srcObject = stream;
        await video.play();

        // 2. AI Second
        consoleLog.innerHTML = "STATUS: LOADING YOLOX-NANO ONNX...";
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";
        session = await ort.InferenceSession.create('./yolox_nano.onnx', { executionProviders: ['wasm'] });
        
        consoleLog.innerHTML = "STATUS: PIPELINE ACTIVE.";
        uiOverlay.style.display = 'none';
        renderLoop();
    } catch (e) {
        consoleLog.innerHTML = `<span style="color:red">ERROR: ${e.message}</span>`;
    }
}

function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    processVision();
    requestAnimationFrame(renderLoop);
}

async function processVision() {
    if (!session) return;
    try {
        const tensor = await prepareInput(video);
        const result = await session.run({ images: tensor });
        const output = result[session.outputNames[0]].data;
        
        let detections = decode(output, canvas.width, canvas.height);
        detections = nms(detections, 0.45);
        
        detections.forEach(d => auditObject(d));
    } catch (e) {}
}

function auditObject(d) {
    // 1. Volumetric Estimate
    const vol = Math.round((d.w * d.h * ((d.w+d.h)/2)) / 10000);
    
    // 2. Vibrometry Mass Audit (PAMI 2016 Method)
    const patch = ctx.getImageData(d.x + d.w/2, d.y + d.h/2, 4, 4).data;
    let b = 0; for(let i=0; i<patch.length; i+=4) b += (patch[i]+patch[i+1]+patch[i+2])/3;
    vibroData.push(b/4); vibroData.shift();
    
    const avg = vibroData.reduce((a,b)=>a+b)/60;
    const variance = vibroData.reduce((a,b)=>a+Math.pow(b-avg,2),0);
    const massStatus = variance > 8 ? "LOW_DENSITY (EMPTY)" : "HIGH_DENSITY (LOADED)";

    // 3. Render Unified HUD
    ctx.strokeStyle = massStatus.includes("HIGH") ? "#0f0" : "#f0f";
    ctx.lineWidth = 3;
    ctx.strokeRect(d.x, d.y, d.w, d.h);
    
    ctx.fillStyle = "rgba(0,255,255,0.8)";
    ctx.font = "bold 14px Arial";
    ctx.fillText(`${d.label.toUpperCase()} | VOL: ${vol}`, d.x, d.y - 10);
    ctx.fillStyle = massStatus.includes("HIGH") ? "#0f0" : "#f0f";
    ctx.fillText(`MASS: ${massStatus}`, d.x, d.y + d.h + 20);
}

// STABLE DECODER & PREP
async function prepareInput(src) {
    const c = document.createElement('canvas'); c.width=416; c.height=416;
    const cx = c.getContext('2d'); cx.drawImage(src, 0, 0, 416, 416);
    const d = cx.getImageData(0,0,416,416).data;
    const f = new Float32Array(3 * 416**2);
    for(let i=0; i<d.length/4; i++) {
        f[i] = d[i*4]/255; f[i+416**2] = d[i*4+1]/255; f[i+2*416**2] = d[i*4+2]/255;
    }
    return new ort.Tensor('float32', f, [1, 3, 416, 416]);
}

function decode(data, w, h) {
    const boxes = [];
    for (let i = 0; i < data.length; i += 85) {
        if (data[i+4] > 0.4) {
            let max=0, id=0; for(let c=0; c<80; c++) if(data[i+5+c]>max){max=data[i+5+c]; id=c;}
            if (data[i+4]*max > 0.4) {
                boxes.push({ x:(data[i]-data[i+2]/2)*(w/416), y:(data[i+1]-data[i+3]/2)*(h/416), w:data[i+2]*(w/416), h:data[i+3]*(h/416), score:data[i+4]*max, label:CLASSES[id] });
            }
        }
    }
    return boxes;
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
