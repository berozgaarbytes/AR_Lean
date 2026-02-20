const log = (m) => document.getElementById('console').innerText = `>> ${m}`;
const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

let session = null;
const MODEL_DIM = 416;
let vibroCache = new Array(60).fill(0);

const CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

async function bootSequence() {
    log("WASM_INIT...");
    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";
        
        log("FETCHING_MODEL...");
        // This relative path works on GitHub Pages if the .onnx is in the same folder
        session = await ort.InferenceSession.create('./yolox_nano.onnx', { executionProviders: ['wasm'] });
        
        log("CAMERA_INIT...");
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        await video.play();

        document.getElementById('boot_layer').style.display = 'none';
        log("PIPELINE_ACTIVE");
        render();
    } catch (e) { log(`CRITICAL_ERR: ${e.message}`); }
}

async function render() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
        const tensor = await prepareInput(video);
        const result = await session.run({ images: tensor });
        const output = result[session.outputNames[0]].data;

        // Unified 80-Class Decoder with Auto-Scaling
        let dets = decode(output, canvas.width, canvas.height);
        dets = nms(dets, 0.45);
        
        dets.forEach(d => {
            // PAMI 2016 Visual Vibrometry logic
            audit(d);
            drawHUD(d);
        });
    } catch (e) { }
    requestAnimationFrame(render);
}

function decode(data, vW, vH) {
    const boxes = [];
    // YOLOX standard output size is 3549 boxes (416 resolution)
    for (let i = 0; i < 3549; i++) {
        const idx = i * 85;
        const objScore = data[idx + 4];
        if (objScore > 0.35) {
            let maxCls = 0, clsId = 0;
            for(let c=0; c<80; c++) if(data[idx+5+c] > maxCls){ maxCls=data[idx+5+c]; clsId=c; }
            
            const conf = objScore * maxCls;
            if (conf > 0.4) {
                let cx = data[idx], cy = data[idx+1], w = data[idx+2], h = data[idx+3];
                // Auto-scale check (Normalised vs Raw)
                if (w < 1.0) { cx*=416; cy*=416; w*=416; h*=416; }

                boxes.push({
                    x: (cx - w/2) * (vW / 416),
                    y: (cy - h/2) * (vH / 416),
                    w: w * (vW / 416),
                    h: h * (vH / 416),
                    score: conf,
                    label: CLASSES[clsId]
                });
            }
        }
    }
    return boxes;
}

function audit(d) {
    // Extract center intensity for PAMI vibrometry
    const p = ctx.getImageData(d.x + d.w/2, d.y + d.h/2, 1, 1).data;
    const brightness = (p[0]+p[1]+p[2])/3;
    vibroCache.push(brightness); vibroCache.shift();

    const mean = vibroCache.reduce((a,b)=>a+b)/60;
    const variance = vibroCache.reduce((a,b)=>a + Math.pow(b-mean,2), 0);
    
    // Mass Inference: Low variance = Heavy/Solid, High variance = Light/Hollow
    d.massStatus = variance > 15 ? "LIGHT/HOLLOW" : "HEAVY/LOADED";
    d.vol = Math.round((d.w * d.h * ((d.w+d.h)/2)) / 15000);
}

function drawHUD(d) {
    ctx.strokeStyle = d.massStatus.includes("HEAVY") ? "#0f0" : "#f0f";
    ctx.lineWidth = 3;
    ctx.strokeRect(d.x, d.y, d.w, d.h);
    
    ctx.fillStyle = "rgba(0,0,0,0.8)";
    ctx.fillRect(d.x, d.y - 45, 180, 45);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 12px monospace";
    ctx.fillText(`${d.label.toUpperCase()} | VOL:${d.vol}`, d.x+5, d.y-30);
    ctx.fillStyle = d.massStatus.includes("HEAVY") ? "#0f0" : "#f0f";
    ctx.fillText(`MASS: ${d.massStatus}`, d.x+5, d.y-10);
}

// Utils
async function prepareInput(src) {
    const c = document.createElement('canvas'); c.width=416; c.height=416;
    c.getContext('2d').drawImage(src, 0, 0, 416, 416);
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
        for(let j=i+1; j<b.length; j++) {
            const x1=Math.max(b[i].x,b[j].x), y1=Math.max(b[i].y,b[j].y), x2=Math.min(b[i].x+b[i].w,b[j].x+b[j].w), y2=Math.min(b[i].y+b[i].h,b[j].y+b[j].h);
            const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);
            if(inter/(b[i].w*b[i].h+b[j].w*b[j].h-inter) > l) s[j]=true;
        }
    }
    return r;
}
