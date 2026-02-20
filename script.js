document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video_input');
    const canvas = document.getElementById('canvas_output');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const uiOverlay = document.getElementById('ui_overlay');
    const consoleLog = document.getElementById('console');
    const btnInit = document.getElementById('btn_init');

    let session = null;
    const MODEL_DIM = 416;
    let vibroBuffer = new Array(60).fill(0);

    const CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

    btnInit.addEventListener('click', boot);
    btnInit.addEventListener('touchstart', (e) => { e.preventDefault(); boot(); });

    async function boot() {
        consoleLog.innerText = "STATUS: INITIALIZING CAMERA...";
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            video.srcObject = stream;
            await video.play();

            consoleLog.innerText = "STATUS: LOADING OmniV CORE...";
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";
            session = await ort.InferenceSession.create('./yolox_nano.onnx', { executionProviders: ['wasm'] });
            
            uiOverlay.style.display = 'none';
            renderLoop();
        } catch (e) { consoleLog.innerText = "ERROR: " + e.message; }
    }

    function renderLoop() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        runAudit();
        requestAnimationFrame(renderLoop);
    }

    async function runAudit() {
        if (!session) return;
        try {
            const tensor = await prepareInput(video);
            const result = await session.run({ images: tensor });
            const outputName = session.outputNames[0];
            const output = result[outputName].data;
            
            // DETECT OUTPUT FORMAT AUTOMATICALLY
            let dets = [];
            if (output.length === 3549 * 85) {
                dets = decodeStandard(output);
            } else {
                dets = decodeTransposed(output);
            }
            
            dets = nms(dets, 0.45);
            
            dets.forEach(d => {
                const patch = ctx.getImageData(d.x + d.w/2, d.y + d.h/2, 2, 2).data;
                const brightness = (patch[0]+patch[1]+patch[2])/3;
                vibroBuffer.push(brightness); vibroBuffer.shift();
                
                const avg = vibroBuffer.reduce((a,b)=>a+b)/60;
                const variance = vibroBuffer.reduce((a,b)=>a+Math.pow(b-avg,2),0);
                d.mass = variance > 12 ? "LOW_DENSITY" : "HIGH_DENSITY";
                d.vol = Math.round((d.w * d.h * ((d.w+d.h)/2)) / 15000);
                
                drawHUD(d);
            });
        } catch (e) { }
    }

    // Standard Decoder: [3549, 85]
    function decodeStandard(data) {
        const boxes = [];
        for (let i = 0; i < 3549; i++) {
            const row = i * 85;
            const score = data[row + 4];
            if (score > 0.3) {
                let max = 0, id = 0;
                for (let c = 0; c < 80; c++) if (data[row+5+c] > max) { max = data[row+5+c]; id = c; }
                if (score * max > 0.35) {
                    processBox(data[row], data[row+1], data[row+2], data[row+3], score * max, id, boxes);
                }
            }
        }
        return boxes;
    }

    // Transposed Decoder: [85, 3549]
    function decodeTransposed(data) {
        const boxes = [];
        for (let i = 0; i < 3549; i++) {
            const score = data[4 * 3549 + i] * data[5 * 3549 + i];
            if (score > 0.35) {
                processBox(data[0*3549+i], data[1*3549+i], data[2*3549+i], data[3*3549+i], score, 0, boxes);
            }
        }
        return boxes;
    }

    function processBox(cx, cy, w, h, score, id, boxes) {
        if (w <= 1.0) { cx *= 416; cy *= 416; w *= 416; h *= 416; }
        boxes.push({
            x: (cx - w/2) * (canvas.width / 416),
            y: (cy - h/2) * (canvas.height / 416),
            w: w * (canvas.width / 416),
            h: h * (canvas.height / 416),
            score: score,
            label: CLASSES[id]
        });
    }

    function drawHUD(d) {
        ctx.strokeStyle = d.mass.includes("HIGH") ? "#0f0" : "#f0f";
        ctx.lineWidth = 4;
        ctx.strokeRect(d.x, d.y, d.w, d.h);
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(d.x, d.y - 50, 180, 50);
        ctx.fillStyle = "#fff";
        ctx.font = "bold 12px monospace";
        ctx.fillText(`${d.label.toUpperCase()} | VOL: ${d.vol}`, d.x+5, d.y-30);
        ctx.fillStyle = d.mass.includes("HIGH") ? "#0f0" : "#f0f";
        ctx.fillText(`MASS: ${d.mass}`, d.x+5, d.y-10);
    }

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
});
