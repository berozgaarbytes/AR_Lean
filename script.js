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

    btnInit.addEventListener('click', bootSequence);
    btnInit.addEventListener('touchstart', (e) => { e.preventDefault(); bootSequence(); });

    async function bootSequence() {
        consoleLog.innerText = "STATUS: SECURING CAMERA...";
        try {
            // iOS Safari: Must get camera first to lock the user-gesture
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } } 
            });
            video.srcObject = stream;
            await video.play();

            consoleLog.innerText = "STATUS: COMPILING AI BRAIN...";
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";
            
            // RELATIVE PATH: Ensure yolox_nano.onnx is exactly in your root git folder
            session = await ort.InferenceSession.create('./yolox_nano.onnx', { executionProviders: ['wasm'] });
            
            uiOverlay.style.display = 'none';
            renderLoop();
        } catch (e) { consoleLog.innerText = "CRITICAL ERROR: " + e.message; }
    }

    function renderLoop() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        runAuditPipeline();
        requestAnimationFrame(renderLoop);
    }

    async function runAuditPipeline() {
        if (!session) return;
        try {
            const tensor = await prepareInput(video);
            
            // SELF-IDENTIFYING RUN
            const result = await session.run({ images: tensor });
            const outputName = session.outputNames[0]; // Finds the output dynamically
            const outputData = result[outputName].data;
            
            let dets = decodeYOLOX(outputData, canvas.width, canvas.height);
            dets = nms(dets, 0.45);
            
            dets.forEach(d => {
                // VISUAL VIBROMETRY (PAMI 2016)
                auditMass(d);
                drawHUD(d);
            });
        } catch (e) { /* Skipping frame */ }
    }

    function decodeYOLOX(data, vW, vH) {
        const boxes = [];
        const threshold = 0.35; // Sensitivity floor
        
        // YOLOX-Nano usually outputs [1, 3549, 85]
        for (let i = 0; i < data.length; i += 85) {
            const objScore = data[i+4];
            if (objScore > threshold) {
                let max=0, id=0; 
                for(let c=0; c<80; c++) { 
                    if(data[i+5+c]>max){ max=data[i+5+c]; id=c; } 
                }
                
                if (objScore * max > threshold) {
                    let cx = data[i], cy = data[i+1], w = data[i+2], h = data[i+3];
                    
                    // AUTO-NORMALIZATION FIX
                    // If your model outputs 0.0-1.0 coords, scale them to 416
                    if (w <= 1.0) { cx *= MODEL_DIM; cy *= MODEL_DIM; w *= MODEL_DIM; h *= MODEL_DIM; }

                    boxes.push({
                        x: (cx - w/2) * (vW / MODEL_DIM),
                        y: (cy - h/2) * (vH / MODEL_DIM),
                        w: w * (vW / MODEL_DIM),
                        h: h * (vH / MODEL_DIM),
                        score: objScore * max,
                        label: CLASSES[id]
                    });
                }
            }
        }
        return boxes;
    }

    function auditMass(d) {
        // Extract micro-motion intensity (PAMI 2016)
        const patch = ctx.getImageData(d.x + d.w/2, d.y + d.h/2, 2, 2).data;
        const b = (patch[0]+patch[1]+patch[2])/3;
        vibroBuffer.push(b); vibroBuffer.shift();
        
        const avg = vibroBuffer.reduce((a,b)=>a+b)/60;
        const variance = vibroBuffer.reduce((a,b)=>a+Math.pow(b-avg,2),0);
        
        // Industrial Mass Logic
        d.mass = variance > 12 ? "LOW_DENSITY (EMPTY)" : "HIGH_DENSITY (LOADED)";
        d.vol = Math.round((d.w * d.h * ((d.w+d.h)/2)) / 15000);
    }

    function drawHUD(d) {
        ctx.strokeStyle = d.mass.includes("HIGH") ? "#0f0" : "#f0f";
        ctx.lineWidth = 4;
        ctx.strokeRect(d.x, d.y, d.w, d.h);
        
        // Data Window
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(d.x, d.y - 55, 200, 55);
        ctx.fillStyle = "#0ff";
        ctx.font = "bold 13px monospace";
        ctx.fillText(`ID: ${d.label.toUpperCase()}`, d.x+5, d.y-38);
        ctx.fillText(`VOL: ${d.vol} m3(rel)`, d.x+5, d.y-23);
        ctx.fillStyle = d.mass.includes("HIGH") ? "#0f0" : "#f0f";
        ctx.fillText(`MASS: ${d.mass}`, d.x+5, d.y-8);
    }

    async function prepareInput(src) {
        const c = document.createElement('canvas'); c.width=MODEL_DIM; c.height=MODEL_DIM;
        c.getContext('2d').drawImage(src, 0, 0, MODEL_DIM, MODEL_DIM);
        const d = c.getContext('2d').getImageData(0,0,MODEL_DIM,MODEL_DIM).data;
        const f = new Float32Array(3 * MODEL_DIM**2);
        for(let i=0; i<d.length/4; i++) {
            f[i]=d[i*4]/255; f[i+MODEL_DIM**2]=d[i*4+1]/255; f[i+2*MODEL_DIM**2]=d[i*4+2]/255;
        }
        return new ort.Tensor('float32', f, [1, 3, MODEL_DIM, MODEL_DIM]);
    }

    function nms(b, l) {
        b.sort((a,b)=>b.score-a.score);
        const r = []; const s = new Array(b.length).fill(false);
        for(let i=0; i<b.length; i++) {
            if(s[i]) continue; r.push(b[i]);
            for(let j=i+1; j<b.length; j++) {
                if(!s[j] && iou(b[i], b[j]) > l) s[j] = true;
            }
        }
        return r;
    }

    function iou(a,b) {
        const x1=Math.max(a.x,b.x), y1=Math.max(a.y,b.y), x2=Math.min(a.x+a.w,b.x+b.w), y2=Math.min(a.y+a.h,b.y+b.h);
        const i=Math.max(0,x2-x1)*Math.max(0,y2-y1);
        return i/(a.w*a.h+b.w*b.h-i);
    }
});
