const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');
const bootLayer = document.getElementById('boot_layer');
const status = document.getElementById('status');

let session = null;
const MODEL_DIM = 416;

async function bootSystem() {
    status.innerText = "INITIALIZING AI CORE...";
    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";
        session = await ort.InferenceSession.create('./yolox_nano.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: 1280, height: 720 }
        });

        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            bootLayer.style.display = 'none';
            renderLoop();
        };
    } catch (e) {
        status.innerText = "CRITICAL ERROR: " + e.message;
    }
}

async function renderLoop() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const tensor = await prepareInput(video);
    try {
        const result = await session.run({ images: tensor });
        // Some models name the output 'output', others 'output0'. This finds it automatically.
        const outputName = session.outputNames[0];
        const outputData = result[outputName].data;
        
        const detections = decodeRawYOLOX(outputData, canvas.width, canvas.height);
        drawDetections(detections);
    } catch (e) { console.warn("Frame Syncing..."); }

    requestAnimationFrame(renderLoop);
}

async function prepareInput(source) {
    const off = new OffscreenCanvas(MODEL_DIM, MODEL_DIM);
    const oCtx = off.getContext('2d');
    oCtx.drawImage(source, 0, 0, MODEL_DIM, MODEL_DIM);
    const imgData = oCtx.getImageData(0, 0, MODEL_DIM, MODEL_DIM).data;

    const floatData = new Float32Array(3 * MODEL_DIM * MODEL_DIM);
    // Standard Normalization (Dividing by 255)
    for (let i = 0; i < imgData.length / 4; i++) {
        floatData[i] = imgData[i * 4] / 255.0; // R
        floatData[i + MODEL_DIM * MODEL_DIM] = imgData[i * 4 + 1] / 255.0; // G
        floatData[i + 2 * MODEL_DIM * MODEL_DIM] = imgData[i * 4 + 2] / 255.0; // B
    }
    return new ort.Tensor('float32', floatData, [1, 3, MODEL_DIM, MODEL_DIM]);
}

function decodeRawYOLOX(data, viewW, viewH) {
    const detections = [];
    const threshold = 0.35; 
    
    // YOLOX-Nano output is usually 3549 rows of 85 columns
    // Col 0-3: Box, Col 4: Confidence, Col 5: Person
    for (let i = 0; i < data.length; i += 85) {
        const objScore = data[i + 4];
        if (objScore > threshold) {
            const clsScore = data[i + 5]; // 0 is Person
            const finalScore = objScore * clsScore;

            if (finalScore > threshold) {
                // Mapping relative coordinates back to screen
                let cx = data[i] * (viewW / MODEL_DIM);
                let cy = data[i + 1] * (viewH / MODEL_DIM);
                let w = data[i + 2] * (viewW / MODEL_DIM);
                let h = data[i + 3] * (viewH / MODEL_DIM);

                detections.push({
                    x: cx - w/2,
                    y: cy - h/2,
                    w: w,
                    h: h,
                    score: finalScore
                });
            }
        }
    }
    // Simple NMS: Remove overlapping boxes
    return detections.sort((a,b) => b.score - a.score).slice(0, 5);
}

function drawDetections(detections) {
    detections.forEach(d => {
        // High-Vis Pink for Industrial Safety
        ctx.strokeStyle = "#ff00ff";
        ctx.lineWidth = 4;
        ctx.strokeRect(d.x, d.y, d.w, d.h);
        
        ctx.fillStyle = "#ff00ff";
        ctx.font = "bold 16px monospace";
        ctx.fillText(`HUMAN ${Math.round(d.score * 100)}%`, d.x + 5, d.y - 10);
        
        // Add a small "Target" circle in center
        ctx.beginPath();
        ctx.arc(d.x + d.w/2, d.y + d.h/2, 5, 0, Math.PI*2);
        ctx.fill();
    });
}
