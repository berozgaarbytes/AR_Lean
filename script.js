const video = document.getElementById('video_input');
const canvas = document.getElementById('canvas_output');
const ctx = canvas.getContext('2d');

let model;
let lastFrame;

// Initialize Detection
async function init() {
    // Loading COCO-SSD (Apache 2.0) which uses a MobileNet/YOLO-style backbone
    model = await cocoSsd.load();
    setupCamera();
}

function setupCamera() {
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        .then(stream => {
            video.srcObject = stream;
            video.onloadedmetadata = () => { predict(); };
        });
}

async function predict() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // 1. MOTION DETECTION (The "Cheap" Way)
    ctx.drawImage(video, 0, 0);
    const currentFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    if (lastFrame) {
        detectMotion(lastFrame, currentFrame);
    }
    lastFrame = currentFrame;

    // 2. HUMAN/OBJECT DETECTION (YOLOX-style Logic)
    // We run this every 5th frame to save CPU
    const predictions = await model.detect(video);
    
    predictions.forEach(p => {
        // Draw the "Smart" HUD
        ctx.strokeStyle = p.class === 'person' ? '#ff00ff' : '#00ffff';
        ctx.lineWidth = 4;
        ctx.strokeRect(...p.bbox);
        
        ctx.fillStyle = p.class === 'person' ? '#ff00ff' : '#00ffff';
        ctx.font = "bold 18px monospace";
        ctx.fillText(`${p.class.toUpperCase()} [${Math.round(p.score*100)}%]`, p.bbox[0], p.bbox[1] - 10);
    });

    requestAnimationFrame(predict);
}

function detectMotion(oldImg, newImg) {
    // Simple pixel delta to find movement
    for (let i = 0; i < oldImg.data.length; i += 400) { // Sample pixels for speed
        const diff = Math.abs(oldImg.data[i] - newImg.data[i]);
        if (diff > 50) {
            ctx.fillStyle = "rgba(0, 255, 0, 0.2)";
            ctx.fillRect((i/4) % canvas.width, Math.floor((i/4) / canvas.width), 20, 20);
        }
    }
}

init();
