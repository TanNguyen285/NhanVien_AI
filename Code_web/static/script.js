/* =========================================
   EmotionScan v2 — script.js
   Server-side camera edition:
   - Video hiển thị qua <img id="mjpeg"> MJPEG stream
   - Canvas overlay vẽ bounding boxes lên trên img
   - /predict (GET) polling mỗi 300ms để lấy detections
   - Scan: server tự vòng nội bộ, client chỉ polling /scan/status
   ========================================= */

const mjpeg       = document.getElementById('mjpeg');
const canvas      = document.getElementById('overlay');
const ctx         = canvas.getContext('2d');
const emotionPill = document.getElementById('emotionPill');
const emotionIcon = document.getElementById('emotionIcon');
const emotionLabel= document.getElementById('emotionLabel');
const emotionConf = document.getElementById('emotionConf');
const faceCount   = document.getElementById('faceCount');
const fpsDisplay  = document.getElementById('fpsDisplay');
const logBody     = document.getElementById('logBody');
const emptyState  = document.getElementById('emptyState');
const logCount    = document.getElementById('logCount');
const statOut     = document.getElementById('statOut');
const statHappy   = document.getElementById('statHappy');
const toast       = document.getElementById('toast');

// App state
let currentData   = { emotion: 'Neutral', confidence: 0, allScores: {} };
let currentDets   = [];   // detections mới nhất để vẽ overlay
let logs          = [];
let countOut      = 0;
let countHappy    = 0;
let moodCountsOut = {};

// Scan state
let scanActive    = false;
let statusPoller  = null;
let scanTimer     = null;
let scanCountdown = 0;

// Predict polling
let predictPoller    = null;
let predictInFlight  = false;
// FPS UNLOCKED - Chạy tối đa tốc độ (không giới hạn)

// FPS tracking (đếm lần overlay được cập nhật)
let fpsCount = 0;
let fpsTs    = performance.now();

const EMO_META = {
    'Angry':    { icon: '😠', cls: 'angry'    },
    'Disgust':  { icon: '🤢', cls: 'disgust'  },
    'Fear':     { icon: '😨', cls: 'fear'     },
    'Happy':    { icon: '😄', cls: 'happy'    },
    'Neutral':  { icon: '😐', cls: 'neutral'  },
    'Sad':      { icon: '😢', cls: 'sad'      },
    'Surprise': { icon: '😲', cls: 'surprise' },
};

const MOOD_COLORS = {
    'Happy':    '#22c55e',
    'Neutral':  '#64748b',
    'Sad':      '#38bdf8',
    'Angry':    '#f43f5e',
    'Surprise': '#facc15',
    'Fear':     '#fb923c',
    'Disgust':  '#a78bfa',
};

// ── Canvas overlay sync với MJPEG img ────────────────────────────────────────

function syncCanvas() {
    // Đồng bộ kích thước canvas với img element
    const w = mjpeg.clientWidth  || mjpeg.naturalWidth  || 640;
    const h = mjpeg.clientHeight || mjpeg.naturalHeight || 480;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width  = w;
        canvas.height = h;
    }
}

// Scale bbox từ tọa độ ảnh gốc (640x480) sang kích thước hiển thị
function scaleBbox(x1, y1, x2, y2) {
    const scaleX = canvas.width  / (mjpeg.naturalWidth  || 640);
    const scaleY = canvas.height / (mjpeg.naturalHeight || 480);
    return [
        Math.round(x1 * scaleX),
        Math.round(y1 * scaleY),
        Math.round(x2 * scaleX),
        Math.round(y2 * scaleY),
    ];
}

function drawDetections(results) {
    syncCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!results || results.length === 0) return;

    results.forEach((p, i) => {
        const [x1, y1, x2, y2] = scaleBbox(...p.bbox);
        const personLabel = `Người ${i + 1}`;
        const meta    = EMO_META[p.label] || EMO_META['Neutral'];
        const isHappy = p.label === 'Happy';

        ctx.strokeStyle = isHappy ? '#22c55e' : '#5b7cf6';
        ctx.lineWidth   = 2.5;
        ctx.shadowColor = isHappy ? '#22c55e' : '#5b7cf6';
        ctx.shadowBlur  = 8;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.shadowBlur  = 0;

        const tagText = `${personLabel} • ${meta.icon} ${p.label} ${p.score.toFixed(0)}%`;
        ctx.font       = 'bold 13px monospace';
        const tw       = ctx.measureText(tagText).width + 16;
        const tagY     = Math.max(y1 - 30, 0);

        ctx.fillStyle  = isHappy ? '#22c55e' : '#5b7cf6';
        roundRect(ctx, x1, tagY, tw, 24, 5);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.fillText(tagText, x1 + 8, tagY + 16);
    });

    // FPS counter
    fpsCount++;
    const now = performance.now();
    if (now - fpsTs >= 1000) {
        fpsDisplay.textContent = `${fpsCount} fps`;
        fpsCount = 0;
        fpsTs    = now;
    }
}

// ── Predict polling — chạy liên tục khi không scan ───────────────────────────

function startPredictPolling() {
    if (predictPoller) return;
    // Chạy liên tục, không delay - FPS tối đa
    const poll = async () => {
        if (!scanActive) { predictPoller = null; return; }
        if (!predictInFlight) {
            predictInFlight = true;
            try {
                const res = await fetch('/predict');
                if (res.ok) {
                    const data = await res.json();
                    updateUIWithDetections(data.results || []);
                }
            } catch (_) {}
            predictInFlight = false;
        }
        predictPoller = setTimeout(poll, 0);  // 0ms delay = tối đa FPS
    };
    predictPoller = setTimeout(poll, 0);
}

function stopPredictPolling() {
    if (predictPoller) clearTimeout(predictPoller);
    predictPoller = null;
    predictInFlight = false;
}

// Không dùng fetchPredict riêng lẻ nữa - đã tích hợp vào poll

// ── Scan session — server-side, client chỉ chờ ───────────────────────────────

async function startScan() {
    if (scanActive) return;

    try {
        const res  = await fetch('/scan/start', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'already_running') {
            showToast('⚠ Phiên scan đang chạy!');
            return;
        }
        if (data.status !== 'started') {
            showToast('⚠ Không thể bắt đầu scan');
            return;
        }
    } catch (_) {
        showToast('⚠ Không kết nối được server');
        return;
    }

    scanActive    = true;
    scanCountdown = 5;
    updateScanBtn(true, scanCountdown);
    showToast('🔍 Đang scan cảm xúc...');

    // Bắt đầu polling /predict để vẽ real-time detections
    startPredictPolling();

    // Đếm ngược UI
    scanTimer = setInterval(() => {
        scanCountdown--;
        updateScanBtn(true, scanCountdown);
        if (scanCountdown <= 0) clearInterval(scanTimer);
    }, 1000);

    // Polling status để check xem scan xong chưa
    statusPoller = setInterval(pollScanStatus, 500);
}

async function pollScanStatus() {
    try {
        const res  = await fetch('/scan/status');
        const data = await res.json();
        if (data.status === 'done') {
            stopScan();
            showScanResult(data.result);
        } else if (data.status === 'idle') {
            // scan kết thúc không có kết quả
            stopScan();
            showToast('⚠ Scan kết thúc không có kết quả');
        }
    } catch (_) {}
}

function stopScan() {
    scanActive = false;
    clearInterval(scanTimer);
    clearInterval(statusPoller);
    scanTimer = statusPoller = null;
    stopPredictPolling();   // dừng polling /predict
    updateScanBtn(false);
    // Xóa khung bao nhưng giữ bars
    currentDets = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function updateScanBtn(active, countdown) {
    const btn = document.getElementById('btnScan');
    if (!btn) return;
    if (active) {
        const label = countdown !== undefined ? `⏳ ${countdown}s...` : '⏳ Đang scan...';
        btn.innerHTML = label;
        btn.disabled  = true;
    } else {
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Scan cảm xúc (5s)`;
        btn.disabled = false;
    }
}

function showScanResult(result) {
    if (result.error) {
        showToast('⚠ ' + result.error);
        faceCount.textContent = '0 khuôn mặt';
        return;
    }

    const stars      = result.star_rating;
    const summary    = result.summary;
    const confidence = result.average_confidence;
    const dominant   = result.dominant_emotion;
    const meta       = EMO_META[dominant] || EMO_META['Neutral'];

    emotionIcon.textContent  = meta.icon;
    emotionLabel.textContent = dominant.toUpperCase();
    emotionConf.textContent  = `Độ hài lòng: ${stars} ★ | Độ tin cậy: ${confidence.toFixed(0)}%`;
    Object.values(EMO_META).forEach(m => emotionPill.classList.remove(m.cls));
    emotionPill.classList.add(meta.cls);

    currentData = {
        emotion:    dominant,
        confidence: `${confidence.toFixed(0)}%`,
        allScores:  {},
        stars,
        summary,
    };

    showToast(`✅ ${stars}★ — ${summary} (${dominant})`);
    logEntry('CHECK-OUT');
}

// ── UI Updates ────────────────────────────────────────────────────────────────

function updateUIWithDetections(results) {
    // Chỉ vẽ bounding boxes + cập nhật bars khi scan chạy
    if (!scanActive) return;
    
    if (!results || results.length === 0) {
        faceCount.textContent = '0 khuôn mặt';
        currentDets = [];
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
    }

    currentDets = results;
    drawDetections(results);
    faceCount.textContent = `${results.length} khuôn mặt`;
    
    // Cập nhật emotion bars real-time khi scan
    results.sort((a, b) => a.bbox[0] - b.bbox[0]);
    const primary = results[0];
    updateBars(primary.all_scores || {}, primary.label, primary.score);
}

function updateUI(results) {
    if (scanActive) return;   // khi scan đang chạy, không cập nhật pill từ predict

    if (!results || results.length === 0) {
        faceCount.textContent = '0 khuôn mặt';
        currentDets = [];
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setNeutralPill();
        clearBars();
        return;
    }

    currentDets = results;
    drawDetections(results);

    faceCount.textContent = `${results.length} khuôn mặt`;
    results.sort((a, b) => a.bbox[0] - b.bbox[0]);

    const primary = results[0];
    const meta    = EMO_META[primary.label] || EMO_META['Neutral'];

    currentData = {
        emotion:    primary.label,
        confidence: primary.score.toFixed(1),
        allScores:  primary.all_scores || {},
    };

    emotionIcon.textContent  = meta.icon;
    emotionLabel.textContent = primary.label.toUpperCase();
    emotionConf.textContent  = `${primary.score.toFixed(1)}%`;

    Object.values(EMO_META).forEach(m => emotionPill.classList.remove(m.cls));
    emotionPill.classList.add(meta.cls);
    updateBars(primary.all_scores || {}, primary.label, primary.score);
}

function setNeutralPill() {
    Object.values(EMO_META).forEach(m => emotionPill.classList.remove(m.cls));
    emotionIcon.textContent  = '😐';
    emotionLabel.textContent = 'Chờ nhận diện...';
    emotionConf.textContent  = '';
}

function clearBars() {
    document.querySelectorAll('.emo-bar-fill').forEach(b => b.style.width = '0%');
    document.querySelectorAll('.emo-bar-val').forEach(v => v.textContent = '0%');
}

function updateBars(allScores, topLabel, topScore) {
    document.querySelectorAll('.emo-bar-row').forEach(row => {
        const emo  = row.dataset.emo;
        const fill = row.querySelector('.emo-bar-fill');
        const val  = row.querySelector('.emo-bar-val');
        const score = allScores[emo] !== undefined ? allScores[emo] : (emo === topLabel ? topScore : 0);
        fill.style.width   = `${score.toFixed(1)}%`;
        val.textContent    = `${score.toFixed(0)}%`;
        val.style.color    = emo === topLabel ? '#fff' : '';
        fill.style.opacity = emo === topLabel ? '1' : '0.55';
    });
}

// ── Mood chart ────────────────────────────────────────────────────────────────

function updateMoodChart() {
    renderMoodGroup('moodChartOut', 'moodLegendOut', moodCountsOut);
}

function renderMoodGroup(chartId, legendId, counts) {
    const donut  = document.getElementById(chartId);
    const legend = document.getElementById(legendId);
    const labels = ['Happy','Neutral','Sad','Angry','Surprise','Fear','Disgust'];
    const items  = labels
        .map(label => ({ label, value: counts[label] || 0, color: MOOD_COLORS[label] || '#c8cde8' }))
        .filter(item => item.value > 0);
    const total  = items.reduce((s, i) => s + i.value, 0);

    if (total === 0) {
        donut.innerHTML  = '<div class="chart-empty">Chưa có dữ liệu</div>';
        legend.innerHTML = '';
        return;
    }

    let deg = 0;
    const segs = items.map(item => {
        const slice = Math.round((item.value / total) * 360);
        const from  = deg;
        deg += slice;
        return `${item.color} ${from}deg ${deg}deg`;
    });

    donut.innerHTML = `
        <div class="chart-donut" style="background: radial-gradient(circle at center, var(--panel) 58%, transparent 59%), conic-gradient(${segs.join(', ')});">
            <span>${total} lượt</span>
        </div>`;

    legend.innerHTML = items.map(item => `
        <div class="chart-legend-item">
            <span class="legend-color" style="background:${item.color}"></span>
            <span>${item.label}: ${item.value}</span>
        </div>`).join('');
}

// ── Logging ───────────────────────────────────────────────────────────────────

function logEntry(action) {
    if (action !== 'CHECK-OUT') return;

    const now   = new Date().toLocaleTimeString('vi-VN');
    const emo   = currentData.emotion;
    const conf  = currentData.confidence;
    const stars = currentData.stars || '';

    const entry = { time: now, action, emotion: emo, confidence: conf, stars, user: LOGGED_IN_USER };
    logs.push(entry);

    countOut++;
    statOut.textContent      = countOut;
    moodCountsOut[emo]       = (moodCountsOut[emo] || 0) + 1;
    if (emo === 'Happy') { countHappy++; statHappy.textContent = countHappy; }

    logCount.textContent     = `${logs.length} sự kiện`;
    emptyState.style.display = 'none';
    updateMoodChart();

    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td style="font-family:monospace;font-size:12px;color:var(--muted)">${now}</td>
        <td><span class="badge badge-out">← RA</span></td>
        <td><span class="badge-emo">${EMO_META[emo]?.icon || ''} ${emo}</span></td>
        <td style="text-align:center;color:#fbbf24;font-weight:bold;">${stars} ★</td>
        <td style="text-align:center;">${conf}</td>`;
    tr.style.animation = 'fadeIn 0.2s ease';
    logBody.prepend(tr);

    // Reset bars + pill sau khi log
    clearBars();
    setNeutralPill();
    showToast(`✓ Check-out: ${emo} (${stars}★ - ${conf})`);
}

// ── Export ────────────────────────────────────────────────────────────────────

document.getElementById('btnExport').onclick = () => {
    if (logs.length === 0) { showToast('⚠ Chưa có dữ liệu!'); return; }
    const bom    = '\uFEFF';
    const header = 'Nhân viên,Giờ,Cảm xúc,Độ hài lòng,Độ tin cậy\n';
    const rows   = logs.map(e => `${e.user},${e.time},${e.emotion},${e.stars},${e.confidence}`).join('\n');
    const blob   = new Blob([bom + header + rows], { type: 'text/csv;charset=utf-8;' });
    const url    = URL.createObjectURL(blob);
    const a      = document.createElement('a'); a.href = url; a.download = `EmotionLog_${Date.now()}.csv`; a.click();
    URL.revokeObjectURL(url);

    logs = []; countOut = 0; countHappy = 0;
    moodCountsOut = {};
    statOut.textContent = statHappy.textContent = '0';
    logBody.innerHTML  = '';
    logCount.textContent     = '0 sự kiện';
    emptyState.style.display = '';
    updateMoodChart();
    showToast('✓ Đã xuất & reset!');
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
}

let toastTimer;
function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 2200);
}

// ── Buttons ───────────────────────────────────────────────────────────────────

document.getElementById('btnScan').onclick = () => startScan();

// ── CSS keyframe ──────────────────────────────────────────────────────────────

const style = document.createElement('style');
style.textContent = '@keyframes fadeIn { from { opacity:0; transform:translateY(-6px) } to { opacity:1; transform:none } }';
document.head.appendChild(style);

// ── MJPEG img: đồng bộ canvas khi ảnh load lần đầu ──────────────────────────

mjpeg.addEventListener('load', () => {
    syncCanvas();
});

// Nếu img đã load rồi (cached), sync ngay
if (mjpeg.complete && mjpeg.naturalWidth) {
    syncCanvas();
}

// Resize observer: sync canvas khi window resize
if (window.ResizeObserver) {
    new ResizeObserver(syncCanvas).observe(mjpeg);
}

// ── Start ─────────────────────────────────────────────────────────────────────

updateMoodChart();
// Không tự động polling emotion detection - chỉ chạy khi ấn Scan
// startPredictPolling();