async function fetchJSON(path) {
    const res = await fetch(path, {cache: 'no-store'});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
}

function ensureCards() {
    const cards = document.getElementById('cards');
    if (!cards) return;
    if (!cards.dataset.init) {
        cards.innerHTML = `
      <div class="card"><div>P50</div><div id="p50">-</div></div>
      <div class="card"><div>P95</div><div id="p95">-</div></div>
      <div class="card"><div>P99</div><div id="p99">-</div></div>
      <div class="card"><div>Count</div><div id="count">-</div></div>
      <div class="card"><div>RPS</div><div id="rps">-</div></div>
    `;
        cards.dataset.init = '1';
    }
}

async function refresh() {
    ensureCards();
    try {
        const metrics = await fetchJSON('/metrics');
        document.getElementById('p50').textContent = metrics.p50_ms.toFixed(2) + ' ms';
        document.getElementById('p95').textContent = metrics.p95_ms.toFixed(2) + ' ms';
        document.getElementById('p99').textContent = metrics.p99_ms.toFixed(2) + ' ms';
        document.getElementById('count').textContent = String(metrics.count);
        document.getElementById('rps').textContent = metrics.rps.toFixed(2);
        updateChart(metrics);
    } catch (e) {
        // eslint-disable-next-line no-console
        console.error('Failed to refresh metrics', e);
    }
}

function start() {
    ensureCards();
    initChart();
    refresh();
    setInterval(refresh, 1500);
    initCoverageChart();
    refreshCoverage();
    setInterval(refreshCoverage, 3000);
}

document.addEventListener('DOMContentLoaded', start);

// --- Simple latency chart (p95 over time) ---
const chartState = {
    values: [],
    maxPoints: 60,
};

function initChart() {
    const canvas = document.getElementById('latencyChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#333';
    ctx.fillText('P95 latency (ms)', 10, 16);
}

function updateChart(metrics) {
    const canvas = document.getElementById('latencyChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width || 640;
    const h = canvas.height || 240;
    canvas.width = w;
    canvas.height = h;

    chartState.values.push(metrics.p95_ms);
    if (chartState.values.length > chartState.maxPoints) chartState.values.shift();

    const vals = chartState.values;
    const minY = Math.max(0, Math.min(...vals) - 5);
    const maxY = Math.max(minY + 1, Math.max(...vals) + 5);

    // axes
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#ddd';
    ctx.beginPath();
    ctx.moveTo(40, 10);
    ctx.lineTo(40, h - 20);
    ctx.lineTo(w - 10, h - 20);
    ctx.stroke();
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#555';
    ctx.fillText(maxY.toFixed(0), 5, 12);
    ctx.fillText(minY.toFixed(0), 5, h - 20);

    // line
    if (vals.length > 1) {
        ctx.strokeStyle = '#0077ff';
        ctx.beginPath();
        for (let i = 0; i < vals.length; i++) {
            const x = 40 + (i / (chartState.maxPoints - 1)) * (w - 50);
            const y = (h - 20) - ((vals[i] - minY) / (maxY - minY)) * (h - 30);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
}

// --- Streaming coverage chart ---
const coverageState = {
    idx: [],
    cov: [],
};

function initCoverageChart() {
    const canvas = document.getElementById('coverageChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#333';
    ctx.fillText('Coverage (last points)', 10, 16);
}

async function refreshCoverage() {
    const canvas = document.getElementById('coverageChart');
    if (!canvas) return;
    try {
        const data = await fetchJSON('/metrics/stream?limit=200');
        coverageState.idx = data.idx;
        coverageState.cov = data.coverage;
        drawCoverageChart();
    } catch (e) {
        // eslint-disable-next-line no-console
        console.error('Failed to refresh streaming coverage', e);
    }
}

function drawCoverageChart() {
    const canvas = document.getElementById('coverageChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width || 640;
    const h = canvas.height || 240;
    canvas.width = w;
    canvas.height = h;

    const vals = coverageState.cov;
    if (vals.length < 2) return;
    const minY = Math.max(0, Math.min(...vals) - 0.05);
    const maxY = Math.min(1, Math.max(...vals) + 0.05);

    // axes
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#ddd';
    ctx.beginPath();
    ctx.moveTo(40, 10);
    ctx.lineTo(40, h - 20);
    ctx.lineTo(w - 10, h - 20);
    ctx.stroke();
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#555';
    ctx.fillText(maxY.toFixed(2), 5, 12);
    ctx.fillText(minY.toFixed(2), 5, h - 20);

    // line
    ctx.strokeStyle = '#0a8';
    ctx.beginPath();
    for (let i = 0; i < vals.length; i++) {
        const x = 40 + (i / (vals.length - 1)) * (w - 50);
        const y = (h - 20) - ((vals[i] - minY) / (maxY - minY)) * (h - 30);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

