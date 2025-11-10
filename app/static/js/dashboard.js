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
    initTabs();
    ensureCards();
    initChart();
    refresh();
    setInterval(refresh, 1500);
    try {
        assignFigure('latencyTitle', 'latencyCaption', 'Latency (P95 over time)', 'P95 latency sampled from live requests.');
        assignFigure('coverageTitle', 'coverageCaption', 'Streaming coverage', 'Streaming conformal coverage vs target (dashed).');
    } catch (e) {
        // ignore
    }
    initCoverageChart();
    refreshCoverage();
    setInterval(refreshCoverage, 3000);
    refreshOffline();
    setInterval(refreshOffline, 30000);
    refreshAblation();
    setInterval(refreshAblation, 30000);
    const btnH = document.getElementById('refreshHealth');
    if (btnH) btnH.addEventListener('click', refreshHealth);
    const btnM = document.getElementById('refreshMetrics');
    if (btnM) btnM.addEventListener('click', refreshMetrics);
    const btnEx = document.getElementById('loadPredictExample');
    if (btnEx) btnEx.addEventListener('click', loadPredictExample);
    const btnSend = document.getElementById('sendPredict');
    if (btnSend) btnSend.addEventListener('click', sendPredict);
    const btnAbl = document.getElementById('genAblationBtn');
    if (btnAbl) btnAbl.addEventListener('click', generateAblation);
}

document.addEventListener('DOMContentLoaded', start);

// --- Simple latency chart (p95 over time) ---
const chartState = {
    values: [],
    maxPoints: 60,
};

const figureNumbering = {next: 3};

function assignFigure(titleId, captionId, titleText, captionText) {
    const t = document.getElementById(titleId);
    const c = document.getElementById(captionId);
    if (!t || !c) return;
    let num = 0;
    if (titleId === 'latencyTitle') num = 1;
    else if (titleId === 'coverageTitle') num = 2;
    else num = figureNumbering.next++;
    t.textContent = `Figure ${num}: ${titleText}`;
    c.textContent = captionText;
}

// --- Tabs ---
function initTabs() {
    const nav = document.getElementById('nav');
    if (!nav) return;
    nav.addEventListener('click', (e) => {
        const btn = e.target.closest('.tab-btn');
        if (!btn) return;
        const tab = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        ['dashboard', 'health', 'predict'].forEach(name => {
            const sec = document.getElementById(`section-${name}`);
            if (sec) sec.style.display = (name === tab) ? 'block' : 'none';
        });
        // include metrics tab
        const m = document.getElementById('section-metrics');
        if (m) m.style.display = (tab === 'metrics') ? 'block' : 'none';
    });
}

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

async function refreshMetrics() {
    try {
        const data = await fetchJSON('/metrics');
        const pre = document.getElementById('metricsJson');
        if (pre) pre.textContent = JSON.stringify(data, null, 2);
    } catch (e) {
        console.error('Failed to fetch metrics', e);
    }
}

// --- Streaming coverage chart ---
const coverageState = {
    idx: [],
    cov: [],
    covPos: [],
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
        const [data, summary] = await Promise.all([
            fetchJSON('/metrics/stream?limit=200'),
            fetch('/artifacts/stream_summary.json', {cache: 'no-store'}).then(r => r.ok ? r.json() : null).catch(() => null),
        ]);
        coverageState.idx = data.idx;
        coverageState.cov = data.coverage;
        coverageState.covPos = Array.isArray(data.coverage_pos) ? data.coverage_pos : [];
        drawCoverageChart();
        if (summary) {
            const el = document.getElementById('streamConfig');
            if (el) {
                const target = (1 - (summary.alpha ?? 0.05));
                const mode = summary.mode;
                const windowSize = summary.window;
                const decay = summary.decay;
                const delay = summary.label_delay;
                const warmup = summary.warmup;
                const nEff = summary.effective_n ? Number(summary.effective_n).toFixed(1) : '-';
                let cfg = `Target ${(target * 100).toFixed(1)}% · `;
                if (mode === 'window') {
                    cfg += `mode=window W=${windowSize}`;
                } else {
                    cfg += `mode=exp λ=${decay}`;
                }
                cfg += ` · label_delay=${delay} · warmup=${warmup} · n_eff≈${nEff}`;
                el.textContent = cfg;
            }
            const statsEl = document.getElementById('streamStats');
            if (statsEl) {
                const latest = Array.isArray(coverageState.cov) && coverageState.cov.length ? coverageState.cov[coverageState.cov.length - 1] : null;
                const latestPos = Array.isArray(coverageState.covPos) && coverageState.covPos.length ? coverageState.covPos[coverageState.covPos.length - 1] : null;
                const targetPct = ((1 - (summary.alpha ?? 0.05)) * 100).toFixed(1);
                const actualPct = latest != null ? (latest * 100).toFixed(1) : '-';
                const fraudPct = latestPos != null && Number.isFinite(latestPos) ? (latestPos * 100).toFixed(1) : '-';
                const avgSet = summary.avg_set_size != null ? Number(summary.avg_set_size).toFixed(2) : '-';
                statsEl.textContent = `Coverage (target ${targetPct}%): ${actualPct}% · Fraud: ${fraudPct}% · Avg set size: ${avgSet}`;
            }
        }
    } catch (e) {
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
    const valsPos = coverageState.covPos && coverageState.covPos.length ? coverageState.covPos : [];
    if (vals.length < 2) return;
    const allVals = valsPos.length ? vals.concat(valsPos.filter(v => Number.isFinite(v))) : vals;
    const minY = Math.max(0, Math.min(...allVals) - 0.05);
    const maxY = Math.min(1, Math.max(...allVals) + 0.05);

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

    // fraud-class coverage line (if available)
    if (valsPos.length > 1) {
        ctx.strokeStyle = '#6a0dad'; // purple
        ctx.beginPath();
        for (let i = 0; i < valsPos.length; i++) {
            const x = 40 + (i / (valsPos.length - 1)) * (w - 50);
            const y = (h - 20) - ((valsPos[i] - minY) / (maxY - minY)) * (h - 30);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // target line at 0.95 if within range
    const target = 0.95;
    if (target >= minY && target <= maxY) {
        const yTarget = (h - 20) - ((target - minY) / (maxY - minY)) * (h - 30);
        if (ctx.setLineDash) ctx.setLineDash([5, 5]);
        ctx.strokeStyle = '#e33';
        ctx.beginPath();
        ctx.moveTo(40, yTarget);
        ctx.lineTo(w - 10, yTarget);
        ctx.stroke();
        if (ctx.setLineDash) ctx.setLineDash([]);
    }
}

// --- Offline metrics & images ---
async function refreshOffline() {
    const container = document.getElementById('offlineMetrics');
    const imgContainer = document.getElementById('offlineImages');
    if (!container || !imgContainer) return;
    try {
        // Reset figure numbering for offline images on each refresh so numbers stay stable
        if (typeof figureNumbering !== 'undefined' && figureNumbering && typeof figureNumbering.next === 'number') {
            figureNumbering.next = 3;
        }
        const data = await fetchJSON('/metrics/offline');
        const pre = data.metrics_pre || {};
        const post = data.metrics_post || {};
        // Build metrics table
        const rows = [
            ['ROC-AUC', pre.roc_auc, post.roc_auc],
            ['PR-AUC', pre.pr_auc, post.pr_auc],
            ['Brier', pre.brier, post.brier],
            ['NLL', pre.nll, post.nll],
        ];
        let html = '<table><thead><tr><th>Metric</th><th>Pre</th><th>Post</th></tr></thead><tbody>';
        for (const [k, a, b] of rows) {
            const fa = (a ?? '-');
            const fb = (b ?? '-');
            html += `<tr><td>${k}</td><td>${Number.isFinite(fa) ? Number(fa).toFixed(4) : fa}</td><td>${Number.isFinite(fb) ? Number(fb).toFixed(4) : fb}</td></tr>`;
        }
        html += '</tbody></table>';
        container.innerHTML = html;
        imgContainer.innerHTML = '';
        const maybe = [];
        const descriptions = {
            reliability_pre: 'Reliability diagram before calibration.',
            reliability_post: 'Reliability diagram after calibration.',
            roc: 'Receiver Operating Characteristic (ROC) curve.',
            pr: 'Precision–Recall (PR) curve.',
            hist: 'Predicted score histogram.',
        };
        // Pair reliability images if both exist
        const relPre = data?.artifacts?.['reliability_pre'];
        const relPost = data?.artifacts?.['reliability_post'];
        if (relPre && relPost) {
            const row = document.createElement('div');
            row.className = 'grid two-col';
            const items = [
                {key: 'reliability_pre', label: 'Reliability (pre)', url: relPre},
                {key: 'reliability_post', label: 'Reliability (post)', url: relPost},
            ];
            for (const it of items) {
                const card = document.createElement('div');
                card.className = 'card';
                const img = document.createElement('img');
                img.src = `/${it.url}?t=${Date.now()}`;
                img.alt = it.label;
                const h = document.createElement('h3');
                const num = figureNumbering.next++;
                h.textContent = `Figure ${num}: ${it.label}`;
                const p = document.createElement('p');
                p.className = 'figure-caption';
                p.textContent = descriptions[it.key] || 'Figure.';
                card.appendChild(h);
                card.appendChild(img);
                card.appendChild(p);
                row.appendChild(card);
            }
            imgContainer.appendChild(row);
        }
        // Pair ROC and PR side by side if both exist
        const rocUrl = data?.artifacts?.['roc'];
        const prUrl = data?.artifacts?.['pr'];
        if (rocUrl && prUrl) {
            const row = document.createElement('div');
            row.className = 'grid two-col';
            const items = [
                {key: 'roc', label: 'ROC', url: rocUrl},
                {key: 'pr', label: 'PR', url: prUrl},
            ];
            for (const it of items) {
                const card = document.createElement('div');
                card.className = 'card';
                const img = document.createElement('img');
                img.src = `/${it.url}?t=${Date.now()}`;
                img.alt = it.label;
                const h = document.createElement('h3');
                const num = figureNumbering.next++;
                h.textContent = `Figure ${num}: ${it.label}`;
                const p = document.createElement('p');
                p.className = 'figure-caption';
                p.textContent = descriptions[it.key] || 'Figure.';
                card.appendChild(h);
                card.appendChild(img);
                card.appendChild(p);
                row.appendChild(card);
            }
            imgContainer.appendChild(row);
        }
        // Place histogram next to Streaming Coverage if slot exists
        const histUrl = data?.artifacts?.['hist'];
        const histSlot = document.getElementById('histSlot');
        if (histUrl && histSlot) {
            histSlot.innerHTML = '';
            const h = document.createElement('h3');
            const num = figureNumbering.next++;
            h.textContent = `Figure ${num}: Score histogram`;
            const img = document.createElement('img');
            img.src = `/${histUrl}?t=${Date.now()}`;
            img.alt = 'Score histogram';
            const p = document.createElement('p');
            p.className = 'figure-caption';
            p.textContent = descriptions['hist'];
            histSlot.appendChild(h);
            histSlot.appendChild(img);
            histSlot.appendChild(p);
        }
        for (const {key, label} of maybe) {
            const url = data?.artifacts?.[key];
            if (!url) continue;
            const card = document.createElement('div');
            card.className = 'card';
            const img = document.createElement('img');
            img.src = `/${url}?t=${Date.now()}`;
            img.alt = label;
            const h = document.createElement('h3');
            const num = figureNumbering.next++;
            h.textContent = `Figure ${num}: ${label}`;
            const p = document.createElement('p');
            p.className = 'figure-caption';
            p.textContent = descriptions[key] || 'Figure.';
            card.appendChild(h);
            card.appendChild(img);
            card.appendChild(p);
            imgContainer.appendChild(card);
        }
        // Update stream image (if exists)
        const streamImg = document.getElementById('affh_stream_img') || document.getElementById('streamImg');
        if (streamImg) {
            const url = '/artifacts/stream_coverage.png';
            // Optimistically set; if 404, ignore
            fetch(url, {method: 'HEAD'})
                .then(resp => {
                    if (resp.ok) {
                        streamImg.src = `${url}?t=${Date.now()}`;
                    }
                })
                .catch(() => {
                });
        }
    } catch (e) {
    }
}

// --- Streaming ablation table ---
async function refreshAblation() {
    const container = document.getElementById('ablationTable');
    if (!container) return;
    try {
        // Prefer JSON; fallback to CSV
        let rows = null;
        try {
            const json = await fetchJSON('/artifacts/stream_ablation.json');
            if (Array.isArray(json)) {
                rows = json;
            }
        } catch (_) {
        }
        if (!rows) {
            const resp = await fetch('/artifacts/stream_ablation.csv', {cache: 'no-store'});
            if (resp.ok) {
                const text = await resp.text();
                const lines = text.trim().split('\n');
                const header = lines[0].split(',');
                rows = lines.slice(1).map(line => {
                    const parts = line.split(',');
                    const obj = {};
                    header.forEach((h, i) => obj[h] = parts[i]);
                    return obj;
                });
            }
        }
        const btn = document.getElementById('genAblationBtn');
        if (!rows || !rows.length) {
            if (btn) btn.style.display = 'inline-block';
            container.innerHTML = '<div class="muted">No ablation results available yet.</div>';
            return;
        }
        if (btn) btn.style.display = 'none';
        // Render concise table: Param, Value, Coverage, Under-coverage gap (target - cov), n_eff
        let html = '<table><thead><tr><th>Mode</th><th>Param</th><th>Value</th><th>Coverage</th><th>Violation (target gap)</th><th>n_eff</th></tr></thead><tbody>';
        for (const r of rows) {
            const cov = r.final_coverage != null ? Number(r.final_coverage).toFixed(4) : '-';
            const viol = r.final_violation_rate != null ? Number(r.final_violation_rate).toFixed(4) : '-';
            const neff = r.effective_n != null ? Number(r.effective_n).toFixed(1) : '-';
            html += `<tr><td>${r.mode}</td><td>${r.param}</td><td>${r.value}</td><td>${cov}</td><td>${viol}</td><td>${neff}</td></tr>`;
        }
        html += '</tbody></table>';
        container.innerHTML = html;
    } catch (e) {
        console.error('Failed to refresh ablation', e);
    }
}

async function generateAblation() {
    const btn = document.getElementById('genAblationBtn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Running...';
    }
    try {
        await fetch('/metrics/stream/ablate', {method: 'POST'});
        // poll for results
        const poll = setInterval(async () => {
            await refreshAblation();
            const container = document.getElementById('ablationTable');
            if (container && container.querySelector('table')) {
                clearInterval(poll);
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = 'Generate ablation';
                }
            }
        }, 5000);
        setTimeout(() => clearInterval(poll), 15 * 60 * 1000); // safety stop after 15 min
    } catch (e) {
        console.error('Failed to start ablation', e);
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Generate ablation';
        }
    }
}

// --- Health & Predict ---
async function refreshHealth() {
    try {
        const data = await fetchJSON('/health');
        const pre = document.getElementById('healthJson');
        if (pre) pre.textContent = JSON.stringify(data, null, 2);
    } catch (e) {
        console.error('Failed to fetch health', e);
    }
}

async function loadPredictExample() {
    try {
        const data = await fetchJSON('/predict/schema');
        const ta = document.getElementById('predictPayload');
        if (ta) ta.value = JSON.stringify(data.example, null, 2);
    } catch (e) {
        console.error('Failed to load example schema', e);
    }
}

async function sendPredict() {
    try {
        const ta = document.getElementById('predictPayload');
        const out = document.getElementById('predictOut');
        if (!ta) return;
        const payload = JSON.parse(ta.value);
        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });
        const text = await res.text();
        try {
            const json = JSON.parse(text);
            if (out) out.textContent = JSON.stringify(json, null, 2);
        } catch (_) {
            if (out) out.textContent = text;
        }
    } catch (e) {
        console.error('Failed to send prediction', e);
    }
}

