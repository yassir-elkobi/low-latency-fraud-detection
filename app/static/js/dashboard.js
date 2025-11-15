// Dashboard client-side logic for latency/coverage/offline metrics.
// Keep DOM updates lightweight and avoid heavy timers or blocking work.

/**
 * Fetch JSON with no-store cache to always get fresh metrics/artifacts.
 * @param {string} path
 * @returns {Promise<any>}
 */
async function fetchJSON(path) {
    const res = await fetch(path, {cache: 'no-store'});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
}

/**
 * Create metric cards once; subsequent calls are no-ops.
 */
function ensureCards() {
    const cards = document.getElementById('cards');
    if (!cards) return;
    if (!cards.dataset.init) {
        cards.innerHTML = `
      <div class="card"><div>P50</div><div id="p50">-</div></div>
      <div class="card"><div>P95</div><div id="p95">-</div></div>
      <div class="card"><div>P99</div><div id="p99">-</div></div>
      <div class="card"><div>Count</div><div id="count">-</div></div>
      <div class="card"><div>RPS (30s)</div><div id="rps">-</div></div>
      <div class="card"><div>RPS (5m)</div><div id="rps5">-</div></div>
    `;
        cards.dataset.init = '1';
    }
}

/**
 * Pull live /metrics and update the top cards and latency mini-chart.
 * Applies 5-minute gating for P95/P99 visibility thresholds.
 */
async function refresh() {
    ensureCards();
    try {
        const metrics = await fetchJSON('/metrics');
        const n = Number(metrics.count_5m || metrics.count || 0);
        const n5 = Number(metrics.count_5m || 0);
        const p95Ready = n5 >= 500;
        const p99Ready = n5 >= 10000;
        const ts = (typeof metrics.ts === 'number') ? new Date(metrics.ts * 1000) : new Date();
        const tsStr = ts.toLocaleTimeString();
        const p50Label = document.getElementById('p50')?.previousElementSibling;
        const p95Label = document.getElementById('p95')?.previousElementSibling;
        const p99Label = document.getElementById('p99')?.previousElementSibling;
        if (p50Label) p50Label.textContent = `P50 (5m N=${n5} · ${tsStr})`;
        if (p95Label) p95Label.textContent = `P95 (5m N=${n5} · ${tsStr})${p95Ready ? '' : ' · low N'}`;
        if (p99Label) p99Label.textContent = `P99 (5m N=${n5} · ${tsStr})${p99Ready ? '' : ' · low N'}`;
        // values
        document.getElementById('p50').textContent = Number(metrics.p50_ms).toFixed(2) + ' ms';
        document.getElementById('p95').textContent = p95Ready ? (Number(metrics.p95_ms).toFixed(2) + ' ms') : '—';
        document.getElementById('p99').textContent = p99Ready ? (Number(metrics.p99_ms).toFixed(2) + ' ms') : '—';
        document.getElementById('count').textContent = String(metrics.count);
        document.getElementById('rps').textContent = metrics.rps.toFixed(2);
        const rps5 = (typeof metrics.rps_5m === 'number') ? Number(metrics.rps_5m).toFixed(2) : '0.00';
        document.getElementById('rps5').textContent = rps5;
        updateChart(metrics);
    } catch (e) {
        console.error('Failed to refresh metrics', e);
    }
}

/**
 * Initialize tabs, charts, event listeners and periodic refresh loops.
 */
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

// --- Latency sparkline state ---
const chartState = {
    values: [],
    maxPoints: 60,
};

const figureNumbering = {next: 3};

/**
 * Assign a figure number and title + caption text to slots.
 */
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

/**
 * Tabs controller: toggles visible sections.
 */
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

/**
 * Prepare latency sparkline canvas.
 */
function initChart() {
    const canvas = document.getElementById('latencyChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const css = getComputedStyle(document.documentElement);
    ctx.font = css.getPropertyValue('--font-small').trim() || '12px sans-serif';
    ctx.fillStyle = css.getPropertyValue('--color-text').trim() || '#333';
    ctx.fillText('P95 latency (ms)', 10, 16);
}

/**
 * Append the latest P95 value and redraw the sparkline with simple axes.
 */
function updateChart(metrics) {
    const canvas = document.getElementById('latencyChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const css = getComputedStyle(document.documentElement);
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
    ctx.strokeStyle = css.getPropertyValue('--color-axis').trim() || '#ddd';
    ctx.beginPath();
    ctx.moveTo(40, 10);
    ctx.lineTo(40, h - 20);
    ctx.lineTo(w - 10, h - 20);
    ctx.stroke();
    ctx.font = css.getPropertyValue('--font-small').trim() || '12px sans-serif';
    ctx.fillStyle = css.getPropertyValue('--color-axis-text').trim() || '#555';
    ctx.fillText(maxY.toFixed(0), 5, 12);
    ctx.fillText(minY.toFixed(0), 5, h - 20);

    // line
    if (vals.length > 1) {
        ctx.strokeStyle = css.getPropertyValue('--color-line-latency').trim() || '#0077ff';
        ctx.beginPath();
        for (let i = 0; i < vals.length; i++) {
            const x = 40 + (i / (chartState.maxPoints - 1)) * (w - 50);
            const y = (h - 20) - ((vals[i] - minY) / (maxY - minY)) * (h - 30);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
}

/**
 * Dump raw /metrics JSON into the Metrics tab.
 */
async function refreshMetrics() {
    try {
        const data = await fetchJSON('/metrics');
        const pre = document.getElementById('metricsJson');
        if (pre) pre.textContent = JSON.stringify(data, null, 2);
    } catch (e) {
        console.error('Failed to fetch metrics', e);
    }
}

// --- Streaming coverage state ---
const coverageState = {
    idx: [],
    cov: [],
    covPos: [],
    covNeg: [],
};

/**
 * Prepare streaming coverage canvas.
 */
function initCoverageChart() {
    const canvas = document.getElementById('coverageChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const css = getComputedStyle(document.documentElement);
    ctx.font = css.getPropertyValue('--font-small').trim() || '12px sans-serif';
    ctx.fillStyle = css.getPropertyValue('--color-text').trim() || '#333';
    ctx.fillText('Coverage (last points)', 10, 16);
}

/**
 * Refresh streaming coverage arrays and summary; update chart and text.
 */
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
        coverageState.covNeg = Array.isArray(data.coverage_neg) ? data.coverage_neg : [];
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
                const latestNeg = Array.isArray(coverageState.covNeg) && coverageState.covNeg.length ? coverageState.covNeg[coverageState.covNeg.length - 1] : null;
                const targetPct = ((1 - (summary.alpha ?? 0.05)) * 100).toFixed(1);
                const fraudPct = latestPos != null && Number.isFinite(latestPos) ? (latestPos * 100).toFixed(1) : '-';
                const nonFraudPct = latestNeg != null && Number.isFinite(latestNeg) ? (latestNeg * 100).toFixed(1) : '-';
                const actualPct = latest != null ? (latest * 100).toFixed(1) : '-';
                const avgSet = summary.avg_set_size != null ? Number(summary.avg_set_size).toFixed(2) : '-';
                const amb = summary.ambiguous_share != null ? (Number(summary.ambiguous_share) * 100).toFixed(1) + '%' : '-';
                statsEl.textContent = `Fraud @ target ${targetPct}%: ${fraudPct}% · Non-fraud: ${nonFraudPct}% · Overall: ${actualPct}% · Avg set size: ${avgSet} · Ambiguous: ${amb}`;
            }
        }
    } catch (e) {
        console.error('Failed to refresh streaming coverage', e);
    }
}

/**
 * Draw streaming coverage lines (overall and optionally fraud-only)
 * with target band overlay when in range.
 */
function drawCoverageChart() {
    const canvas = document.getElementById('coverageChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const css = getComputedStyle(document.documentElement);
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
    ctx.strokeStyle = css.getPropertyValue('--color-axis').trim() || '#ddd';
    ctx.beginPath();
    ctx.moveTo(40, 10);
    ctx.lineTo(40, h - 20);
    ctx.lineTo(w - 10, h - 20);
    ctx.stroke();
    ctx.font = css.getPropertyValue('--font-small').trim() || '12px sans-serif';
    ctx.fillStyle = css.getPropertyValue('--color-axis-text').trim() || '#555';
    ctx.fillText(maxY.toFixed(2), 5, 12);
    ctx.fillText(minY.toFixed(2), 5, h - 20);

    // line
    ctx.strokeStyle = css.getPropertyValue('--color-line-coverage').trim() || '#0a8';
    ctx.beginPath();
    for (let i = 0; i < vals.length; i++) {
        const x = 40 + (i / (vals.length - 1)) * (w - 50);
        const y = (h - 20) - ((vals[i] - minY) / (maxY - minY)) * (h - 30);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // fraud-class coverage line
    if (valsPos.length > 1) {
        ctx.strokeStyle = css.getPropertyValue('--color-line-fraud').trim() || '#6a0dad'; // purple
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
        const dash = (css.getPropertyValue('--dash-target').trim() || '5,5').split(',').map(x => Number(x.trim()) || 0);
        if (ctx.setLineDash) ctx.setLineDash(dash);
        ctx.strokeStyle = css.getPropertyValue('--color-line-target').trim() || '#e33';
        ctx.beginPath();
        ctx.moveTo(40, yTarget);
        ctx.lineTo(w - 10, yTarget);
        ctx.stroke();
        if (ctx.setLineDash) ctx.setLineDash([]);
    }
}

// --- Offline metrics & images ---
/**
 * Refresh offline metrics table, CV baselines, and figures. Idempotent.
 */
async function refreshOffline() {
    const container = document.getElementById('offlineMetrics');
    const imgContainer = document.getElementById('offlineImages');
    const cvContainer = document.getElementById('baselinesCV');
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
        container.innerHTML = html + '<div class="figure-caption">Metrics on held-out test set (no calibration data).</div>';
        imgContainer.innerHTML = '';
        // Baselines CV table
        if (cvContainer) {
            const rows = Array.isArray(data.baselines_cv) ? data.baselines_cv.slice() : [];
            if (!rows.length) {
                cvContainer.innerHTML = '<div class="muted">No CV baselines available yet.</div>';
            } else {
                // Sort by PR-AUC mean descending
                rows.sort((a, b) => (b.pr_auc_mean ?? 0) - (a.pr_auc_mean ?? 0));
                const fmt = (x, d = 4) => (Number.isFinite(x) ? Number(x).toFixed(d) : '-');
                const fmtParams = (p) => {
                    if (!p || typeof p !== 'object') return '';
                    return Object.entries(p).map(([k, v]) => `${k}=${Array.isArray(v) ? JSON.stringify(v) : v}`).join(' ');
                };
                let t = '<table><thead><tr><th>Model</th><th>Params</th><th>ROC-AUC (CV)</th><th>PR-AUC (CV)</th><th>Chosen</th></tr></thead><tbody>';
                for (const r of rows) {
                    const roc = `${fmt(r.roc_auc_mean)}±${fmt(r.roc_auc_std, 3)}`;
                    const pr = `${fmt(r.pr_auc_mean)}±${fmt(r.pr_auc_std, 3)}`;
                    const chosen = r.selected ? '✓' : '';
                    t += `<tr><td>${r.label || r.family}</td><td>${fmtParams(r.params)}</td><td>${roc}</td><td>${pr}</td><td style="text-align:center">${chosen}</td></tr>`;
                }
                t += '</tbody></table>';
                cvContainer.innerHTML = t;
            }
        }
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
                const note = it.key === 'reliability_post'
                    ? 'Class prior ≪ 1% → calibrated scores mostly in [0, 0.3].'
                    : (descriptions[it.key] || 'Figure.');
                p.textContent = note;
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
        // Update stream image
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

/**
 * Read ablation CSV/JSON and render a concise table; show trigger if absent.
 */
async function refreshAblation() {
    const container = document.getElementById('ablationTable');
    if (!container) return;
    try {
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
        // Render concise table with target and fraud coverage when available
        let html = '<table><thead><tr><th>Mode</th><th>Param</th><th>Value</th><th>Target</th><th>Coverage</th><th>Fraud cov</th><th>Ambig%</th><th>Violation (target gap)</th><th>n_eff</th></tr></thead><tbody>';
        for (const r of rows) {
            const cov = r.final_coverage != null ? Number(r.final_coverage).toFixed(4) : '-';
            const covPos = r.final_coverage_pos != null ? Number(r.final_coverage_pos).toFixed(4) : '-';
            const target = r.target != null ? Number(r.target).toFixed(2) : (r.alpha != null ? (1 - Number(r.alpha)).toFixed(2) : '-');
            const viol = r.final_violation_rate != null ? Number(r.final_violation_rate).toFixed(4) : '-';
            const neff = r.effective_n != null ? Number(r.effective_n).toFixed(1) : '-';
            const amb = r.ambiguous_share != null ? (Number(r.ambiguous_share) * 100).toFixed(1) + '%' : '-';
            html += `<tr><td>${r.mode}</td><td>${r.param}</td><td>${r.value}</td><td>${target}</td><td>${cov}</td><td>${covPos}</td><td>${amb}</td><td>${viol}</td><td>${neff}</td></tr>`;
        }
        html += '</tbody></table>';
        container.innerHTML = html;
    } catch (e) {
        console.error('Failed to refresh ablation', e);
    }
}

/**
 * POST to start ablation and poll for table materialization with a timeout.
 */
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

/**
 * Fetch /health and pretty-print into the Health tab.
 */
async function refreshHealth() {
    try {
        const data = await fetchJSON('/health');
        const pre = document.getElementById('healthJson');
        if (pre) pre.textContent = JSON.stringify(data, null, 2);
    } catch (e) {
        console.error('Failed to fetch health', e);
    }
}

/**
 * Load schema and example predict payload into the textarea.
 */
async function loadPredictExample() {
    try {
        const data = await fetchJSON('/predict/schema');
        const ta = document.getElementById('predictPayload');
        if (ta) ta.value = JSON.stringify(data.example, null, 2);
    } catch (e) {
        console.error('Failed to load example schema', e);
    }
}

/**
 * Send predict request with textarea JSON; print raw text or JSON response.
 */
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

