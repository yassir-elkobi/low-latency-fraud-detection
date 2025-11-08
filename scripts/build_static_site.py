from __future__ import annotations

import json
import shutil
from pathlib import Path


def main() -> None:
    repo = Path('.')
    models_dir = repo / 'models'
    artifacts_dir = repo / 'artifacts'
    site_dir = repo / 'site'
    assets_dir = site_dir / 'assets'

    site_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    metrics_path = models_dir / 'metrics_offline.json'
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    base = metrics.get('base_model', {})
    pre = metrics.get('metrics_pre', {})
    post = metrics.get('metrics_post', {})
    calib = metrics.get('calibration', {})

    # Copy images if present
    rel_pre = artifacts_dir / 'reliability_pre.png'
    rel_post = artifacts_dir / 'reliability_post.png'
    stream_cov = artifacts_dir / 'stream_coverage.png'
    copied = {}
    for src in [rel_pre, rel_post, stream_cov]:
        if src.exists():
            dst = assets_dir / src.name
            shutil.copy2(src, dst)
            copied[src.name] = f'assets/{src.name}'

    # Build HTML from template
    template_path = repo / 'scripts' / 'static_site_template.html'
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found at {template_path}")

    rel_pre_html = f'<img src="{copied.get("reliability_pre.png")}" alt="reliability pre" />' if 'reliability_pre.png' in copied else '<div class="muted">Missing</div>'
    rel_post_html = f'<img src="{copied.get("reliability_post.png")}" alt="reliability post" />' if 'reliability_post.png' in copied else '<div class="muted">Missing</div>'
    stream_cov_html = f'<img src="{copied.get("stream_coverage.png")}" alt="stream coverage" />' if 'stream_coverage.png' in copied else '<div class="muted">Run streaming to generate</div>'

    html = template_path.read_text().format(
        base_label=base.get('label', '?'),
        calib_method=calib.get('calibration_method', '?'),
        roc_auc_pre=pre.get('roc_auc', '-'),
        roc_auc_post=post.get('roc_auc', '-'),
        pr_auc_pre=pre.get('pr_auc', '-'),
        pr_auc_post=post.get('pr_auc', '-'),
        brier_pre=pre.get('brier', '-'),
        brier_post=post.get('brier', '-'),
        nll_pre=pre.get('nll', '-'),
        nll_post=post.get('nll', '-'),
        reliability_pre=rel_pre_html,
        reliability_post=rel_post_html,
        stream_coverage=stream_cov_html,
    )

    (site_dir / 'index.html').write_text(html)
    print("Static site built at:", site_dir.resolve())


if __name__ == '__main__':
    main()
