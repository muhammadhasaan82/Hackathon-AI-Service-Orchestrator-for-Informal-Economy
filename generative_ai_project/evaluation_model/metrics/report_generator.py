"""
Report Generator — Produces JSON + HTML evaluation reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("eval.report")


def _score_badge(score: float) -> str:
    if score >= 0.85:
        return f'<span style="color:#22c55e;font-weight:700">{score:.3f} ✅</span>'
    elif score >= 0.65:
        return f'<span style="color:#f59e0b;font-weight:700">{score:.3f} ⚠️</span>'
    else:
        return f'<span style="color:#ef4444;font-weight:700">{score:.3f} ❌</span>'


def generate_report(results: dict, output_path: str):
    """Generate an HTML evaluation report."""
    ts = results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    suite = results.get("suite", "all")
    agg = results.get("aggregate_score", 0.0)
    duration = results.get("duration_seconds", 0.0)
    failures = results.get("failures", [])
    scores = results.get("scores", {})

    def row(label, value, is_score=True):
        cell = _score_badge(value) if is_score and isinstance(value, float) else f"<td>{value}</td>"
        return f"<tr><td>{label}</td><td>{cell}</td></tr>" if not is_score else f"<tr><td>{label}</td><td>{_score_badge(value)}</td></tr>"

    metrics_rows = ""
    for suite_name, suite_results in scores.items():
        if not isinstance(suite_results, dict):
            continue
        metrics_rows += f"<tr><td colspan='2' style='background:#1e293b;color:#94a3b8;font-weight:600;padding:8px 12px'>{suite_name.upper()}</td></tr>"
        for k, v in suite_results.items():
            if k == "per_case" or not isinstance(v, (int, float)):
                continue
            metrics_rows += f"<tr><td style='padding:6px 12px'>{k.replace('_',' ').title()}</td><td style='padding:6px 12px'>{_score_badge(float(v)) if isinstance(v, float) and 0 <= v <= 1 else f'<span>{v}</span>'}</td></tr>"

    failures_html = ""
    if failures:
        failures_html = "<h3 style='color:#ef4444'>Failures</h3><ul>"
        for f in failures:
            failures_html += f"<li><b>{f.get('suite')}</b>: {f.get('error')}</li>"
        failures_html += "</ul>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report — {ts}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 32px; }}
  h1 {{ font-size: 1.8rem; color: #7c3aed; margin-bottom: 8px; }}
  h2 {{ font-size: 1.2rem; color: #94a3b8; margin: 24px 0 12px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; border: 1px solid #334155; }}
  .aggregate {{ font-size: 3.5rem; font-weight: 800; text-align: center; color: {'#22c55e' if agg >= 0.85 else '#f59e0b' if agg >= 0.65 else '#ef4444'}; }}
  .meta {{ color: #64748b; font-size: 0.85rem; margin-top: 4px; text-align: center; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #334155; font-size: 0.9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge-pass {{ background: #14532d; color: #22c55e; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }}
  .badge-warn {{ background: #78350f; color: #f59e0b; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }}
  .badge-fail {{ background: #7f1d1d; color: #ef4444; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }}
</style>
</head>
<body>
<h1>🤖 AI Service Orchestrator — Evaluation Report</h1>
<p class="meta">Suite: <b>{suite}</b> | Timestamp: {ts} | Duration: {duration:.1f}s | Failures: {len(failures)}</p>

<div class="card" style="margin-top:24px">
  <div class="aggregate">{agg:.3f}</div>
  <p class="meta">Aggregate Score</p>
</div>

<div class="card">
  <h2>📊 Metrics Breakdown</h2>
  <table>{metrics_rows}</table>
</div>

{f'<div class="card">{failures_html}</div>' if failures else ''}

<p class="meta" style="margin-top:24px">Generated: {datetime.now().isoformat()} | AI Service Orchestrator v2.0</p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML report written: {output_path}")
