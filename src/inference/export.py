"""Multi-Format Export Engine for Inference Deliverables.

Exports:
- Formatted CSV
- Formatted Excel (.xlsx) with styled headers and color-coded risk bands
- Fusion Framework JSON array payload
- Standalone HTML / PDF printable single-page cohort report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.utils.logging_config import setup_logger
from src.utils.paths import ensure_dir

logger = setup_logger("inference_export")


def export_inference_bundle(
    df_scored: pd.DataFrame,
    cohort_summary: Dict[str, Any],
    output_dir: Path | str = "artifacts/predictions",
) -> Dict[str, Path]:
    """Export complete bundle of inference deliverables."""
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    # 1. Formatted CSV
    csv_path = out_dir / "scored_students.csv"
    cols_to_save = [
        "participant_id", "eri_score", "risk_band", "confidence_note",
        "missing_fields_count", "narrative",
    ]
    df_scored[cols_to_save].to_csv(csv_path, index=False)

    # 2. Formatted Excel with OpenPyXL
    xlsx_path = out_dir / "scored_students.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_scored[cols_to_save].to_excel(writer, sheet_name="ERI_Predictions", index=False)
        # Summary Sheet
        summary_df = pd.DataFrame([
            {"Metric": "Model Version", "Value": cohort_summary.get("model_version")},
            {"Metric": "Total Students Scored", "Value": cohort_summary.get("total_students_scored")},
            {"Metric": "Mean ERI Score", "Value": cohort_summary.get("mean_eri")},
            {"Metric": "Median ERI Score", "Value": cohort_summary.get("median_eri")},
            {"Metric": "Causal Summary Note", "Value": cohort_summary.get("causal_summary")},
        ])
        summary_df.to_excel(writer, sheet_name="Cohort_Summary", index=False)

    # 3. Fusion Framework JSON
    fusion_path = out_dir / "fusion_payload.json"
    fusion_records = [
        {
            "participant_id": row["participant_id"],
            "eri": row["eri_score"],
            "band": row["risk_band"],
            "explanation_type": "model_explanation",
            "model_version": cohort_summary.get("model_version"),
            "top_positive_drivers": row["top_positive_drivers"],
            "top_protective_factors": row["top_protective_factors"],
            "confidence_note": row["confidence_note"],
        }
        for row in df_scored.to_dict(orient="records")
    ]
    with open(fusion_path, "w", encoding="utf-8") as f:
        json.dump(fusion_records, f, indent=2)

    # 4. Cohort Summary JSON
    summary_path = out_dir / "cohort_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(cohort_summary, f, indent=2)

    # 5. One-Page HTML Report
    html_path = out_dir / "cohort_report.html"
    _generate_html_report(df_scored, cohort_summary, html_path)

    logger.info(f"Exported inference deliverables bundle to: {out_dir}")
    return {
        "csv": csv_path,
        "xlsx": xlsx_path,
        "fusion_json": fusion_path,
        "summary_json": summary_path,
        "html_report": html_path,
    }


def _generate_html_report(df_scored: pd.DataFrame, cohort_summary: Dict[str, Any], output_path: Path) -> None:
    """Generate a clean, single-page printable HTML cohort report."""
    risk_dist = cohort_summary.get("risk_band_distribution_pct", {})
    top_drivers = cohort_summary.get("top_cohort_drivers", [])

    rows_html = ""
    for _, r in df_scored.head(15).iterrows():
        badge_color = {
            "Low": "#27ae60",
            "Moderate": "#f39c12",
            "High": "#d35400",
            "Critical": "#c0392b",
        }.get(r["risk_band"], "#7f8c8d")

        rows_html += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; font-family: monospace;">{r['participant_id']}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; font-weight: bold;">{r['eri_score']}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">
                <span style="background-color: {badge_color}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px;">
                    {r['risk_band']}
                </span>
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; font-size: 12px;">{r['narrative']}</td>
        </tr>
        """

    drivers_html = "".join(
        f"<li><b>{d['feature']}</b> (Impact: {d['mean_abs_shap']})</li>"
        for d in top_drivers
    )

    dist_html = "".join(
        f"<div style='margin-right: 20px;'><b>{band}:</b> {pct}%</div>"
        for band, pct in risk_dist.items()
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ERI Cohort Risk Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; color: #2c3e50; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
        .metric-box {{ display: inline-block; background: #ecf0f1; padding: 15px 25px; border-radius: 6px; margin-right: 15px; }}
        .metric-val {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background-color: #f8f9fa; padding: 10px; border-bottom: 2px solid #bdc3c7; text-align: left; }}
    </style>
</head>
<body>
    <h1>Environmental Risk Intelligence (ERI) — Cohort Assessment</h1>
    <p><i>Generated by SLIIT CDAP Project KND-CDAP-IT-05 Production Inference Engine ({cohort_summary.get('model_version')})</i></p>

    <div style="margin: 20px 0;">
        <div class="metric-box">
            <div>Total Students</div>
            <div class="metric-val">{cohort_summary.get('total_students_scored')}</div>
        </div>
        <div class="metric-box">
            <div>Mean ERI Score</div>
            <div class="metric-val">{cohort_summary.get('mean_eri')} / 100</div>
        </div>
        <div class="metric-box">
            <div>Median ERI Score</div>
            <div class="metric-val">{cohort_summary.get('median_eri')} / 100</div>
        </div>
    </div>

    <h3>Cohort Risk Distribution</h3>
    <div style="display: flex; margin-bottom: 20px;">
        {dist_html}
    </div>

    <h3>Primary Cohort Risk Drivers</h3>
    <ul>
        {drivers_html}
    </ul>

    <h3>Sample Scored Students (First 15)</h3>
    <table>
        <thead>
            <tr>
                <th>Participant ID</th>
                <th>ERI (0-100)</th>
                <th>Risk Band</th>
                <th>Plain-Language Narrative</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <div style="margin-top: 30px; padding: 12px; background: #fdf2e9; border-left: 4px solid #e67e22; font-size: 12px;">
        <b>Causal Inference Note:</b> {cohort_summary.get('causal_summary')}
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
