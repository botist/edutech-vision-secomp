from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("generate_session_report", ROOT_DIR / "scripts" / "generate_session_report.py")
assert SPEC is not None
generate_session_report = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generate_session_report)


def test_synthetic_report_is_labeled_and_generates_charts(tmp_path) -> None:
    generate_session_report.create_synthetic_demo_results(tmp_path)
    charts = generate_session_report.generate_charts(tmp_path, tmp_path / "charts", synthetic=True)
    report = generate_session_report.build_report(tmp_path, synthetic=True, chart_paths=charts, output_dir=tmp_path)

    assert "SIMULACAO" in report
    assert "gere novamente usando os csvs reais" in report.lower()
    assert all(path.exists() for path in charts.values())
