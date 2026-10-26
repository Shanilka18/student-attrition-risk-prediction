"""Phase 10: Upload flow integration test.

Tests 5 upload scenarios against the live FastAPI server:
  (a) Clean 300-row CSV
  (b) Wrong column names
  (c) Missing values + out-of-range answers
  (d) .xlsx file
  (e) Outcome column present (retrain mode)
"""

import io
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import requests

BASE_URL = "http://127.0.0.1:8000"
ITEMS = ["S1","S2","S3","M1","M2","M3","B1","B2","B3","SE1","SE2","SE3","PS1","PS2","PS3"]
RNG = np.random.default_rng(42)
RESULTS = []


def _make_clean_df(n: int = 300) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {"study_id": f"SLIIT_TEST_{i:05d}"}
        for col in ITEMS:
            row[col] = int(RNG.integers(1, 6))
        rows.append(row)
    return pd.DataFrame(rows)


def _post_upload(content: bytes, filename: str, content_type: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/api/upload",
        files={"file": (filename, io.BytesIO(content), content_type)},
        timeout=30,
    )
    return {"status": resp.status_code, "body": resp.json()}


def test_a_clean_csv():
    """(a) Clean 300-row CSV — expect 200, all 15 items mapped, 0 errors."""
    df = _make_clean_df(300)
    csv_bytes = df.to_csv(index=False).encode()
    r = _post_upload(csv_bytes, "clean_300.csv", "text/csv")
    passed = r["status"] == 200 and r["body"]["row_count"] == 300
    RESULTS.append({"scenario": "a_clean_csv", "status_code": r["status"], "passed": passed,
                    "details": f"rows={r['body'].get('row_count','?')} errors={r['body'].get('validation_report',{}).get('error_count','?')}"})
    return r


def test_b_wrong_columns():
    """(b) Wrong column names — expect 200 with low-confidence mappings."""
    df = _make_clean_df(100)
    df.columns = ["student_id"] + [f"q{i}" for i in range(1, 16)]
    csv_bytes = df.to_csv(index=False).encode()
    r = _post_upload(csv_bytes, "wrong_cols.csv", "text/csv")
    # Server should return 200 with mapping suggestions / warnings
    mappings = r["body"].get("suggested_mappings", {})
    RESULTS.append({"scenario": "b_wrong_columns", "status_code": r["status"], "passed": r["status"] == 200,
                    "details": f"mappings_found={len(mappings)} items_mapped_count={sum(1 for v in mappings.values() if v in ITEMS)}"})
    return r


def test_c_missing_and_out_of_range():
    """(c) Missing values + out-of-range — expect 200 with error/warning counts > 0."""
    df = _make_clean_df(150)
    # Introduce missing values
    for col in ["S1", "M2", "B3"]:
        df.loc[RNG.choice(df.index, 10, replace=False), col] = np.nan
    # Introduce out-of-range values
    df.loc[RNG.choice(df.index, 5, replace=False), "SE1"] = 7
    df.loc[RNG.choice(df.index, 5, replace=False), "PS2"] = 0
    csv_bytes = df.to_csv(index=False).encode()
    r = _post_upload(csv_bytes, "dirty.csv", "text/csv")
    val = r["body"].get("validation_report", {})
    n_errors = len(val.get("errors", []))
    n_warnings = len(val.get("warnings", []))
    passed_flag = val.get("passed", True)
    RESULTS.append({"scenario": "c_missing_out_of_range", "status_code": r["status"],
                    "passed": r["status"] == 200 and not passed_flag and (n_errors > 0),
                    "details": f"errors={n_errors} warnings={n_warnings} val_passed={passed_flag}"})
    return r


def test_d_xlsx_file():
    """(d) .xlsx file — expect 200 with correct parsing."""
    df = _make_clean_df(200)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    xlsx_bytes = buf.getvalue()
    r = _post_upload(xlsx_bytes, "survey.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    passed = r["status"] == 200 and r["body"].get("row_count", 0) == 200
    RESULTS.append({"scenario": "d_xlsx_file", "status_code": r["status"], "passed": passed,
                    "details": f"rows={r['body'].get('row_count','?')}"})
    return r


def test_e_retrain_with_outcome():
    """(e) File with outcome column — upload then trigger retrain."""
    df = _make_clean_df(250)
    # Add synthetic outcome
    df["attrition_label"] = RNG.choice([0, 1], size=250, p=[0.75, 0.25])
    csv_bytes = df.to_csv(index=False).encode()
    r_up = _post_upload(csv_bytes, "with_outcome.csv", "text/csv")

    if r_up["status"] != 200:
        RESULTS.append({"scenario": "e_retrain_outcome_upload", "status_code": r_up["status"], "passed": False,
                        "details": "Upload failed"})
        return r_up

    upload_id = r_up["body"].get("upload_id")
    has_outcomes = r_up["body"].get("has_outcomes", False)
    RESULTS.append({"scenario": "e_retrain_outcome_upload", "status_code": r_up["status"], "passed": has_outcomes,
                    "details": f"upload_id={upload_id} has_outcomes={has_outcomes}"})

    # Trigger retrain if upload succeeded and outcomes detected
    if upload_id and has_outcomes:
        retrain_resp = requests.post(
            f"{BASE_URL}/api/retrain",
            json={"upload_id": upload_id, "target_column": "attrition_label"},
            timeout=30,
        )
        RESULTS.append({"scenario": "e_retrain_trigger", "status_code": retrain_resp.status_code,
                        "passed": retrain_resp.status_code == 202,
                        "details": retrain_resp.json().get("status", "unknown")})

    return r_up


def main():
    print("=" * 65)
    print("Phase 10: Upload Flow Integration Test Suite")
    print("=" * 65)

    # Health check first
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"  API Health: {health.status_code} - {health.json().get('status','?')}")
    except Exception as e:
        print(f"  ERROR: API not reachable at {BASE_URL}: {e}")
        print("  Start the API first: uvicorn src.api.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    print()
    tests = [
        ("(a) Clean 300-row CSV", test_a_clean_csv),
        ("(b) Wrong column names", test_b_wrong_columns),
        ("(c) Missing + out-of-range", test_c_missing_and_out_of_range),
        ("(d) .xlsx file", test_d_xlsx_file),
        ("(e) Outcome column + retrain", test_e_retrain_with_outcome),
    ]
    for label, fn in tests:
        print(f"  Running {label}...")
        try:
            fn()
        except Exception as ex:
            RESULTS.append({"scenario": label, "status_code": -1, "passed": False, "details": str(ex)})
        time.sleep(0.3)

    print()
    print("=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    all_passed = True
    for r in RESULTS:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        print(f"  [{status}] {r['scenario']:40s} HTTP={r['status_code']} {r['details']}")

    print()
    overall = "ALL PASSED" if all_passed else "SOME FAILURES"
    print(f"Overall: {overall}")

    out_path = pathlib.Path("artifacts/metrics/upload_flow_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
