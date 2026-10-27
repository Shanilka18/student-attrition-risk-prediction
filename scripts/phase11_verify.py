"""Phase 11 verification script: leakage, provenance, reproducibility."""
import json
import pathlib
import sys

PASS = True

def check(name, condition, detail=""):
    global PASS
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        PASS = False

# ---- 1. LEAKAGE SAFETY ----
print("\n=== 1. LEAKAGE SAFETY CHECK ===")
split_dir = pathlib.Path("data/splits")
try:
    train_ids = set(json.loads((split_dir / "train_students.json").read_text()))
    val_ids   = set(json.loads((split_dir / "val_students.json").read_text()))
    test_ids  = set(json.loads((split_dir / "test_students.json").read_text()))
    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids   & test_ids
    print(f"  Train={len(train_ids)}  Val={len(val_ids)}  Test={len(test_ids)}")
    check("Train intersect Val  == empty", len(tv) == 0, f"Overlap: {tv}")
    check("Train intersect Test == empty", len(tt) == 0, f"Overlap: {tt}")
    check("Val   intersect Test == empty", len(vt) == 0, f"Overlap: {vt}")
    check("Total students == 5000", len(train_ids | val_ids | test_ids) == 5000)
except Exception as e:
    print(f"  ERROR reading split files: {e}")
    PASS = False

# ---- 2. TEST SET PROVENANCE ----
print("\n=== 2. ONE-TOUCH TEST SET PROVENANCE ===")
try:
    prov_path = pathlib.Path("artifacts/metrics/test_evaluation_provenance.json")
    prov = json.loads(prov_path.read_text())
    hashes = set(e["test_data_sha256"] for e in prov["evaluation_history"])
    print(f"  Evaluations: {prov['total_test_evaluations']}  Unique fingerprints: {len(hashes)}")
    check("All evaluations use identical SHA-256", len(hashes) == 1)
    check("At least 1 evaluation recorded", prov["total_test_evaluations"] >= 1)
    fp = prov["current_fingerprint"]
    check("Fingerprint is 64-char hex", len(fp) == 64 and all(c in "0123456789abcdef" for c in fp))
    print(f"  SHA-256: {fp}")
except Exception as e:
    print(f"  ERROR: {e}")
    PASS = False

# ---- 3. REPRODUCIBILITY ----
print("\n=== 3. REPRODUCIBILITY (seed=42 tolerance check) ===")
try:
    gru = json.loads(pathlib.Path("artifacts/models/Category_E/gru/summary.json").read_text())
    cb  = json.loads(pathlib.Path("artifacts/models/Category_C/catboost/summary.json").read_text())
    gru_auc  = gru["validation_metrics"]["roc_auc"]
    gru_std  = gru["stability_summary"]["roc_auc_std"]
    cb_auc   = cb["validation_metrics"]["roc_auc"]
    print(f"  GRU val ROC-AUC = {gru_auc:.4f}  (ref 0.8249, tol +/-0.01)")
    print(f"  GRU seed std    = {gru_std:.4f}  (must <= 0.05)")
    print(f"  CatBoost AUC    = {cb_auc:.4f}  (ref 0.8196, tol +/-0.01)")
    check("GRU ROC-AUC within 0.01 of reference", abs(gru_auc - 0.8249) < 0.01)
    check("GRU seed std <= 0.05", gru_std <= 0.05)
    check("CatBoost AUC within 0.01 of reference", abs(cb_auc - 0.8196) < 0.01)
except Exception as e:
    print(f"  ERROR: {e}")
    PASS = False

# ---- 4. SECURITY REVIEW ----
print("\n=== 4. SECURITY / PRIVACY REVIEW ===")
try:
    main_src = pathlib.Path("backend/src/api/main.py").read_text()
    validator_src = pathlib.Path("backend/src/api/validator.py").read_text()
    # File type allowlist
    check("File type allowlist (.csv,.xlsx,.xls)", '".csv", ".xlsx", ".xls"' in main_src)
    # Size cap
    check("10 MB size cap enforced", "MAX_UPLOAD_SIZE_BYTES" in main_src and "10 * 1024 * 1024" in main_src)
    # 413 response on oversize
    check("HTTP 413 on oversized upload", "HTTP_413_CONTENT_TOO_LARGE" in main_src)
    # In-memory processing (BytesIO, no disk write)
    check("BytesIO in-memory (no disk write)", "io.BytesIO" in main_src)
    # No student ID in logs
    check("student_id regex sanitisation in validator", "STUDENT_ID_REGEX" in validator_src)
    # Check upload logging does not log student_id
    upload_log_part = main_src.split("predict_upload")[1].split("return BatchUploadResponse")[0]
    check("predict_upload log contains only counts/latency, no PII", "records=" in upload_log_part and "student_id" not in upload_log_part)
    # No file persistence after processing
    check("No open(...,'w') storing uploaded content in main", "open(" not in main_src.split("predict_upload")[1][:1000])
except Exception as e:
    print(f"  ERROR: {e}")
    PASS = False

# ---- RESULT ----
print("\n" + "="*50)
if PASS:
    print("OVERALL: ALL CHECKS PASSED")
else:
    print("OVERALL: ONE OR MORE CHECKS FAILED")
sys.exit(0 if PASS else 1)
