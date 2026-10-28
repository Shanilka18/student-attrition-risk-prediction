"""Generate test fixtures for upload robustness testing."""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

FIXTURES_DIR = Path("tests/fixtures")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

# 1. Clean sample
src_clean = Path("data/samples/sample_upload_clean.xlsx")
if src_clean.exists():
    shutil.copy(src_clean, FIXTURES_DIR / "clean_sample.xlsx")
    df_clean = pd.read_excel(src_clean)
    df_clean.to_csv(FIXTURES_DIR / "clean_sample.csv", index=False)
    print("Created clean_sample.xlsx and clean_sample.csv")
else:
    # Generate clean
    cols = ["student_id"]
    for ind in ["resource_access", "study_time", "review_time", "study_consistency", "participation", "assignment_engagement", "quiz_practice"]:
        for p in range(1, 5):
            cols.append(f"{ind}_p{p}")
    data = {"student_id": [f"STU_{i+1:04d}" for i in range(200)]}
    for c in cols[1:]:
        data[c] = np.random.randint(1, 6, size=200)
    df_clean = pd.DataFrame(data)
    df_clean.to_csv(FIXTURES_DIR / "clean_sample.csv", index=False)
    df_clean.to_excel(FIXTURES_DIR / "clean_sample.xlsx", index=False)
    print("Generated clean_sample.csv and clean_sample.xlsx")

# 2. Messy sample
src_messy = Path("data/samples/sample_upload_messy.csv")
if src_messy.exists():
    shutil.copy(src_messy, FIXTURES_DIR / "messy_sample.csv")
    print("Copied messy_sample.csv")
else:
    df_messy = df_clean.copy()
    rename_map = {c: c.replace("_", " ").title() for c in df_messy.columns}
    rename_map["student_id"] = "Student ID Number"
    df_messy = df_messy.rename(columns=rename_map)
    df_messy["Campus Branch"] = "Colombo Metropolitan Campus"
    df_messy.to_csv(FIXTURES_DIR / "messy_sample.csv", index=False)
    print("Generated messy_sample.csv")

# 3. Extra and missing columns
df_extra_missing = df_clean.head(50).copy()
# Drop 5 required columns
cols_to_drop = [c for c in df_extra_missing.columns if "quiz_practice" in c] + ["participation_p4"]
df_extra_missing = df_extra_missing.drop(columns=[c for c in cols_to_drop if c in df_extra_missing.columns])
# Add 3 extra columns
df_extra_missing["faculty_advisor"] = "Dr. K. Perera"
df_extra_missing["financial_aid_status"] = "Full Scholarship"
df_extra_missing["high_school_gpa"] = np.round(np.random.uniform(2.5, 4.0, size=50), 2)
df_extra_missing.to_csv(FIXTURES_DIR / "extra_and_missing.csv", index=False)
print("Generated extra_and_missing.csv")

# 4. Large 20,000 rows sample
n_large = 20000
large_data = {"student_id": [f"BATCH_STU_{i+1:06d}" for i in range(n_large)]}
for col in df_clean.columns:
    if col != "student_id":
        large_data[col] = np.random.randint(1, 6, size=n_large)
df_large = pd.DataFrame(large_data)
df_large.to_csv(FIXTURES_DIR / "large_20k_rows.csv", index=False)
print(f"Generated large_20k_rows.csv ({FIXTURES_DIR / 'large_20k_rows.csv'} -> {n_large} rows)")

# 5. Unicode / Sinhala in headers and extra columns
df_sinhala = df_clean.head(35).copy()
df_sinhala["විශේෂ_සටහන්"] = [f"ශිෂ්‍ය විමසීම {i+1} - පැමිණීම සතුටුදායකයි" for i in range(35)]
df_sinhala["කණ්ඩායම"] = "පරිගණක විද්‍යා පීඨය"
# Rename some headers to mixed Sinhala/English
sinhala_rename = {
    "student_id": "ශිෂ්‍ය_අංකය (Student ID)",
    "resource_access_p1": "සම්පත් භාවිතය P1 (Resource Access)",
    "study_time_p1": "ස්වයං අධ්‍යයන කාලය P1 (Study Hours)",
}
df_sinhala = df_sinhala.rename(columns=sinhala_rename)
df_sinhala.to_csv(FIXTURES_DIR / "unicode_sinhala.csv", index=False, encoding="utf-8-sig")
print("Generated unicode_sinhala.csv")

# 6. Google Forms Export with long headers and text answers
text_map = {
    1: "Never / Rarely",
    2: "1 to 2 times a week",
    3: "3 to 4 times a week",
    4: "5 to 6 times a week",
    5: "Every single day",
}
forms_data = {
    "Timestamp": ["2026/10/15 10:24:12 AM GMT+5:30"] * 40,
    "Student Identification Number": [f"GF_STU_{i+1:03d}" for i in range(40)],
}
for ind in ["resource_access", "study_time", "review_time", "study_consistency", "participation", "assignment_engagement", "quiz_practice"]:
    for p in range(1, 5):
        q_label = f"How would you rate your {ind.replace('_', ' ')} during period {p} (Weeks {(p-1)*4+1} to {p*4})?"
        raw_vals = np.random.randint(1, 6, size=40)
        forms_data[q_label] = [text_map.get(v, str(v)) for v in raw_vals]

df_forms = pd.DataFrame(forms_data)
df_forms.to_csv(FIXTURES_DIR / "google_forms_export.csv", index=False)
print("Generated google_forms_export.csv")

# 7. Empty file
empty_path = FIXTURES_DIR / "empty_file.csv"
empty_path.write_text("", encoding="utf-8")
print("Generated empty_file.csv (0 bytes)")

# 8. Wrong file type (PDF header / binary bytes)
wrong_path = FIXTURES_DIR / "wrong_file_type.pdf"
wrong_path.write_bytes(b"%PDF-1.4\n%Invalid survey file format\n")
print("Generated wrong_file_type.pdf")

print("ALL 8 ROBUSTNESS FIXTURES CREATED SUCCESSFULLY!")
