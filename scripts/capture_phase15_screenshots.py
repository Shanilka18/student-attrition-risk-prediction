"""Capture screenshots of the Phase 15 student scoring workflow (/score).
Uses Chrome headless against the real FastAPI uvicorn server.
"""

import os
import sys
import time
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
import urllib.request
import httpx

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_PATH):
    CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PORT = 8008


def wait_for_server(url=f"http://127.0.0.1:{PORT}/health", timeout_s=90):
    t0 = time.time()
    last_print = t0
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            if time.time() - last_print >= 5.0:
                print(
                    f"  Waiting for server on {url} ({int(time.time() - t0)}s elapsed)...",
                    flush=True,
                )
                last_print = time.time()
            time.sleep(0.5)
    return False


def main():
    print(f"[SCREENSHOT] Launching server on http://127.0.0.1:{PORT}...", flush=True)
    server_log = open("uvicorn_score_capture.log", "w", encoding="utf-8")
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mei.api.app:create_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--factory",
        ],
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    try:
        if not wait_for_server():
            print("[ERROR] Server failed to start within timeout.")
            server_process.kill()
            server_log.close()
            with open("uvicorn_score_capture.log", "r", encoding="utf-8") as f:
                print(f.read())
            sys.exit(1)

        print("[SCREENSHOT] Server healthy.", flush=True)

        # 1. Upload clean sample to create real validation session
        print("[SCREENSHOT] Uploading sample_upload_clean.xlsx...", flush=True)
        sample_path = Path("data/samples/sample_upload_clean.xlsx")
        with open(sample_path, "rb") as f:
            resp = httpx.post(
                f"http://127.0.0.1:{PORT}/api/score/validate",
                files={"file": (sample_path.name, f.read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                timeout=30.0,
            )
        assert resp.status_code == 200, f"Validate upload failed: {resp.text}"
        val_data = resp.json()
        upload_id = val_data["upload_id"]
        print(f"[SCREENSHOT] Upload validated: upload_id={upload_id}, total_rows={val_data['total_rows']}", flush=True)

        # 2. Run scoring to create real scored results
        print("[SCREENSHOT] Running scoring execution...", flush=True)
        resp2 = httpx.post(
            f"http://127.0.0.1:{PORT}/api/score/run",
            json={"upload_id": upload_id},
            timeout=60.0,
        )
        assert resp2.status_code == 200, f"Run scoring failed: {resp2.text}"
        run_data = resp2.json()
        print(f"[SCREENSHOT] Scoring completed: scored={run_data['scored_count']}", flush=True)

        targets = [
            ("ui_score_upload.png", 1440, 950, f"http://127.0.0.1:{PORT}/score"),
            ("ui_score_single_student.png", 1440, 1100, f"http://127.0.0.1:{PORT}/score?single=true"),
            ("ui_score_check_columns.png", 1440, 1150, f"http://127.0.0.1:{PORT}/score?upload_id={upload_id}&step=2"),
            ("ui_score_results.png", 1440, 1300, f"http://127.0.0.1:{PORT}/score?upload_id={upload_id}"),
            ("ui_score_drawer.png", 1440, 1200, f"http://127.0.0.1:{PORT}/score?upload_id={upload_id}&student_id=TEST_STU_0037"),
        ]

        time.sleep(2.0)  # Allow DOM / fonts to settle

        for i, (filename, width, height, url) in enumerate(targets):
            out_path = OUTPUT_DIR / filename
            temp_profile = tempfile.mkdtemp(prefix=f"chrome_shot_score_{i}_")
            cmd = [
                CHROME_PATH,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--user-data-dir={temp_profile}",
                "--virtual-time-budget=5000",
                "--run-all-compositor-stages-before-draw",
                f"--window-size={width},{height}",
                f"--screenshot={out_path.resolve()}",
                url,
            ]
            print(f"  Capturing {width}x{height} -> {out_path}...")
            try:
                res = subprocess.run(cmd, capture_output=True, timeout=30)
                if out_path.exists():
                    print(f"  SUCCESS: {filename} ({out_path.stat().st_size / 1024:.1f} KB)")
                else:
                    print(f"  FAILED: {filename}")
                    if res.stderr:
                        print("  stderr:", res.stderr.decode("utf-8", errors="ignore"))
            finally:
                shutil.rmtree(temp_profile, ignore_errors=True)

    finally:
        print("[SCREENSHOT] Stopping uvicorn server...", flush=True)
        server_process.kill()
        server_log.close()


if __name__ == "__main__":
    main()
