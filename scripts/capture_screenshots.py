"""Script to capture headless screenshots of the MEI application shell.
Runs at 1440px, 1024px, and 390px responsive breakpoints.
"""

import os
import sys
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
import urllib.request

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_PATH):
    CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PORT = 8008

TARGETS = [
    ("ui_overview_1440.png", 1440, 1350, f"http://127.0.0.1:{PORT}/overview"),
    ("ui_overview_1024.png", 1024, 1100, f"http://127.0.0.1:{PORT}/overview"),
    ("ui_patterns_1440.png", 1440, 1600, f"http://127.0.0.1:{PORT}/patterns"),
    ("ui_patterns_1024.png", 1024, 1350, f"http://127.0.0.1:{PORT}/patterns"),
    ("ui_drivers_1440.png", 1440, 1800, f"http://127.0.0.1:{PORT}/drivers"),
    ("ui_drivers_1024.png", 1024, 1400, f"http://127.0.0.1:{PORT}/drivers"),
    ("ui_leaderboard_1440.png", 1440, 1200, f"http://127.0.0.1:{PORT}/leaderboard"),
    ("ui_leaderboard_1024.png", 1024, 950, f"http://127.0.0.1:{PORT}/leaderboard"),
    ("ui_model_detail_1440.png", 1440, 1800, f"http://127.0.0.1:{PORT}/models/knn"),
    ("ui_compare_1440.png", 1440, 1100, f"http://127.0.0.1:{PORT}/compare?ids=knn,gru,xgboost_hybrid"),
]

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
                print(f"  Waiting for server on {url} ({int(time.time() - t0)}s elapsed)...", flush=True)
                last_print = time.time()
            time.sleep(0.5)
    return False

def main():
    print(f"[SCREENSHOT] Launching server on http://127.0.0.1:{PORT}...", flush=True)
    server_log = open("uvicorn_capture.log", "w", encoding="utf-8")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "mei.api.app:create_app", "--host", "127.0.0.1", "--port", str(PORT), "--factory"],
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    try:
        if not wait_for_server():
            print("[ERROR] Server failed to start within timeout.")
            server_process.kill()
            server_log.close()
            with open("uvicorn_capture.log", "r", encoding="utf-8") as f:
                print(f.read())
            sys.exit(1)
            sys.exit(1)

        print("[SCREENSHOT] Server healthy. Capturing responsive viewports...", flush=True)
        time.sleep(2.0) # Allow DOM / fonts to settle

        for i, (filename, width, height, url) in enumerate(TARGETS):
            out_path = OUTPUT_DIR / filename
            temp_profile = tempfile.mkdtemp(prefix=f"chrome_shot_{i}_")
            cmd = [
                CHROME_PATH,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--user-data-dir={temp_profile}",
                "--virtual-time-budget=2500",
                "--run-all-compositor-stages-before-draw",
                f"--window-size={width},{height}",
                f"--screenshot={out_path.resolve()}",
                url,
            ]
            print(f"  Capturing {width}x{height} -> {out_path}...")
            try:
                res = subprocess.run(cmd, capture_output=True, timeout=25)
                if out_path.exists():
                    print(f"  SUCCESS: {filename} ({out_path.stat().st_size / 1024:.1f} KB)")
                else:
                    print(f"  FAILED: {filename}")
                    if res.stderr:
                        print("  stderr:", res.stderr.decode("utf-8", errors="ignore"))
            finally:
                shutil.rmtree(temp_profile, ignore_errors=True)

    finally:
        print("[SCREENSHOT] Terminating server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        try:
            server_log.close()
        except Exception:
            pass
        print("[SCREENSHOT] All captures completed.")

if __name__ == "__main__":
    main()
