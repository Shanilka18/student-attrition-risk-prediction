"""Run the repository's repeatable release checks and report each result."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def command_for_npm() -> str:
    command = "npm.cmd" if os.name == "nt" else "npm"
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"Required frontend package manager not found: {command}"
        )
    return resolved


def run_check(name: str, command: list[str], cwd: Path = ROOT) -> dict[str, object]:
    print(f"\n[{name}] $ {' '.join(command)}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, check=False)
    elapsed = round(time.perf_counter() - started, 3)
    status = "passed" if result.returncode == 0 else "failed"
    print(
        f"[{name}] {status.upper()} ({elapsed:.1f}s, exit {result.returncode})",
        flush=True,
    )
    return {
        "name": name,
        "command": command,
        "duration_seconds": elapsed,
        "exit_code": result.returncode,
        "status": status,
    }


def main() -> int:
    npm = command_for_npm()
    checks = [
        ("full_pipeline", [sys.executable, "-m", "mei.cli", "all"], ROOT),
        ("python_tests", [sys.executable, "-m", "pytest"], ROOT),
        (
            "python_lint",
            [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
            ROOT,
        ),
        (
            "python_format",
            [sys.executable, "-m", "black", "--check", "src", "tests", "scripts"],
            ROOT,
        ),
        ("documentation", [sys.executable, "scripts/check_docs.py"], ROOT),
        ("frontend_install", [npm, "ci"], FRONTEND),
        ("frontend_tests", [npm, "test"], FRONTEND),
        ("frontend_build", [npm, "run", "build"], FRONTEND),
        ("frontend_tokens", [npm, "run", "check:tokens"], FRONTEND),
        ("frontend_e2e", [npm, "run", "test:e2e"], FRONTEND),
        ("frontend_production_audit", [npm, "audit", "--omit=dev"], FRONTEND),
    ]

    results: list[dict[str, object]] = []
    for name, command, cwd in checks:
        try:
            results.append(run_check(name, command, cwd))
        except OSError as exc:
            print(f"[{name}] FAILED: {exc}", file=sys.stderr, flush=True)
            results.append(
                {
                    "name": name,
                    "command": command,
                    "duration_seconds": 0,
                    "exit_code": 127,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    report_path = ROOT / "artifacts" / "release_checks"
    report_path.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_file = report_path / f"release_check_{stamp}.json"
    report_file.write_text(
        json.dumps(
            {
                "started_at": stamp,
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": (
                    "passed"
                    if all(item["status"] == "passed" for item in results)
                    else "failed"
                ),
                "checks": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nRelease-check report: {report_file.relative_to(ROOT)}", flush=True)
    failures = [item["name"] for item in results if item["status"] != "passed"]
    if failures:
        print(f"FAILED checks: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("All configured release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
