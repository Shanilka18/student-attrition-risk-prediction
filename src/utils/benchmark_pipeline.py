"""Pipeline Benchmark & Reproducibility Runner.

Executes each research phase end-to-end with fixed seed 42,
records exact wall-clock execution times, and checks determinism.
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable

PHASES = [
    ("Phase 1: Synthetic Data Generation", [PYTHON_EXE, "-m", "src.data_gen.generate", "--n", "2000", "--seed", "42"]),
    ("Phase 2: Cleaning, EDA & Splitting", [PYTHON_EXE, "-m", "src.cleaning.cleaner"]),
    ("Phase 2b: EDA Generation", [PYTHON_EXE, "-m", "src.cleaning.eda"]),
    ("Phase 2c: Preprocessing & Split", [PYTHON_EXE, "-m", "src.preprocessing.split"]),
    ("Phase 4: Model Zoo Training (Track C)", [PYTHON_EXE, "-m", "src.training.run_all", "--track", "C", "--models", "all", "--quick", "--no-tune"]),
    ("Phase 5: Leaderboard & Selection", [PYTHON_EXE, "-c", "from src.evaluation.evaluate_phase5 import run_phase5_evaluation; run_phase5_evaluation()"]),
    ("Phase 6: SHAP Explainability", [PYTHON_EXE, "-c", "from src.explain.run_phase6 import run_phase6_explainability; run_phase6_explainability()"]),
    ("Phase 7: Observational Causal Inference", [PYTHON_EXE, "-m", "src.causal.run_phase7"]),
]

def run_benchmark():
    print("=" * 70)
    print("STARTING END-TO-END PIPELINE BENCHMARK (seed=42)")
    print("=" * 70)
    
    results = []
    total_start = time.perf_counter()
    
    for name, cmd in PHASES:
        print(f"\n[RUNNING] {name}...")
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        duration = time.perf_counter() - start
        
        status = "SUCCESS" if proc.returncode == 0 else "FAILED"
        print(f"[{status}] {name} completed in {duration:.2f}s")
        if proc.returncode != 0:
            print(f"STDERR:\n{proc.stderr[:1000]}")
            results.append({"phase": name, "duration_s": round(duration, 2), "status": status, "error": proc.stderr[:500]})
            break
        else:
            results.append({"phase": name, "duration_s": round(duration, 2), "status": status})
            
    total_duration = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETED IN {total_duration:.2f}s")
    print("=" * 70)
    for r in results:
        print(f"  - {r['phase']:<42} : {r['duration_s']:>6.2f}s [{r['status']}]")
        
    benchmark_file = PROJECT_ROOT / "artifacts" / "pipeline_benchmark.json"
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_file, "w") as f:
        json.dump({"total_duration_s": round(total_duration, 2), "phases": results}, f, indent=2)
    print(f"\nBenchmark results saved to: {benchmark_file}")
    return results

if __name__ == "__main__":
    run_benchmark()
