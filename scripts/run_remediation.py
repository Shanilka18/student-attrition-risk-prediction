from pathlib import Path
from mei.models.remediation import run_remediation_ladder

if __name__ == "__main__":
    res = run_remediation_ladder()
    print("Remediation execution finished!")
    for mid, d in res.items():
        win = d["winning_attempt"]
        print(f"Model: {mid}")
        print(f"  Winning Step: {win['step']} ({win['step_name']})")
        print(f"  Remediated Model ID: {win['model_id']}")
        print(f"  Remediated Val R²: {win['val_r2']:.4f}")
        print(f"  Remediated Gap: {win['gap']:.4f}")
        print(f"  Remediated Status: {win['overfit_status']}")
