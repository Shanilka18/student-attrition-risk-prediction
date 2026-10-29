# PowerShell Task Runner for Windows (SLIIT CDAP Project KND-CDAP-IT-05)
# Component: Environmental & Financial Factors — Study Conditions Index
# Usage: .\tasks.ps1 <target>
# Valid targets: setup, data, clean, train, leaderboard, explain, causal, profiles, api, frontend, test, all

param(
    [Parameter(Position=0)]
    [string]$Target = "test"
)

$ErrorActionPreference = "Stop"
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$VenvPytest = Join-Path $PSScriptRoot ".venv\Scripts\pytest.exe"

switch ($Target) {
    "setup" {
        Write-Host "[TASK: setup] Provisioning virtual environment and installing pinned dependencies..." -ForegroundColor Cyan
        if (-not (Test-Path $VenvPython)) {
            py -3.11 -m venv .venv
        }
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r requirements.txt
        Write-Host "[TASK: setup] Dependencies installed successfully." -ForegroundColor Green
    }
    "data" {
        Write-Host "[TASK: data] Generating synthetic validation dataset..." -ForegroundColor Cyan
        & $VenvPython -m src.data_gen.generate --n 2000 --seed 42
        Write-Host "[TASK: data] Synthetic dataset generated successfully." -ForegroundColor Green
    }
    "clean" {
        Write-Host "[TASK: clean] Cleaning survey data, running EDA, and creating partitions..." -ForegroundColor Cyan
        & $VenvPython -m src.cleaning.cleaner
        & $VenvPython -m src.cleaning.eda
        & $VenvPython -m src.preprocessing.split
        Write-Host "[TASK: clean] Data cleaned, EDA generated, and partitions created successfully." -ForegroundColor Green
    }
    "train" {
        Write-Host "[TASK: train] Training model zoo across 4 algorithm categories..." -ForegroundColor Cyan
        & $VenvPython -m src.training.run_all --track C --models all --quick --no-tune
        Write-Host "[TASK: train] Model training completed successfully." -ForegroundColor Green
    }
    "leaderboard" {
        Write-Host "[TASK: leaderboard] Scoring models, running significance tests, calibrating, and auditing..." -ForegroundColor Cyan
        & $VenvPython -c "from src.evaluation.evaluate_phase5 import run_phase5_evaluation; run_phase5_evaluation()"
        Write-Host "[TASK: leaderboard] Leaderboard, calibration, and audit completed successfully." -ForegroundColor Green
    }
    "explain" {
        Write-Host "[TASK: explain] Computing SHAP attributions, narratives, fold stability, and generating plots..." -ForegroundColor Cyan
        & $VenvPython -c "from src.explain.run_phase6 import run_phase6_explainability; run_phase6_explainability()"
        Write-Host "[TASK: explain] Phase 6 SHAP explainability pipeline completed successfully." -ForegroundColor Green
    }
    "causal" {
        Write-Host "[TASK: causal] Estimating causal risk differences with sIPW, AIPW, Causal Forest, and refutations..." -ForegroundColor Cyan
        & $VenvPython -m src.causal.run_phase7
        Write-Host "[TASK: causal] Phase 7 Causal Inference pipeline completed successfully." -ForegroundColor Green
    }
    "predict" {
        param([string]$File = "data/legacy/samples/sample_upload.csv")
        Write-Host "[TASK: predict] Scoring input survey questionnaire: $File" -ForegroundColor Cyan
        & $VenvPython -m src.inference.predict --file $File --output_dir artifacts/predictions
        Write-Host "[TASK: predict] Inference completed. Deliverables in artifacts/predictions." -ForegroundColor Green
    }
    "api" {
        Write-Host "[TASK: api] Launching FastAPI backend microservice..." -ForegroundColor Cyan
        & $VenvPython -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    }
    "frontend" {
        Write-Host "[TASK: frontend] Launching React counselor dashboard..." -ForegroundColor Cyan
        Push-Location frontend
        try {
            npm run dev
        } finally {
            Pop-Location
        }
    }
    "profiles" {
        Write-Host "[TASK: profiles] Showing latent risk profile configuration..." -ForegroundColor Cyan
        & $VenvPython -m efi.cli profiles
    }
    "test" {
        Write-Host "[TASK: test] Executing automated test suite..." -ForegroundColor Cyan
        & $VenvPytest tests/ -v
        Write-Host "[TASK: test] All tests passed green." -ForegroundColor Green
    }
    "all" {
        Write-Host "[TASK: all] Executing end-to-end research pipeline..." -ForegroundColor Cyan
        & $PSCommandPath data
        & $PSCommandPath clean
        & $PSCommandPath train
        & $PSCommandPath leaderboard
        & $PSCommandPath explain
        & $PSCommandPath causal
        & $PSCommandPath test
        Write-Host "[TASK: all] End-to-end execution completed." -ForegroundColor Green
    }
    Default {
        Write-Host "Error: Target '$Target' not recognized." -ForegroundColor Red
        Write-Host "Available targets: setup, data, clean, train, leaderboard, explain, causal, profiles, api, frontend, test, all."
        exit 1
    }
}
