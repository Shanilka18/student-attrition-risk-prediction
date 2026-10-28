<#
.SYNOPSIS
    PowerShell task runner for MEI (Multimodal Student Engagement Intelligence).
.DESCRIPTION
    Runs CLI pipeline commands, tests, formatting, and linting on Windows without make.
.EXAMPLE
    .\scripts\tasks.ps1 setup
    .\scripts\tasks.ps1 test
#>

param(
    [Parameter(Position=0, Mandatory=$false)]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# Determine Python path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

# Set MPLCONFIGDIR to prevent permission warnings
$env:MPLCONFIGDIR = Join-Path $ProjectRoot ".matplotlib_cache"
if (-not (Test-Path $env:MPLCONFIGDIR)) {
    New-Item -ItemType Directory -Path $env:MPLCONFIGDIR -Force | Out-Null
}

switch ($Task.ToLower()) {
    "setup" {
        & $PythonExe -m mei.cli setup
    }
    "make-sample-data" {
        & $PythonExe -m mei.cli make-sample-data
    }
    "clean" {
        & $PythonExe -m mei.cli clean
    }
    "features" {
        & $PythonExe -m mei.cli features
    }
    "states" {
        & $PythonExe -m mei.cli states
    }
    "train-classic" {
        & $PythonExe -m mei.cli train-classic
    }
    "train-deep" {
        & $PythonExe -m mei.cli train-deep
    }
    "train-regression" {
        & $PythonExe -m mei.cli train-regression
    }
    "finalize" {
        & $PythonExe -m mei.cli finalize
    }
    "explain" {
        & $PythonExe -m mei.cli explain
    }
    "serve" {
        & $PythonExe -m mei.cli serve
    }
    "retrain" {
        & $PythonExe -m mei.cli retrain
    }
    "docs" {
        & $PythonExe -m mei.cli docs
    }
    "all" {
        & $PythonExe -m mei.cli all
    }
    "test" {
        & $PythonExe -m pytest tests/
    }
    "lint" {
        & $PythonExe -m ruff check src tests
        & $PythonExe -m black --check src tests
    }
    "format" {
        & $PythonExe -m ruff check --fix src tests
        & $PythonExe -m black src tests
    }
    default {
        Write-Host "MEI Task Runner" -ForegroundColor Cyan
        Write-Host "Usage: .\scripts\tasks.ps1 <command>"
        Write-Host "Available commands:"
        Write-Host "  setup, make-sample-data, clean, features, states, train-classic,"
        Write-Host "  train-deep, train-regression, finalize, explain, serve, retrain, docs, all"
        Write-Host "  test, lint, format"
    }
}
