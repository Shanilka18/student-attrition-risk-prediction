$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "The project virtual environment is missing. Follow the setup steps in README.md first."
}

if (-not (Test-Path (Join-Path $ProjectRoot "artifacts\active_run.txt"))) {
    throw "No active model run exists. Run '.\.venv\Scripts\python.exe -m mei.cli all' first."
}

& $PythonExe -m mei.cli serve @args
exit $LASTEXITCODE
