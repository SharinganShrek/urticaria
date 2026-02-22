# Install all project dependencies in the virtual environment.
# Run from project root: .\install_dependencies.ps1
#
# If you haven't created a venv yet, run first:
#   python -m venv .venv
#   .\.venv\Scripts\Activate.ps1

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Done! You can now run:" -ForegroundColor Green
Write-Host "  python topic_analysis_keywords.py"
Write-Host "  python unmet_needs_analysis.py"
