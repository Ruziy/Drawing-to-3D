param([int]$Port = 8000, [switch]$Classic)
$ErrorActionPreference = 'Stop'
$applicationRoot = Join-Path $PSScriptRoot 'application'
$pythonPath = Join-Path $applicationRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run INSTALL.ps1 first (see README.txt).' }
$settingsPath = Join-Path $applicationRoot '.runtime\settings.json'
if (Test-Path -LiteralPath $settingsPath) { $settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json; $env:TESSERACT_CMD = $settings.tesseract }
if ($Classic) { $env:LLM_ANALYSIS_MODE = 'classic' }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $applicationRoot
Write-Host "Drawing-to-3D: http://127.0.0.1:$Port ; stop: Ctrl+C"
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
