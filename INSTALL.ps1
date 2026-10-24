param([string]$Model = '', [switch]$WithoutLLM, [switch]$Plan)
$ErrorActionPreference = 'Stop'
$applicationRoot = Join-Path $PSScriptRoot 'application'
if ($WithoutLLM -and $Model) { throw 'Use either -Model or -WithoutLLM.' }
if ($Plan) {
    Write-Output 'Install missing uv and Tesseract using winget.'
    Write-Output 'Download Python 3.11, Python runtime dependencies, OCR data and FreeCAD.'
    Write-Output 'Extract FreeCAD using 7-Zip (install via winget if missing); reuse the SHA256-verified archive.'
    Write-Output 'If -Model is specified: install/start local Ollama and pull ONLY the selected vision models.'
    Write-Output "Requested models: $Model ; without LLM: $WithoutLLM"
    return
}
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { throw 'This build requires Windows x64.' }
if (-not $Model -and -not $WithoutLLM) {
    $Model = Read-Host 'Local vision model tag(s), e.g. qwen3.5:9b or gemma4:12b; comma-separated; Enter = CV/OCR only'
}
$models = @($Model.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -Unique)
foreach ($name in $models) {
    if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*:[A-Za-z0-9][A-Za-z0-9_.-]*$' -or $name -match 'cloud') {
        throw "Use an exact LOCAL Ollama name:tag, not an endpoint or cloud model: $name"
    }
}
function Refresh-InstallPath {
    $env:PATH = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User') + ';' + $env:PATH
}
function Install-WinGetPackage([string]$Id, [string]$Version = '') {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw 'WinGet is required. Install Microsoft App Installer, then retry.' }
    $arguments = @('install','--id',$Id,'-e','--source','winget','--accept-package-agreements','--accept-source-agreements')
    if ($Version) { $arguments += @('--version',$Version) }
    & winget @arguments
    if ($LASTEXITCODE -ne 0) { throw "Cannot install $Id (exit $LASTEXITCODE)." }
    Refresh-InstallPath
}
function Resolve-UV {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidate = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    return $null
}
function Resolve-Tesseract {
    if ($env:TESSERACT_CMD -and (Test-Path -LiteralPath $env:TESSERACT_CMD)) { return $env:TESSERACT_CMD }
    $command = Get-Command tesseract -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidate = Join-Path $env:ProgramFiles 'Tesseract-OCR\tesseract.exe'
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    return $null
}
function Resolve-Ollama {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    return $null
}
function Resolve-SevenZip {
    if ($env:SEVENZIP_CMD -and (Test-Path -LiteralPath $env:SEVENZIP_CMD -PathType Leaf)) { return $env:SEVENZIP_CMD }
    foreach ($name in @('7z.exe','7za.exe')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
    }
    foreach ($base in @($env:ProgramFiles, [Environment]::GetFolderPath('ProgramFilesX86'))) {
        if ($base) {
            $candidate = Join-Path $base '7-Zip\7z.exe'
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
        }
    }
    return $null
}
function Test-FreeCADRuntime([string]$Python) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { return $false }
    $previousHome = $env:PYTHONHOME; $previousPath = $env:PYTHONPATH
    try {
        Remove-Item Env:PYTHONHOME,Env:PYTHONPATH -ErrorAction SilentlyContinue
        & $Python -c "import FreeCAD,Part,Sketcher; s=Part.makeBox(1,2,3); assert s.isValid() and abs(s.Volume-6)<1e-9" >$null 2>$null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
    finally { $env:PYTHONHOME=$previousHome; $env:PYTHONPATH=$previousPath }
}
function Expand-FreeCADArchive([string]$ArchivePath, [string]$Destination, [string]$SevenZip) {
    # A separate -o argument remains one native argument with spaces/Cyrillic.
    # -aoa repairs files left by an interrupted/unsupported earlier extraction.
    & $SevenZip x $ArchivePath "-o$Destination" -y -aoa -bd
    if ($LASTEXITCODE -ne 0) { throw "FreeCAD extraction failed: 7-Zip exit $LASTEXITCODE. The downloaded archive was retained." }
}
function Local-Ollama([string]$Endpoint) {
    $request = [System.Net.WebRequest]::Create('http://127.0.0.1:11434' + $Endpoint)
    $request.Proxy = $null
    $request.Timeout = 2000
    $response = $request.GetResponse()
    try { $reader = New-Object System.IO.StreamReader($response.GetResponseStream()); try { return ($reader.ReadToEnd() | ConvertFrom-Json) } finally { $reader.Dispose() } }
    finally { $response.Close() }
}
Refresh-InstallPath
$uvPath = Resolve-UV
if (-not $uvPath) { Install-WinGetPackage 'astral-sh.uv'; $uvPath = Resolve-UV }
if (-not $uvPath) { throw 'uv was installed but not found. Reopen PowerShell and rerun this command.' }
$tesseractPath = Resolve-Tesseract
if (-not $tesseractPath) { Install-WinGetPackage 'tesseract-ocr.tesseract' '5.5.0.20241111'; $tesseractPath = Resolve-Tesseract }
if (-not $tesseractPath) { throw 'Tesseract was not found after installation.' }
Set-Location -LiteralPath $applicationRoot
$runtimeRoot = Join-Path $applicationRoot '.runtime'
$env:UV_CACHE_DIR = Join-Path $applicationRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtimeRoot 'python'
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
& $uvPath python install 3.11
if ($LASTEXITCODE -ne 0) { throw 'Python download failed.' }
$pythonPath = Join-Path $applicationRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & $uvPath venv .venv --python 3.11
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
& $uvPath pip install --python $pythonPath -r (Join-Path $applicationRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
$tessdata = Join-Path $runtimeRoot 'tessdata'
New-Item -ItemType Directory -Force -Path $tessdata | Out-Null
foreach ($language in @('eng','rus')) {
    $destination = Join-Path $tessdata "$language.traineddata"
    if (-not (Test-Path -LiteralPath $destination) -or (Get-Item -LiteralPath $destination).Length -eq 0) {
        & curl.exe -L --fail --retry 2 -o $destination "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/$language.traineddata"
        if ($LASTEXITCODE -ne 0) { throw "OCR language download failed: $language" }
    }
}
$freecadPython = Join-Path $runtimeRoot 'FreeCAD_1.1.4-Windows-x86_64-py311\bin\python.exe'
$freecadReady = Join-Path $runtimeRoot 'FreeCAD-1.1.4.ready'
if (-not ((Test-Path -LiteralPath $freecadReady) -and (Test-FreeCADRuntime $freecadPython))) {
    $archive = Join-Path $runtimeRoot 'FreeCAD.7z'
    $expectedHash = '4828741fc91ee37fafcdb97a1abacb18b04ba451ac4372d9ff7a7349b36f4d6d'
    if (-not (Test-Path -LiteralPath $archive) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
        & curl.exe -L --fail --retry 2 -o $archive 'https://github.com/FreeCAD/FreeCAD/releases/download/1.1.4/FreeCAD_1.1.4-Windows-x86_64-py311.7z'
        if ($LASTEXITCODE -ne 0) { throw 'FreeCAD download failed.' }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) { throw 'FreeCAD SHA256 mismatch.' }
    $sevenZipPath = Resolve-SevenZip
    if (-not $sevenZipPath) { Install-WinGetPackage '7zip.7zip'; $sevenZipPath = Resolve-SevenZip }
    if (-not $sevenZipPath) { throw '7-Zip was not found. Install 7zip.7zip or set SEVENZIP_CMD to its executable, then rerun INSTALL.ps1.' }
    Expand-FreeCADArchive $archive $runtimeRoot $sevenZipPath
    if (-not (Test-FreeCADRuntime $freecadPython)) { throw 'FreeCAD was extracted but its Python/Part/Sketcher check failed. Rerun INSTALL.ps1 to repair it.' }
    Set-Content -LiteralPath $freecadReady -Value $expectedHash -Encoding ASCII
}
if ($models.Count -gt 0) {
    $ollamaPath = Resolve-Ollama
    if (-not $ollamaPath) { Install-WinGetPackage 'Ollama.Ollama'; $ollamaPath = Resolve-Ollama }
    if (-not $ollamaPath) { throw 'Ollama was not found after installation.' }
    $env:OLLAMA_HOST = '127.0.0.1:11434'
    $ready = $false
    try { $null = Local-Ollama '/api/version'; $ready = $true } catch {}
    if (-not $ready) {
        Start-Process -FilePath $ollamaPath -ArgumentList @('serve') -WindowStyle Hidden | Out-Null
        for ($attempt=0; $attempt -lt 60; $attempt++) {
            try { $null = Local-Ollama '/api/version'; $ready = $true; break } catch { Start-Sleep -Milliseconds 500 }
        }
    }
    if (-not $ready) { throw 'Local Ollama did not start. Open the Ollama application, then retry.' }
    foreach ($name in $models) {
        $installed = @((Local-Ollama '/api/tags').models | ForEach-Object { $_.name })
        if ($name -notin $installed) {
            & $ollamaPath pull $name
            if ($LASTEXITCODE -ne 0) { throw "Cannot download selected model: $name" }
        }
        & $pythonPath -c "import sys; from app.ollama_client import OllamaClient,LLMSettings; c=OllamaClient(LLMSettings(model=sys.argv[1])); s=c.status(); print(s['reason']); sys.exit(0 if s['available'] else 1)" $name
        if ($LASTEXITCODE -ne 0) { throw "Selected model is not compatible with local image analysis: $name" }
    }
}
$selected = if ($models.Count) { $models[0] } else { 'qwen3.5:9b' }
$mode = if ($models.Count) { 'hybrid' } else { 'classic' }
$environmentPath = Join-Path $applicationRoot '.env'
$lines = @()
if (Test-Path -LiteralPath $environmentPath) { $lines = @(Get-Content -LiteralPath $environmentPath | Where-Object { $_ -notmatch '^\s*LLM_(MODEL|BASE_URL|ANALYSIS_MODE|ENABLED)\s*=' }) }
$lines += @('LLM_ENABLED=true','LLM_BASE_URL=http://127.0.0.1:11434',"LLM_MODEL=$selected","LLM_ANALYSIS_MODE=$mode")
[IO.File]::WriteAllLines($environmentPath, [string[]]$lines, (New-Object Text.UTF8Encoding($false)))
@{tesseract=$tesseractPath} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeRoot 'settings.json') -Encoding UTF8
Write-Host 'Installed successfully. Start: powershell -ExecutionPolicy Bypass -File .\START.ps1'
