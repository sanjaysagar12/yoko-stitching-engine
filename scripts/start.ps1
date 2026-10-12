# One command: install, build the frontend, start the app.  Usage: ./scripts/start.ps1 [-Port 8000] [-SkipInstall]
param(
    [int]$Port = 8000,
    [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path (Join-Path $root "models\smplx"))) {
    Write-Warning "models\smplx not found: put SMPLX_MALE/FEMALE/NEUTRAL.npz there (they are never downloaded for you)."
}

if (-not $SkipInstall) {
    python -m pip install -e ".[web,dev]"
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

Push-Location (Join-Path $root "src\web\frontend")
try {
    if (-not $SkipInstall -or -not (Test-Path "node_modules")) {
        npm install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
}
finally {
    Pop-Location
}

python -m web.backend --port $Port
