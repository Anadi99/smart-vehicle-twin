$ErrorActionPreference = "Stop"

Write-Host "Running backend tests..."
Set-Location "$PSScriptRoot\..\backend"
python -m pytest
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Running frontend lint, test, build..."
Set-Location "$PSScriptRoot\..\frontend"
npm run lint
if ($LASTEXITCODE -ne 0) { exit 1 }
npm run test
if ($LASTEXITCODE -ne 0) { exit 1 }
npm run build
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "All checks passed."
