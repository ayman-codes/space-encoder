param(
    [switch]$Down,
    [switch]$Gpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path (Join-Path $RepoRoot "docker") "docker-compose.yml"
$EnvFile = Join-Path $RepoRoot ".env"

if (-not (Test-Path $EnvFile)) {
    throw "Missing .env file. Copy .env.example to .env and fill in the service credentials."
}

$composeArguments = @("-f", $ComposeFile, "--env-file", $EnvFile)
if ($Gpu) {
    $composeArguments += @("--profile", "gpu")
}

if ($Down) {
    docker compose @composeArguments down
    Write-Host "Services stopped." -ForegroundColor Yellow
    exit 0
}

$engine = docker info --format "{{.ServerVersion}}" 2>$null
if (-not $engine) {
    Write-Host "Starting Docker Desktop..." -ForegroundColor Cyan
    Start-Process -FilePath "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Seconds 5
        $engine = docker info --format "{{.ServerVersion}}" 2>$null
        if ($engine) { break }
    }
}

if (-not $engine) {
    throw "Docker engine did not become available."
}

docker compose @composeArguments up -d --build
Write-Host "Waiting for services..." -ForegroundColor Cyan

$healthy = $false
for ($attempt = 0; $attempt -lt 12; $attempt++) {
    Start-Sleep -Seconds 5
    & uv run python scripts/health_check.py 2>$null
    if ($LASTEXITCODE -eq 0) {
        $healthy = $true
        break
    }
}

docker compose @composeArguments ps

if (-not $healthy) {
    Write-Warning "One or more services did not pass the health check."
    exit 1
}

Write-Host ""
Write-Host "MLflow      http://localhost:5001"
Write-Host "Prometheus  http://localhost:9092"
Write-Host "Pushgateway http://localhost:9091"
Write-Host "Grafana     http://localhost:3001"
if ($Gpu) { Write-Host "JupyterLab  http://localhost:8888" }
