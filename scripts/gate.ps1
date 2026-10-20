param(
    [switch]$Fix
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:UV_CACHE_DIR = Join-Path $RepoRoot ".cache/uv"
$env:TMPDIR = Join-Path $RepoRoot ".cache/tmp"
$env:TMP = $env:TMPDIR
$env:TEMP = $env:TMPDIR
$env:PYTHONUTF8 = "1"

New-Item -ItemType Directory -Force -Path $env:TMPDIR | Out-Null

if ($Fix) {
    & uv run ruff format src scripts tests notebooks
    & uv run ruff check src scripts tests notebooks --fix
} else {
    & uv run ruff format --check src scripts tests notebooks
    & uv run ruff check src scripts tests notebooks
}

if ($LASTEXITCODE -ne 0) { exit 1 }

& uv run mypy
if ($LASTEXITCODE -ne 0) { exit 1 }

& uv run python scripts/validate_contracts.py
if ($LASTEXITCODE -ne 0) { exit 1 }

& uv run pytest --cov
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "All quality gates passed." -ForegroundColor Green
