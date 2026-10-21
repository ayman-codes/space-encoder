param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PRE_COMMIT_HOME = Join-Path $RepoRoot ".cache/pre-commit"
$env:UV_CACHE_DIR = Join-Path $RepoRoot ".cache/uv"
$env:TMPDIR = Join-Path $RepoRoot ".cache/tmp"
$env:TMP = $env:TMPDIR
$env:TEMP = $env:TMPDIR
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $env:TMPDIR "torchinductor"
$env:PYTHONUTF8 = "1"

New-Item -ItemType Directory -Force -Path $env:PRE_COMMIT_HOME, $env:TMPDIR | Out-Null

& uv run pre-commit run @Arguments
exit $LASTEXITCODE
