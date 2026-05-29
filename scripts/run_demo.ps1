param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& (Join-Path $Root "run.ps1") -SetupOnly
$Python = if (Test-Path (Join-Path $Root ".venv\Scripts\python.exe")) { Join-Path $Root ".venv\Scripts\python.exe" } else { Join-Path $Root ".venv\python.exe" }
& $Python -m edutech_vision --mode demo --showcase --fullscreen @ExtraArgs
