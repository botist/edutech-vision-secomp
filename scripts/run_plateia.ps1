param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& (Join-Path $Root "run.ps1") -SetupOnly
$Python = if (Test-Path (Join-Path $Root ".venv\Scripts\python.exe")) { Join-Path $Root ".venv\Scripts\python.exe" } else { Join-Path $Root ".venv\python.exe" }
& $Python -m edutech_vision --mode plateia --showcase --fullscreen --max-faces 24 --face-confidence 0.60 --audience-yaw-tolerance 30 --audience-pitch-tolerance 20 @ExtraArgs
