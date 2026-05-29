$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $Root "run.ps1") -SetupOnly
