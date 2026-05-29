param(
    [switch]$SetupOnly,
    [switch]$NoLaunch,
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$PythonVersion = "3.11.9"
$PythonDir = Join-Path $Root ".tools\python-$PythonVersion"
$PythonInstaller = Join-Path $Root ".tools\python-$PythonVersion-amd64.exe"
$BootstrapPython = Join-Path $PythonDir "python.exe"
$StampPath = Join-Path $Root ".venv\.edutech-bootstrap"

function Step($Message) {
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Assert-Windows {
    if (-not $IsWindows -and $PSVersionTable.PSVersion.Major -ge 6) {
        throw "Este bootstrap foi feito para Windows 10/11."
    }
    if ([Environment]::Is64BitOperatingSystem -ne $true) {
        throw "Windows 64-bit e necessario para os wheels do MediaPipe/OpenCV."
    }
}

function Get-CompatiblePython {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += @{ Exe = "py"; Args = @("-3.11-64") }
    }
    foreach ($name in @("python", "python3")) {
        if (Get-Command $name -ErrorAction SilentlyContinue) {
            $candidates += @{ Exe = $name; Args = @() }
        }
    }
    if (Test-Path $BootstrapPython) {
        $candidates += @{ Exe = $BootstrapPython; Args = @() }
    }

    foreach ($candidate in $candidates) {
        $probe = @"
import platform, sys, tkinter
ok = sys.version_info[:2] == (3, 11) and platform.architecture()[0] == '64bit'
raise SystemExit(0 if ok else 1)
"@
        if (Invoke-PythonProbe -Exe $candidate.Exe -PythonArgs $candidate.Args -Probe $probe) {
            return @{ Exe = $candidate.Exe; Args = $candidate.Args }
        }
    }
    return $null
}

function Invoke-PythonProbe($Exe, [string[]]$PythonArgs, $Probe) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @PythonArgs -c $Probe *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Install-ProjectPython {
    Step "Baixando Python $PythonVersion local"
    New-Item -ItemType Directory -Force -Path (Join-Path $Root ".tools") | Out-Null
    if (-not (Test-Path $PythonInstaller)) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $PythonInstaller
    }

    Step "Instalando Python local do projeto"
    $arguments = @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=0",
        "Include_launcher=0",
        "Include_pip=1",
        "Include_tcltk=1",
        "Include_test=0",
        "SimpleInstall=1",
        "TargetDir=`"$PythonDir`""
    )
    $process = Start-Process -FilePath $PythonInstaller -ArgumentList $arguments -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Instalador do Python falhou com codigo $($process.ExitCode)."
    }
    if (-not (Test-Path $BootstrapPython)) {
        throw "Python local nao foi encontrado em $BootstrapPython."
    }
    return @{ Exe = $BootstrapPython; Args = @() }
}

function Ensure-Venv {
    $existing = Resolve-VenvPython
    if ($existing -and (Test-ProjectPython $existing)) {
        return
    }
    if (Test-Path (Join-Path $Root ".venv")) {
        $backup = Join-Path $Root (".venv.backup-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
        Step "Arquivando .venv incompativel"
        Move-Item -LiteralPath (Join-Path $Root ".venv") -Destination $backup
    }
    $python = Get-CompatiblePython
    if ($null -eq $python) {
        $python = Install-ProjectPython
    }

    Step "Criando ambiente virtual .venv"
    & $python.Exe @($python.Args) -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao criar .venv."
    }
}

function Test-ProjectPython($PythonExe) {
    $probe = @"
import platform, sys, tkinter
ok = sys.version_info[:2] == (3, 11) and platform.architecture()[0] == '64bit'
raise SystemExit(0 if ok else 1)
"@
    return Invoke-PythonProbe -Exe $PythonExe -PythonArgs @() -Probe $probe
}

function Resolve-VenvPython {
    foreach ($path in @(
        (Join-Path $Root ".venv\Scripts\python.exe"),
        (Join-Path $Root ".venv\python.exe")
    )) {
        if (Test-Path $path) {
            return $path
        }
    }
    return $null
}

function Get-FileSha256($Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $stream = [System.IO.File]::OpenRead($resolved)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $bytes = $sha.ComputeHash($stream)
            return ([BitConverter]::ToString($bytes) -replace "-", "").ToUpperInvariant()
        } finally {
            $sha.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Get-SetupStamp {
    $files = @("requirements.txt", "pyproject.toml", "scripts/download_models.py")
    $text = foreach ($file in $files) {
        if (Test-Path $file) {
            Get-FileSha256 $file
        }
    }
    return ($text -join "|")
}

function Ensure-Dependencies {
    $VenvPython = Resolve-VenvPython
    if (-not $VenvPython) {
        throw "Python do ambiente .venv nao encontrado."
    }
    $stamp = Get-SetupStamp
    $current = if (Test-Path $StampPath) { Get-Content $StampPath -Raw } else { "" }
    if ($current.Trim() -eq $stamp) {
        return
    }

    Step "Instalando dependencias Python"
    & $VenvPython -m pip install --upgrade pip --no-warn-script-location
    if ($LASTEXITCODE -ne 0) { throw "Falha ao atualizar pip." }

    & $VenvPython -m pip install --no-warn-script-location -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar requirements.txt." }

    & $VenvPython -m pip install --no-warn-script-location -e .
    if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar pacote local." }

    Set-Content -Path $StampPath -Value $stamp -Encoding UTF8
}

function Ensure-Model {
    $VenvPython = Resolve-VenvPython
    if (-not $VenvPython) {
        throw "Python do ambiente .venv nao encontrado."
    }
    if ($SkipModel) {
        return
    }
    Step "Verificando modelos MediaPipe e YuNet"
    & $VenvPython scripts/download_models.py
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao verificar ou baixar modelos MediaPipe e YuNet."
    }
}

Assert-Windows
Ensure-Venv
Ensure-Dependencies
Ensure-Model

$VenvPython = Resolve-VenvPython
if (-not $VenvPython) {
    throw "Python do ambiente .venv nao encontrado."
}

if ($SetupOnly -or $NoLaunch) {
    Write-Host ""
    Write-Host "Setup concluido." -ForegroundColor Green
    exit 0
}

Step "Abrindo EduTech Vision Control Center"
& $VenvPython -m edutech_vision.launcher
