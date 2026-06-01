param(
    [switch]$SetupOnly,
    [switch]$NoLaunch,
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$UvDir = Join-Path $Root ".tools\uv"
$UvExe = Join-Path $UvDir "uv.exe"
$UvZip = Join-Path $Root ".tools\uv-x86_64-pc-windows-msvc.zip"
$UvPythonDir = Join-Path $Root ".tools\uv-python"
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

function Remove-ProjectItem($Path) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $rootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd("\")
    if (-not $fullPath.StartsWith($rootPath + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Recusando remover caminho fora do projeto: $fullPath"
    }
    if (Test-Path $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    }
}

function Install-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return (Get-Command uv).Source
    }
    if (Test-Path $UvExe) {
        return $UvExe
    }

    Step "Baixando uv local"
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    $downloadUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -Uri $downloadUrl -OutFile $UvZip

    $extractDir = Join-Path $Root ".tools\uv-extract"
    if (Test-Path $extractDir) {
        Remove-ProjectItem $extractDir
    }
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    Expand-Archive -LiteralPath $UvZip -DestinationPath $extractDir -Force

    $candidate = Get-ChildItem -LiteralPath $extractDir -Recurse -Filter "uv.exe" | Select-Object -First 1
    if (-not $candidate) {
        throw "uv.exe nao foi encontrado no pacote baixado."
    }
    Copy-Item -LiteralPath $candidate.FullName -Destination $UvExe -Force
    Remove-ProjectItem $extractDir

    if (-not (Test-Path $UvExe)) {
        throw "uv local nao foi encontrado em $UvExe."
    }
    return $UvExe
}

function Create-VenvWithUv {
    $uv = Install-Uv
    Step "Criando ambiente virtual .venv com uv"
    New-Item -ItemType Directory -Force -Path $UvPythonDir | Out-Null
    $env:UV_PYTHON_INSTALL_DIR = $UvPythonDir
    & $uv python install "3.11"
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao baixar Python 3.11 com uv."
    }
    & $uv venv --seed --python "3.11" ".venv"
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao criar .venv com uv."
    }
    Assert-VenvReady
}

function Assert-VenvReady {
    $venvPython = Resolve-VenvPython
    if (-not $venvPython) {
        throw "Python do ambiente .venv nao encontrado apos criacao."
    }
    if (-not (Test-ProjectPython $venvPython)) {
        throw "Ambiente .venv criado, mas Python local nao atende aos requisitos: Windows 64-bit, Python 3.11 e Tkinter funcional."
    }
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
        Create-VenvWithUv
        return
    }

    Step "Criando ambiente virtual .venv"
    & $python.Exe @($python.Args) -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path (Join-Path $Root ".venv")) {
            Remove-ProjectItem (Join-Path $Root ".venv")
        }
        Write-Host "Python encontrado, mas venv/ensurepip falhou; usando uv como fallback." -ForegroundColor Yellow
        Create-VenvWithUv
    } else {
        Assert-VenvReady
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
