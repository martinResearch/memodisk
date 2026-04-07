param(
    [string[]]$PythonVersions = @("3.12", "3.13", "3.14"),
    [string]$VenvRoot = ".python-venvs",
    [switch]$Recreate,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return $PSScriptRoot
}

function Get-UvCommand {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        throw "uv is required but was not found on PATH. Install uv first: https://docs.astral.sh/uv/"
    }
    return $uv.Source
}

function Get-VenvName {
    param([string]$PythonVersion)

    return "py" + $PythonVersion.Replace(".", "")
}

function Get-VenvPath {
    param(
        [string]$Root,
        [string]$PythonVersion
    )

    return Join-Path $Root (Get-VenvName -PythonVersion $PythonVersion)
}

function Ensure-Interpreter {
    param(
        [string]$Uv,
        [string]$PythonVersion
    )

    Write-Host "Installing Python $PythonVersion via uv..."
    & $Uv python install $PythonVersion
}

function Ensure-Venv {
    param(
        [string]$Uv,
        [string]$PythonVersion,
        [string]$VenvPath,
        [bool]$ForceRecreate
    )

    if ($ForceRecreate -and (Test-Path $VenvPath)) {
        Write-Host "Removing existing venv at $VenvPath"
        Remove-Item -Recurse -Force $VenvPath
    }

    if (-not (Test-Path $VenvPath)) {
        Write-Host "Creating venv for Python $PythonVersion at $VenvPath"
        & $Uv venv $VenvPath --python $PythonVersion
    }
    else {
        Write-Host "Using existing venv for Python $PythonVersion at $VenvPath"
    }
}

function Sync-Project {
    param(
        [string]$Uv,
        [string]$RepoRoot,
        [string]$VenvPath,
        [string]$PythonVersion
    )

    Write-Host "Syncing memodisk into $VenvPath for Python $PythonVersion"
    Push-Location $RepoRoot
    try {
        & $Uv pip install --python (Join-Path $VenvPath "Scripts\python.exe") -e ".[test,dev]"
    }
    finally {
        Pop-Location
    }
}

$repoRoot = Get-RepoRoot
$uv = Get-UvCommand
$venvRootPath = Join-Path $repoRoot $VenvRoot

New-Item -ItemType Directory -Force -Path $venvRootPath | Out-Null

foreach ($pythonVersion in $PythonVersions) {
    $venvPath = Get-VenvPath -Root $venvRootPath -PythonVersion $pythonVersion
    Ensure-Interpreter -Uv $uv -PythonVersion $pythonVersion
    Ensure-Venv -Uv $uv -PythonVersion $pythonVersion -VenvPath $venvPath -ForceRecreate:$Recreate.IsPresent

    if (-not $SkipInstall) {
        Sync-Project -Uv $uv -RepoRoot $repoRoot -VenvPath $venvPath -PythonVersion $pythonVersion
    }
}

Write-Host "Done. Created environments under $venvRootPath"