param(
    [string[]]$PythonVersions = @("3.12", "3.13", "3.14"),
    [string]$VenvRoot = ".python-venvs",
    [string[]]$PytestArgs = @(),
    [switch]$RunRuff,
    [switch]$RunTy
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return $PSScriptRoot
}

function Get-VenvName {
    param([string]$PythonVersion)

    return "py" + $PythonVersion.Replace(".", "")
}

function Get-VenvPython {
    param(
        [string]$Root,
        [string]$PythonVersion
    )

    $venvPath = Join-Path $Root (Get-VenvName -PythonVersion $PythonVersion)
    $pythonPath = Join-Path $venvPath "Scripts\python.exe"
    if (-not (Test-Path $pythonPath)) {
        throw "Missing environment for Python $PythonVersion at $venvPath. Run .\setup_python_venvs.ps1 first."
    }
    return $pythonPath
}

$repoRoot = Get-RepoRoot
$venvRootPath = Join-Path $repoRoot $VenvRoot
$failures = @()

Push-Location $repoRoot
try {
    foreach ($pythonVersion in $PythonVersions) {
        $pythonPath = Get-VenvPython -Root $venvRootPath -PythonVersion $pythonVersion

        Write-Host ""
        Write-Host "=== Python $pythonVersion ==="
        & $pythonPath --version

        Write-Host "> python -m pytest $($PytestArgs -join ' ')"
        & $pythonPath -m pytest @PytestArgs
        if ($LASTEXITCODE -ne 0) {
            $failures += "Python $pythonVersion failed: pytest"
            continue
        }

        if ($RunRuff) {
            Write-Host "> python -m ruff check ."
            & $pythonPath -m ruff check .
            if ($LASTEXITCODE -ne 0) {
                $failures += "Python $pythonVersion failed: ruff check ."
                continue
            }
        }

        if ($RunTy) {
            Write-Host "> python -m ty check ."
            & $pythonPath -m ty check .
            if ($LASTEXITCODE -ne 0) {
                $failures += "Python $pythonVersion failed: ty check ."
                continue
            }
        }
    }
}
finally {
    Pop-Location
}

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "Failures:"
    foreach ($failure in $failures) {
        Write-Host "- $failure"
    }
    exit 1
}

Write-Host ""
Write-Host "All commands passed for all requested Python versions."