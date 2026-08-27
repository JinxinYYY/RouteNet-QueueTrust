param(
    [ValidateSet("smoke", "dryrun", "formal")]
    [string]$Task = "smoke"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceRoot = (Resolve-Path (Join-Path $Here "..\..\..\RouteNet-Fermi")).Path
$PythonBin = Join-Path $SourceRoot ".conda-env\python.exe"
if (-not (Test-Path -LiteralPath $PythonBin)) {
    $PythonBin = "python"
}

switch ($Task) {
    "smoke" {
        & $PythonBin (Join-Path $Here "smoke_test.py")
    }
    "dryrun" {
        & $PythonBin (Join-Path $Here "run_five_seeds.py") `
            --source-root $SourceRoot `
            --python-bin $PythonBin `
            --dry-run
    }
    "formal" {
        & $PythonBin -u (Join-Path $Here "run_five_seeds.py") `
            --source-root $SourceRoot `
            --python-bin $PythonBin `
            --skip-environment-check
    }
}

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

