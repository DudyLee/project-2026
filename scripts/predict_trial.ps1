param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [ValidateSet('0.1','0.2')][string]$ModelVersion = '0.2',
    [string]$BimPath,
    [switch]$Overwrite
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$inputAbsolute = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InputPath)
$outputAbsolute = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPath)
$environmentRoot = Join-Path $projectRoot '.conda\envs\pvsd-ml-v1'
$pythonExe = Join-Path $environmentRoot 'python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Project conda environment missing. See docs/trial_v0_2_usage.md." }
$oldPath = $env:PATH
$oldUtf8 = $env:PYTHONUTF8
try {
    $env:PATH = "$environmentRoot;$environmentRoot\Library\bin;$environmentRoot\Scripts;$env:PATH"
    $env:PYTHONUTF8 = '1'
    $arguments = @((Join-Path $projectRoot 'pvsd_trial.py'), '--input', $inputAbsolute, '--output', $outputAbsolute)
    $arguments += @('--version', $ModelVersion)
    if ($BimPath) {
        $bimAbsolute = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BimPath)
        $arguments += @('--bim', $bimAbsolute)
    }
    if ($Overwrite) { $arguments += '--overwrite' }
    & $pythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Prediction failed (exit $LASTEXITCODE). See error above." }
} finally {
    $env:PATH = $oldPath
    $env:PYTHONUTF8 = $oldUtf8
}
