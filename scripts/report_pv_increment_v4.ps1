$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$environmentRoot=Join-Path $projectRoot '.conda\envs\pvsd-ml-v1'
$env:PATH="$environmentRoot;$environmentRoot\Library\bin;$environmentRoot\Scripts;$env:PATH"
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:MPLCONFIGDIR=Join-Path $projectRoot '.cache\matplotlib'
$env:PYTHONUTF8='1'
& (Join-Path $environmentRoot 'python.exe') -u (Join-Path $PSScriptRoot 'evaluate_pv_increment_v4.py')
if($LASTEXITCODE -ne 0){throw 'V4 consistency verification failed.'}
& (Join-Path $environmentRoot 'python.exe') -u (Join-Path $PSScriptRoot 'report_pv_increment_v4.py')
if($LASTEXITCODE -ne 0){throw 'V4 report generation failed.'}
