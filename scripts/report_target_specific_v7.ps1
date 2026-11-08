$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$environmentRoot=Join-Path $projectRoot '.conda\envs\pvsd-ml-v1'
$env:PATH="$environmentRoot;$environmentRoot\Library\bin;$environmentRoot\Scripts;$env:PATH"
$env:PYTHONUTF8='1';$env:OMP_NUM_THREADS='4';$env:OPENBLAS_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$env:MPLCONFIGDIR=Join-Path $projectRoot '.cache\matplotlib'
& (Join-Path $environmentRoot 'python.exe') (Join-Path $PSScriptRoot 'verify_target_specific_v7.py')
if($LASTEXITCODE -ne 0){throw 'V7 verification failed'}
& (Join-Path $environmentRoot 'python.exe') (Join-Path $PSScriptRoot 'report_target_specific_v7.py')
if($LASTEXITCODE -ne 0){throw 'V7 report failed'}
