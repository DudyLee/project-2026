param([int]$TimeoutSeconds=3600)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$environmentRoot=Join-Path $projectRoot '.conda\envs\pvsd-ml-v1'
$env:PATH="$environmentRoot;$environmentRoot\Library\bin;$environmentRoot\Scripts;$env:PATH"
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:MPLCONFIGDIR=Join-Path $projectRoot '.cache\matplotlib'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONUTF8='1'
$logs=Join-Path $projectRoot 'reports\experiment_v2\logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$script=Join-Path $PSScriptRoot 'train_p10_v2.py'
$stdout=Join-Path $logs 'training.log'
$stderr=Join-Path $logs 'stderr.log'
$process=Start-Process -FilePath (Join-Path $environmentRoot 'python.exe') -ArgumentList @('-u',('"'+$script+'"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output "P10 v2 PID: $($process.Id); hard timeout $TimeoutSeconds seconds."
if(-not $process.WaitForExit($TimeoutSeconds*1000)){Write-Output 'Hard timeout reached; stopping this experiment.';Stop-Process -Id $process.Id -Force;throw 'Experiment hard timeout; no automatic retry.'}
$process.Refresh()
Write-Output "P10 v2 exit code: $($process.ExitCode)"
Get-Content -LiteralPath $stdout -Tail 20
if($process.ExitCode -ne 0){Get-Content -LiteralPath $stderr -Tail 35;throw 'P10 experiment failed; no automatic retry.'}
& (Join-Path $environmentRoot 'python.exe') -u (Join-Path $PSScriptRoot 'report_p10_v2.py')
if($LASTEXITCODE -ne 0){throw 'P10 experiment ran but verification/report generation failed.'}
& (Join-Path $environmentRoot 'python.exe') -u (Join-Path $PSScriptRoot 'build_p10_v2_views.py')
if($LASTEXITCODE -ne 0){throw 'P10 experiment ran but result-view generation failed.'}
