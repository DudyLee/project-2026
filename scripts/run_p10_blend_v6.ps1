param([int]$TimeoutSeconds=1800)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$environmentRoot=Join-Path $projectRoot '.conda\envs\pvsd-ml-v1'
$env:PATH="$environmentRoot;$environmentRoot\Library\bin;$environmentRoot\Scripts;$env:PATH"
$env:OMP_NUM_THREADS='4';$env:OPENBLAS_NUM_THREADS='4';$env:MKL_NUM_THREADS='4';$env:PYTHONUTF8='1';$env:PYTHONUNBUFFERED='1'
$logs=Join-Path $projectRoot 'reports\experiment_v6\logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
if(Test-Path -LiteralPath (Join-Path $projectRoot 'reports\experiment_v6\run_manifest.json')){throw 'V6 already started; refusing overwrite.'}
$script=Join-Path $PSScriptRoot 'run_p10_blend_v6.py'
$process=Start-Process -FilePath (Join-Path $environmentRoot 'python.exe') -ArgumentList @('-u',('"'+$script+'"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'experiment.log') -RedirectStandardError (Join-Path $logs 'stderr.log') -PassThru
Write-Output "V6 process $($process.Id), timeout $TimeoutSeconds seconds"
if(-not $process.WaitForExit($TimeoutSeconds*1000)){Write-Output 'Hard timeout; stopping V6 process';Stop-Process -Id $process.Id -Force;throw 'V6 timeout'}
$process.Refresh();Get-Content -LiteralPath (Join-Path $logs 'experiment.log') -Tail 30
if($process.ExitCode -ne 0){Get-Content -LiteralPath (Join-Path $logs 'stderr.log') -Tail 30;throw 'V6 failed'}
