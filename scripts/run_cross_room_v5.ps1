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
$logs=Join-Path $projectRoot 'reports\experiment_v5\logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$script=Join-Path $PSScriptRoot 'train_cross_room_v5.py'
$stdout=Join-Path $logs 'training.log'
$stderr=Join-Path $logs 'stderr.log'
if(Test-Path -LiteralPath (Join-Path $projectRoot 'reports\experiment_v5\run_manifest.json')){throw 'V5 already started; refusing overwrite.'}
$process=Start-Process -FilePath (Join-Path $environmentRoot 'python.exe') -ArgumentList @('-u',('"'+$script+'"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output "Cross-room v5 PID: $($process.Id); hard timeout $TimeoutSeconds seconds."
if(-not $process.WaitForExit($TimeoutSeconds*1000)){Write-Output 'Hard timeout reached; stopping this experiment.';Stop-Process -Id $process.Id -Force;throw 'Experiment hard timeout.'}
$process.Refresh()
Write-Output "Cross-room v5 exit code: $($process.ExitCode)"
Get-Content -LiteralPath $stdout -Tail 18
if($process.ExitCode -ne 0){Get-Content -LiteralPath $stderr -Tail 35;throw 'Cross-room v5 experiment failed.'}
