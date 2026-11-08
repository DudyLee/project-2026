param([int]$TimeoutSeconds=7200)
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
$logs=Join-Path $projectRoot 'reports\experiment_v3\logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$script=Join-Path $PSScriptRoot 'train_nested_v3.py'
$stdout=Join-Path $logs 'training.log'
$stderr=Join-Path $logs 'stderr.log'
$process=Start-Process -FilePath (Join-Path $environmentRoot 'python.exe') -ArgumentList @('-u',('"'+$script+'"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output "Nested v3 PID: $($process.Id); hard timeout $TimeoutSeconds seconds."
if(-not $process.WaitForExit($TimeoutSeconds*1000)){Write-Output 'Hard timeout reached; stopping this experiment.';Stop-Process -Id $process.Id -Force;throw 'Experiment hard timeout.'}
$process.Refresh()
Write-Output "Nested v3 exit code: $($process.ExitCode)"
Get-Content -LiteralPath $stdout -Tail 15
if($process.ExitCode -ne 0){Get-Content -LiteralPath $stderr -Tail 35;throw 'Nested experiment failed.'}
& (Join-Path $PSScriptRoot 'report_nested_v3.ps1')
