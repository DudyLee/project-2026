param([int]$TimeoutSeconds=1800)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$environmentRoot=Join-Path $projectRoot '.conda\envs\pvsd-ml-v1'
$env:PATH="$environmentRoot;$environmentRoot\Library\bin;$environmentRoot\Scripts;$env:PATH"
$env:OMP_NUM_THREADS='4';$env:OPENBLAS_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$env:MPLCONFIGDIR=Join-Path $projectRoot '.cache\matplotlib'
$env:PYTHONUTF8='1';$env:PYTHONUNBUFFERED='1'
$logs=Join-Path $projectRoot 'reports\cross_room_diagnosis_v5\logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
if(Test-Path -LiteralPath (Join-Path $projectRoot 'reports\cross_room_diagnosis_v5\run_manifest.json')){throw 'Diagnosis already started; refuse overwrite.'}
$script=Join-Path $PSScriptRoot 'diagnose_cross_room_v5.py'
$process=Start-Process -FilePath (Join-Path $environmentRoot 'python.exe') -ArgumentList @('-u',('"'+$script+'"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'analysis.log') -RedirectStandardError (Join-Path $logs 'stderr.log') -PassThru
Write-Output "Diagnostic process $($process.Id), timeout $TimeoutSeconds seconds"
if(-not $process.WaitForExit($TimeoutSeconds*1000)){Write-Output 'Hard timeout; stopping diagnostic process';Stop-Process -Id $process.Id -Force;throw 'Diagnostic timeout'}
$process.Refresh();Get-Content -LiteralPath (Join-Path $logs 'analysis.log') -Tail 12
if($process.ExitCode -ne 0){Get-Content -LiteralPath (Join-Path $logs 'stderr.log') -Tail 30;throw 'Diagnostic analysis failed'}
