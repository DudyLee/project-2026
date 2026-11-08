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
$logFolder=Join-Path $projectRoot 'reports\diagnosis_v1\logs'
New-Item -ItemType Directory -Path $logFolder -Force | Out-Null
$scriptFile=Join-Path $PSScriptRoot 'diagnose_rooms_v1.py'
$stdout=Join-Path $logFolder 'diagnosis.log'
$stderr=Join-Path $logFolder 'stderr.log'
$process=Start-Process -FilePath (Join-Path $environmentRoot 'python.exe') -ArgumentList @('-u',('"'+$scriptFile+'"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output "Diagnosis PID: $($process.Id); hard timeout $TimeoutSeconds seconds."
if(-not $process.WaitForExit($TimeoutSeconds*1000)){Write-Output 'Hard timeout reached; stopping the diagnostic process.';Stop-Process -Id $process.Id -Force;throw 'Diagnostic hard timeout; no automatic retry.'}
$process.Refresh()
Write-Output "Diagnosis exit code: $($process.ExitCode)"
Get-Content -LiteralPath $stdout -Tail 20
if($process.ExitCode -ne 0){Get-Content -LiteralPath $stderr -Tail 30;throw 'Diagnosis failed; no automatic retry.'}
& (Join-Path $environmentRoot 'python.exe') -u (Join-Path $PSScriptRoot 'report_diagnosis_v1.py')
if($LASTEXITCODE -ne 0){throw 'Diagnosis ran but report verification failed.'}
& (Join-Path $environmentRoot 'python.exe') -u (Join-Path $PSScriptRoot 'build_diagnosis_view.py')
if($LASTEXITCODE -ne 0){throw 'Diagnosis ran but the result view failed.'}
