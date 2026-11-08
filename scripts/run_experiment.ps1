param([int]$TimeoutSeconds=3600)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$environmentRoot=Join-Path $projectRoot '.conda\envs\pvsd-ml-v1'
$pythonExecutable=Join-Path $environmentRoot 'python.exe'
if(-not(Test-Path -LiteralPath $pythonExecutable)){throw 'Create the project conda environment first.'}
$logFolder=Join-Path $projectRoot 'reports\experiment_v1\logs'
New-Item -ItemType Directory -Path $logFolder -Force | Out-Null
$env:PATH="$environmentRoot;$environmentRoot\Library\bin;$environmentRoot\Scripts;$env:PATH"
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:MPLCONFIGDIR=Join-Path $projectRoot '.cache\matplotlib'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONUTF8='1'
$scriptFile=Join-Path $PSScriptRoot 'train_logo_v1.py'
$stdoutFile=Join-Path $logFolder 'training.log'
$stderrFile=Join-Path $logFolder 'stderr.log'
$trainingProcess=Start-Process -FilePath $pythonExecutable -ArgumentList @('-u',('"'+$scriptFile+'"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile -PassThru
Write-Output "Training PID: $($trainingProcess.Id). Logs: $stdoutFile. Hard timeout: $TimeoutSeconds seconds."
if(-not $trainingProcess.WaitForExit($TimeoutSeconds*1000)){
    Write-Output 'Hard timeout reached; stopping this experiment process.'
    Stop-Process -Id $trainingProcess.Id -Force
    throw 'Experiment exceeded the hard timeout. Check partial outputs; no automatic retry.'
}
$trainingProcess.Refresh()
Write-Output "Training exit code: $($trainingProcess.ExitCode)"
Get-Content -LiteralPath $stdoutFile -Tail 12
if($trainingProcess.ExitCode -ne 0){Get-Content -LiteralPath $stderrFile -Tail 35;throw 'Experiment failed; no automatic retry.'}
& $pythonExecutable -u (Join-Path $PSScriptRoot 'report_experiment_v1.py')
if($LASTEXITCODE -ne 0){throw 'Training succeeded but report verification failed.'}
& $pythonExecutable -u (Join-Path $PSScriptRoot 'build_result_views.py')
if($LASTEXITCODE -ne 0){throw 'Training succeeded but result-view generation failed.'}
