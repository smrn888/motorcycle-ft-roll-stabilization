param([switch]$Smoke,[switch]$AllFaults,[switch]$Force)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$argsList = @('-u', 'run_multiseed.py')
if ($Smoke) { $argsList += '--smoke' }
if ($AllFaults) { $argsList += '--eval-all-faults' }
if ($Force) { $argsList += '--force' }
& python @argsList
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE; inspect multiseed_runs\pair_*\*.log" }
