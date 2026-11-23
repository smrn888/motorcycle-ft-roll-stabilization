param(
  [string]$Python = 'python',
  [string]$Benchmark = 'unified_fault_benchmark.py',
  [string]$Checkpoint = 'neural_mpc.pt',
  [string]$ResidualStem = 'res_main1m_best',
  [int]$Episodes = 100,
  [int]$Seed0 = 77000,
  [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
foreach ($p in @($Benchmark, $Checkpoint, "$ResidualStem.zip", "${ResidualStem}_vecnorm.pkl", 'motorcycle_env_residual.py', 'motorcycle_env_mpc19.py', 'motorcycle_dynamics.py', 'final_fault_benchmark.py', 'bench_stats.py', 'train_neural_mpc.py')) {
  if (-not (Test-Path $p)) { throw "Missing required file: $p" }
}
if ($Smoke) { $Episodes = 3 }
$scales = @(0, 2.5, 5, 7.5, 10)
foreach ($scale in $scales) {
  $str = $scale.ToString([System.Globalization.CultureInfo]::InvariantCulture)
  $tag = $str.Replace('.', 'p')
  $spec = "${ResidualStem}@${str}"
  foreach ($experiment in @('iw', 'fault')) {
    $out = "ablation_${experiment}_${tag}"
    $argsList = @($Benchmark, '--ckpt', $Checkpoint, '--res', $spec, '--methods', '--episodes', "$Episodes", '--seed0', "$Seed0", '--boot', '500', '--out', $out)
    if ($experiment -eq 'iw') {
      $argsList += @('--only', 'healthy', '--iw', '1.0', '0.45', '0.40', '0.30')
    } else {
      $argsList += @('--only', 'healthy', 'act_sat_30pct', 'act_delay_3', 'dropout_imu_p0.5', '--iw', '1.0')
    }
    Write-Host "Running $experiment / scale=$str -> $out"
    & $Python @argsList
    if ($LASTEXITCODE -ne 0) { throw "Failed: $out (exit=$LASTEXITCODE)" }
  }
}
& $Python analyze_ablation_verified.py --glob 'ablation_*_*_all_episodes.csv' --out ablation_verified
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed' }
