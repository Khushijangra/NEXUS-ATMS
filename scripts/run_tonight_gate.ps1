Param(
    [int]$Timesteps = 30000,
    [string]$PythonExe = "C:\Python313\python.exe",
    [double]$GateMaxThroughputDropPct = 15.0,
    [int]$GateRequiredImprovedSeeds = 2,
    [string]$GateOutput = "results/d3qn_gate_report.json",
    [switch]$FinalizeRelease,
    [string]$ReleaseManifestOutput = "results/release_candidate.json",
    [string]$ReleasePolicyName = "release-gate-15.0",
    [switch]$GenerateReleaseNotes,
    [string]$ReleaseNotesOutput = "results/release_notes.md"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
Set-Location ..

Write-Host "[run] Accelerated multi-seed gate"
& $PythonExe scripts/run_multiseed_d3qn.py `
    --config configs/default.yaml `
    --timesteps $Timesteps `
    --seeds 42 123 999 `
    --n-episodes 5 `
    --python $PythonExe `
    --run-gate `
    --gate-required-improved-seeds $GateRequiredImprovedSeeds `
    --gate-max-throughput-drop-pct $GateMaxThroughputDropPct `
    --gate-output $GateOutput

Write-Host "`n[report] Gate decision"
if (Test-Path $GateOutput) {
    Get-Content $GateOutput
} else {
    Write-Host "$GateOutput not found"
}

Write-Host "`n[report] Multi-seed summary"
if (Test-Path results/d3qn_multiseed_summary.json) {
    Get-Content results/d3qn_multiseed_summary.json
} else {
    Write-Host "results/d3qn_multiseed_summary.json not found"
}

if ($FinalizeRelease) {
    Write-Host "`n[release] Finalizing release candidate manifest"
    & $PythonExe scripts/finalize_release_candidate.py `
        --gate-report $GateOutput `
        --summary results/d3qn_multiseed_summary.json `
        --output $ReleaseManifestOutput `
        --policy $ReleasePolicyName
}

    if ($GenerateReleaseNotes) {
        Write-Host "`n[release] Generating markdown release notes"
        & $PythonExe scripts/generate_release_notes.py `
        --release-manifest $ReleaseManifestOutput `
        --gate-report $GateOutput `
        --output $ReleaseNotesOutput
    }
