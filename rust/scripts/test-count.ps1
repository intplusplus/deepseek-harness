# Per-crate test counts.
#
# Attribution by binary stem is unreliable: integration-test binaries are named
# after their test file, not their crate, so a workspace transcript cannot be
# split per package. This runs cargo once per workspace member instead; every
# binary is already built, so each pass is a fraction of a second.
param([string]$TargetDir = $env:CARGO_TARGET_DIR)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot\..
try {
    $metadata = cargo metadata --format-version 1 --no-deps | ConvertFrom-Json
    $rows = @()
    foreach ($package in $metadata.packages) {
        $name = $package.name
        # Honour CARGO_TARGET_DIR instead of forcing a project-local target:
        # the project directory is backed up per modified file by the host, so a
        # 4 GB target tree there also becomes 4 GB of backups.
        $cargoArgs = @("test", "-p", $name)
        if ($TargetDir) { $cargoArgs += @("--target-dir", $TargetDir) }
        $output = cargo @cargoArgs 2>&1 | Out-String
        $passed = 0
        foreach ($m in [regex]::Matches($output, 'test result: ok\. (\d+) passed')) {
            $passed += [int]$m.Groups[1].Value
        }
        if ($output -match 'test result: FAILED') { $passed = "FAILED" }
        $rows += [pscustomobject]@{ Crate = $name; Tests = $passed }
    }
    $rows | Sort-Object Crate | Format-Table -AutoSize
    ($rows | Where-Object { $_.Tests -is [int] } | Measure-Object -Property Tests -Sum).Sum
}
finally {
    Pop-Location
}
