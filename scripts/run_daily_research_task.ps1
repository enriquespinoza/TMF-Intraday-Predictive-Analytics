param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Repo = 'C:\Dev\TMF-Intraday-Predictive-Analytics'
$Python = 'C:\Dev\.venvs\tmf-analytics\Scripts\python.exe'
$TimesFM = 'C:\Dev\.venvs\tmf-timesfm25\Scripts\python.exe'
$LogDir = Join-Path $Repo 'reports\automation\local_logs'

if (-not (Test-Path -LiteralPath $Repo -PathType Container)) { throw 'Repository not found' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Analytics Python missing' }
if (-not (Test-Path -LiteralPath $TimesFM -PathType Leaf)) { throw 'TimesFM Python missing' }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $LogDir ("TMF_cycle_$Stamp.log")

Push-Location $Repo
try {
    $RunArgs = @('-m', 'scripts.automate_daily_research', '--timesfm-python', $TimesFM)
    if ($DryRun) { $RunArgs += '--dry-run' }

    & $Python @RunArgs *> $Log
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "Research cycle failed with exit code $ExitCode; inspect $Log"
    }
    Write-Host "Research cycle completed. Log: $Log"
}
finally {
    Pop-Location
}
