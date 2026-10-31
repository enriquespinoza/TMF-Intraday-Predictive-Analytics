# Run once in PowerShell as the intended Windows account.
# Task Scheduler uses the local computer timezone (set to Eastern).
# Tasks run only when this user is logged on, so existing Schwab user-scoped
# credentials can be accessed. No automatic login or brokerage order permission.
$ErrorActionPreference = 'Stop'
$TaskName = 'TMF V3 TimesFM Rolling Research'
$Repo = 'C:\Dev\TMF-Intraday-Predictive-Analytics'
$Runner = Join-Path $Repo 'scripts\run_daily_research_task.ps1'

if (-not (Test-Path $Runner)) { throw "Missing runner: $Runner" }
if ((Get-TimeZone).Id -notin @('Eastern Standard Time')) {
    throw "Windows timezone must be Eastern Standard Time; found $((Get-TimeZone).Id)"
}

$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
    '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $Runner + '"'
) -WorkingDirectory $Repo
$Evening = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '17:30'
$Night = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '20:15'
# Keep recurring wall-clock times across DST transitions. An explicit UTC
# offset in StartBoundary fixes the trigger to that offset (wrong after DST).
# Offset-free boundaries instruct Task Scheduler to use local timezone/DST.
$LocalDate = (Get-Date).ToString('yyyy-MM-dd')
$Evening.StartBoundary = "${LocalDate}T17:30:00"
$Night.StartBoundary = "${LocalDate}T20:15:00"
$Principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task exists. Inspect it first; this installer will not silently replace it."
}
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger @($Evening, $Night) -Principal $Principal -Settings $Settings -Description 'Research-only rolling TMF model issuance at 17:30 ET and evaluation at 20:15 ET; no orders'
Write-Host "Registered: $TaskName"
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
