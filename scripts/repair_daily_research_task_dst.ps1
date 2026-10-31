# One-time repair for pre-DST-switchover fixed-offset triggers.
# Change only the two weekly triggers; preserve task actions and principal.
$ErrorActionPreference = 'Stop'
$Name = 'TMF V3 TimesFM Rolling Research'
$Task = Get-ScheduledTask -TaskName $Name -ErrorAction Stop
$CurrentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
if ($Task.Principal.UserId -ne $CurrentUser) {
    throw "Task belongs to another principal; manual review required"
}
if ((Get-TimeZone).Id -ne 'Eastern Standard Time') {
    throw 'Windows timezone is not Eastern Standard Time'
}
$Existing = @($Task.Triggers)
if ($Existing.Count -ne 2) {
    throw "Expected two triggers; found $($Existing.Count). Do not overwrite."
}
$Times = @($Existing | ForEach-Object { ([datetime]$_.StartBoundary).ToString('HH:mm') } | Sort-Object)
if (($Times -join ',') -ne '17:30,20:15') {
    throw "Unexpected existing trigger times: $($Times -join ',')"
}
$Days = @('Monday','Tuesday','Wednesday','Thursday','Friday')
$Early = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Days -At '17:30'
$Late = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Days -At '20:15'
$LocalDate = (Get-Date).ToString('yyyy-MM-dd')
$Early.StartBoundary = "${LocalDate}T17:30:00"
$Late.StartBoundary = "${LocalDate}T20:15:00"
Set-ScheduledTask -TaskName $Name -Trigger @($Early,$Late) -ErrorAction Stop | Out-Null
$Updated = Get-ScheduledTask -TaskName $Name -ErrorAction Stop
Write-Host 'StartBoundary after DST repair:'
$Updated.Triggers | Select-Object StartBoundary, Enabled | Format-Table -AutoSize
Write-Host 'NextRunTime after DST repair:'
Get-ScheduledTaskInfo -TaskName $Name | Select-Object NextRunTime, LastRunTime, LastTaskResult | Format-Table -AutoSize
