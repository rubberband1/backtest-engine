<#
.SYNOPSIS
    Starts, stops and inspects the forward test. Survives this shell closing.

.DESCRIPTION
    The forward test has to stay up for weeks, which means it cannot be a
    process attached to a terminal somebody eventually closes. This starts it
    genuinely detached, and records the PID so the same script can stop it or
    say whether it is still alive.

    Genuinely is the operative word. `Start-Process` is not enough: a caller
    running inside a Windows job object - a CI agent, a task runner, an
    automation harness - has its whole process tree killed when the job
    closes, and a child started that way goes with it. Measured here: the
    runner started with `Start-Process` was gone the moment the shell that
    launched it was reaped. The process is therefore created through WMI, so
    it is spawned by the WMI provider host and inherits none of the caller's
    job. It writes its own log rather than having a shell redirect for it,
    because a shell wrapper is a second process that would also have to
    survive for weeks.

    It is idempotent by construction: the runner takes a PID lock on the
    diary and a second instance refuses to start rather than doubling the
    position, so running -Start twice is safe.

    Dry run is the default and this script does not offer a way to turn it
    off. Sending orders is `python -m scripts.run_live ... --send`, typed
    deliberately, on an account the broker itself confirms is a demo.

.EXAMPLE
    ./scripts/forward_test.ps1 -Start
    ./scripts/forward_test.ps1 -Status
    ./scripts/forward_test.ps1 -Report
    ./scripts/forward_test.ps1 -Stop
#>
[CmdletBinding(DefaultParameterSetName = 'Status')]
param(
    [Parameter(ParameterSetName = 'Start')][switch]$Start,
    [Parameter(ParameterSetName = 'Stop')][switch]$Stop,
    [Parameter(ParameterSetName = 'Status')][switch]$Status,
    [Parameter(ParameterSetName = 'Report')][switch]$Report,

    [string]$Strategy = 'strategies/rsi-mean-reversion.json',
    [string]$Symbol = 'AUDUSD.r',
    [string]$Timeframe = 'H4',
    [double]$Equity = 100
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$liveDir = Join-Path $root 'logs\live'
$stem = "$([System.IO.Path]::GetFileNameWithoutExtension($Strategy))-$Symbol-$Timeframe"
$journal = Join-Path $liveDir "$stem.jsonl"
$logFile = Join-Path $liveDir "$stem.out.log"
$pidFile = Join-Path $liveDir "$stem.supervisor.pid"

function Get-RunnerProcess {
    if (-not (Test-Path $pidFile)) { return $null }
    $recorded = (Get-Content $pidFile -Raw).Trim([char]0xFEFF, ' ', "`r", "`n", "`t")
    if (-not $recorded) { return $null }
    try { return Get-Process -Id ([int]$recorded) -ErrorAction Stop } catch { return $null }
}

if ($Start) {
    $existing = Get-RunnerProcess
    if ($null -ne $existing) {
        Write-Host "already running (pid $($existing.Id)); nothing to do"
        exit 0
    }
    New-Item -ItemType Directory -Force -Path $liveDir | Out-Null

    $commandLine = '"{0}" -m scripts.run_live {1} --symbol {2} --timeframe {3} --equity {4} --log-level INFO --log-file "{5}"' -f `
        $python, $Strategy, $Symbol, $Timeframe, $Equity, $logFile

    # Win32_Process.Create rather than Start-Process: the new process is
    # spawned by the WMI service, so it belongs to no job object of this
    # shell's and survives the shell being killed rather than merely closed.
    # The PID that comes back is python's own, because nothing wraps it.
    $created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
        -Arguments @{ CommandLine = $commandLine; CurrentDirectory = $root }

    if ($created.ReturnValue -ne 0) {
        Write-Warning "WMI refused to create the process (code $($created.ReturnValue)); falling back to Start-Process, which does NOT survive a killed parent"
        $fallback = Start-Process -FilePath $python -WorkingDirectory $root `
            -WindowStyle Hidden -PassThru -ArgumentList @(
                '-m', 'scripts.run_live', $Strategy, '--symbol', $Symbol,
                '--timeframe', $Timeframe, '--equity', $Equity,
                '--log-level', 'INFO', '--log-file', $logFile
            )
        $processId = $fallback.Id
    } else {
        $processId = $created.ProcessId
    }

    # ascii, not utf8: PowerShell 5.1 writes a BOM with -Encoding utf8, and
    # the first thing this file does on the way back in is become an int
    Set-Content -Path $pidFile -Value $processId -Encoding ascii

    Write-Host "started pid $processId"
    Write-Host "  diary : $journal"
    Write-Host "  log   : $logFile"
    Write-Host "  stop  : ./scripts/forward_test.ps1 -Stop"
    exit 0
}

if ($Stop) {
    $process = Get-RunnerProcess
    if ($null -eq $process) {
        Write-Host 'not running'
        if (Test-Path $pidFile) { Remove-Item $pidFile }
        exit 0
    }
    # SIGTERM has no Windows equivalent that Python handles here, so the
    # process is stopped outright. It holds no unflushed state: the diary is
    # fsynced per line and an open position is the broker's, not the
    # process's - the next start reads both back.
    Stop-Process -Id $process.Id -Force
    Remove-Item $pidFile
    Write-Host "stopped pid $($process.Id)"
    exit 0
}

if ($Report) {
    & $python -m scripts.compare_live $journal --equity $Equity
    exit $LASTEXITCODE
}

$process = Get-RunnerProcess
if ($null -eq $process) {
    Write-Host 'forward test: not running'
} else {
    $uptime = (Get-Date) - $process.StartTime
    Write-Host "forward test: running (pid $($process.Id), up $([int]$uptime.TotalHours)h $($uptime.Minutes)m)"
}
if (Test-Path $journal) {
    $lines = (Get-Content $journal | Measure-Object -Line).Lines
    $size = [math]::Round((Get-Item $journal).Length / 1KB, 1)
    Write-Host "  diary : $lines events, $size KB, last written $((Get-Item $journal).LastWriteTime)"
} else {
    Write-Host '  diary : not created yet'
}
if (Test-Path $logFile) {
    $tail = (Get-Content $logFile -Tail 3) -join "`n    "
    if ($tail) { Write-Host "  log   : $tail" }
}
