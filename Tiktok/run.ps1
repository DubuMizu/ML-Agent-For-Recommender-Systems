<#
.SYNOPSIS
    Deploy the autonomous ML research agent and watch it work.

.DESCRIPTION
    One command. It starts the agent as a detached process writing to
    runs/agent_stdout.log, then takes over this window with the live dashboard.

    The two are deliberately separate processes. The agent runs for hours and
    must survive anything that happens to the terminal -- closing the dashboard,
    resizing the window, an accidental Ctrl-C -- so the dashboard is a pure
    reader of runs/status.json and never shares a process with the run.

.EXAMPLE
    .\run.ps1                          # start the agent + attach the dashboard
    .\run.ps1 -Hours 6 -MaxIterations 50
    .\run.ps1 -DashboardOnly           # attach to an agent already running
    .\run.ps1 -NoDashboard             # agent in the foreground, raw log
    .\run.ps1 -Stop                    # stop the running agent
    .\run.ps1 -Status                  # one-shot status, no live view
#>
[CmdletBinding()]
param(
    [int]$MaxIterations = 50,
    [double]$Hours = 6.0,
    [int]$Seeds = 3,
    [int]$TimeBudget = 900,
    [string]$Model = 'claude-opus-5',
    [string]$OpeningMove,
    [int]$OpeningTurns = 8,
    [switch]$DashboardOnly,
    [switch]$NoDashboard,
    [switch]$Stop,
    [switch]$Status,
    [switch]$Fresh
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$RunsDir = Join-Path $PSScriptRoot 'runs'
$PidFile = Join-Path $RunsDir 'agent.pid'
$OutLog  = Join-Path $RunsDir 'agent_stdout.log'
$ErrLog  = Join-Path $RunsDir 'agent_stderr.log'
if (-not (Test-Path $RunsDir)) { New-Item -ItemType Directory $RunsDir | Out-Null }

# The agent writes minus signs, arrows and Greek; without this the console codec
# raises mid-print on Windows and takes the turn's token accounting with it.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

function Get-AgentProcess {
    if (-not (Test-Path $PidFile)) { return $null }
    $agentPid = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $agentPid) { return $null }
    try { return Get-Process -Id ([int]$agentPid) -ErrorAction Stop } catch { return $null }
}

# ---------------------------------------------------------------------- stop --
if ($Stop) {
    $proc = Get-AgentProcess
    if ($null -eq $proc) {
        Write-Host 'No agent is running (no live PID in runs/agent.pid).' -ForegroundColor Yellow
    } else {
        Write-Host ("Stopping agent PID {0} ..." -f $proc.Id) -ForegroundColor Yellow
        Stop-Process -Id $proc.Id -Force -Confirm:$false
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        # A forced stop gives the agent no chance to record its own shutdown, so
        # status.json would keep saying alive=true and the dashboard would report
        # a deliberate stop as a red KILLED. Mark it here instead.
        python -c "import json,os,time;p=os.path.join('runs','status.json');d=json.load(open(p,encoding='utf-8'));d.update(alive=False,phase='stopped',phase_detail='stopped by the operator',updated=time.time());json.dump(d,open(p,'w',encoding='utf-8'),default=str)" 2>$null
        Write-Host 'Stopped. The journal at runs/journal.jsonl is intact; re-running resumes from it.' -ForegroundColor Green
    }
    return
}

# -------------------------------------------------------------------- status --
if ($Status) {
    python dashboard.py --once
    return
}

# ------------------------------------------------------------------- preflight -
if (-not $DashboardOnly) {
    Write-Host 'Preflight ...' -ForegroundColor Cyan
    python -c "import claude_agent_sdk, torch, numpy" 2>$null
    if (-not $?) {
        Write-Host 'Missing dependencies. Install them with:' -ForegroundColor Red
        Write-Host '    pip install claude-agent-sdk torch numpy pandas' -ForegroundColor Red
        return
    }

    $existing = Get-AgentProcess
    if ($null -ne $existing) {
        Write-Host ("An agent is already running (PID {0}). Attaching the dashboard instead." -f $existing.Id) -ForegroundColor Yellow
        Write-Host '    Use .\run.ps1 -Stop first if you meant to restart it.' -ForegroundColor Yellow
        $DashboardOnly = $true
    }
}

if ($Fresh -and -not $DashboardOnly) {
    # Archive rather than delete: the journal is the run's evidence, and a run
    # that silently erased its own history would be unscoreable.
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $archive = Join-Path $RunsDir ("archive-" + $stamp)
    New-Item -ItemType Directory $archive | Out-Null
    foreach ($f in @('journal.jsonl', 'status.json', 'RUN_LOG.md', 'ensemble_search.json', 'FINAL.json')) {
        $p = Join-Path $RunsDir $f
        if (Test-Path $p) { Move-Item $p $archive }
    }
    Write-Host ("Previous run archived to {0}" -f $archive) -ForegroundColor Yellow
}

# --------------------------------------------------------------- launch agent -
if (-not $DashboardOnly) {
    $agentArgs = @(
        'run_agent.py',
        '--max-iterations', $MaxIterations,
        '--hours', $Hours,
        '--seeds', $Seeds,
        '--time-budget-s', $TimeBudget,
        '--model', $Model
    )

    # An operator directive steers the opening turns and then expires. The value
    # is usually a path to a file, because a directive worth giving runs to
    # several paragraphs.
    if ($OpeningMove) {
        if (Test-Path $OpeningMove) {
            Write-Host ("Opening directive: {0} (first {1} turns)" -f $OpeningMove, $OpeningTurns) -ForegroundColor Cyan
        } else {
            Write-Host ("Opening directive: inline text (first {0} turns)" -f $OpeningTurns) -ForegroundColor Cyan
        }
        $agentArgs += @('--opening-move', $OpeningMove, '--opening-turns', $OpeningTurns)
    }

    if ($NoDashboard) {
        Write-Host 'Running the agent in the foreground. Ctrl-C stops it.' -ForegroundColor Cyan
        python @agentArgs
        return
    }

    Write-Host ("Starting agent: {0} iterations, {1}h budget, {2} seeds, {3}" -f `
                $MaxIterations, $Hours, $Seeds, $Model) -ForegroundColor Cyan
    # -WindowStyle Hidden, NOT -NoNewWindow. -NoNewWindow attaches the agent to
    # THIS console, which means a Ctrl-C aimed at the dashboard -- or closing the
    # window -- delivers the signal to the whole console process group and kills
    # a six-hour run mid-epoch. A hidden console of its own is what actually
    # makes "press q, the agent keeps running" true.
    $proc = Start-Process -FilePath 'python' -ArgumentList $agentArgs `
        -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog `
        -WindowStyle Hidden -PassThru
    Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii
    Write-Host ("Agent PID {0}; log at runs/agent_stdout.log" -f $proc.Id) -ForegroundColor Green

    # The first status write happens after the SDK finds the CLI and the dataset
    # is encoded, which is tens of seconds. Waiting here means the dashboard's
    # first frame is real state rather than a "no status file" panel.
    for ($i = 0; $i -lt 40; $i++) {
        if (Test-Path (Join-Path $RunsDir 'status.json')) { break }
        if ($proc.HasExited) {
            Write-Host 'The agent exited during start-up. Last output:' -ForegroundColor Red
            if (Test-Path $ErrLog) { Get-Content $ErrLog -Tail 25 }
            if (Test-Path $OutLog) { Get-Content $OutLog -Tail 25 }
            Remove-Item $PidFile -ErrorAction SilentlyContinue
            return
        }
        Start-Sleep -Milliseconds 500
    }
}

# ------------------------------------------------------------------ dashboard -
Write-Host 'Attaching dashboard - press q to detach (the agent keeps running).' -ForegroundColor Cyan
Start-Sleep -Milliseconds 400
python dashboard.py

Write-Host ''
$proc = Get-AgentProcess
if ($null -ne $proc) {
    Write-Host ("Agent still running (PID {0})." -f $proc.Id) -ForegroundColor Green
    Write-Host '  .\run.ps1 -DashboardOnly    reattach the dashboard'
    Write-Host '  .\run.ps1 -Status           one-shot snapshot'
    Write-Host '  .\run.ps1 -Stop             stop the agent'
    Write-Host '  Get-Content runs\agent_stdout.log -Wait -Tail 40    raw log'
} else {
    Write-Host 'The agent is no longer running.' -ForegroundColor Yellow
    Write-Host '  runs\RUN_LOG.md      per-iteration evidence'
    Write-Host '  runs\FINAL.json      the submission it designated'
    Write-Host '  runs\resources.json  tokens, cost, wall clock, interventions'
}
