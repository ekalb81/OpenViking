<#
.SYNOPSIS
  Deploy OpenViking from this repository to the local tool install.

.DESCRIPTION
  Replaces the previous practice of copying individual .py files into the
  installed package. That produced a hybrid: most files from one release, a few
  from a much newer branch, with nothing checking the two were compatible.

  This installs the whole package at one commit, renders the config from a
  tracked template, restarts the server, and records what was deployed so
  check-drift.ps1 can verify it later.

  The API key is never stored in the repository. It is taken from
  $env:OPENVIKING_VLM_API_KEY, else carried over from the existing live config,
  else read from 1Password when -OpRef is supplied.

.EXAMPLE
  pwsh deploy/deploy.ps1                 # full deploy
  pwsh deploy/deploy.ps1 -DryRun         # show what would happen, change nothing
  pwsh deploy/deploy.ps1 -SkipInstall    # config + restart only
  pwsh deploy/deploy.ps1 -OpRef "op://Private/OpenRouter/credential"
#>
[CmdletBinding()]
param(
  [string]$RepoRoot   = (Resolve-Path "$PSScriptRoot/..").Path,
  [string]$OvHome     = "$env:USERPROFILE/.openviking",
  [string]$TaskName   = "OpenViking Server",
  [string]$HealthUrl  = "http://127.0.0.1:1933/health",
  [string]$OpRef      = "",
  [switch]$SkipInstall,
  [switch]$AllowDirty,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Note($m) { Write-Host "    $m" }
function Die($m)  { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- preconditions
Step "Checking the repository is in a deployable state"
Push-Location $RepoRoot
try {
  $dirty = git status --porcelain
  if ($dirty -and -not $AllowDirty) {
    Die "Working tree is dirty. Deploying an uncommitted tree makes the stamp a lie. Commit, or pass -AllowDirty."
  }
  $commit = (git rev-parse HEAD).Trim()
  $branch = (git rev-parse --abbrev-ref HEAD).Trim()
  Note "branch $branch at $($commit.Substring(0,9))"
  if ($dirty) { Note "WARNING: deploying a dirty tree (-AllowDirty)" }
} finally { Pop-Location }

# ------------------------------------------------------------------- the secret
Step "Resolving the VLM API key"
$apiKey = $env:OPENVIKING_VLM_API_KEY
$source = "environment"
if (-not $apiKey) {
  $liveConf = Join-Path $OvHome "ov.conf"
  if (Test-Path $liveConf) {
    try {
      $existing = (Get-Content $liveConf -Raw | ConvertFrom-Json).vlm.api_key
      if ($existing -and $existing -notmatch '^\$\{') { $apiKey = $existing; $source = "existing live config" }
    } catch { }
  }
}
if (-not $apiKey -and $OpRef) {
  $apiKey = (op read $OpRef 2>$null)
  if ($LASTEXITCODE -eq 0 -and $apiKey) { $source = "1Password ($OpRef)" } else { $apiKey = $null }
}
if (-not $apiKey) {
  Die "No API key available. Set OPENVIKING_VLM_API_KEY, keep an existing ov.conf, or pass -OpRef."
}
Note "key resolved from $source ($($apiKey.Length) chars, not logged)"

# ------------------------------------------------------------------ render conf
Step "Rendering ov.conf from the tracked template"
$templatePath = Join-Path $PSScriptRoot "ov.conf.template"
if (-not (Test-Path $templatePath)) { Die "Missing $templatePath" }
$rendered = (Get-Content $templatePath -Raw).
  Replace('${OPENVIKING_VLM_API_KEY}', $apiKey).
  Replace('${OPENVIKING_HOME}', $env:USERPROFILE.Replace('\', '/'))
if ($rendered -match '\$\{[A-Z_]+\}') {
  Die "Unsubstituted placeholder(s) remain: $($Matches[0]). Refusing to write a broken config."
}
try { $null = $rendered | ConvertFrom-Json } catch { Die "Rendered config is not valid JSON: $_" }

$confPath = Join-Path $OvHome "ov.conf"
if ($DryRun) { Note "DRY RUN: would write $confPath" }
else {
  if (Test-Path $confPath) {
    $backup = "$confPath.bak-$(Get-Date -Format yyyyMMdd-HHmmss)"
    Copy-Item $confPath $backup
    Note "backed up existing config -> $(Split-Path $backup -Leaf)"
  }
  Set-Content -Path $confPath -Value $rendered -Encoding utf8 -NoNewline
  Note "wrote $confPath"
}

# --------------------------------------------------------------------- install
if ($SkipInstall) { Step "Skipping install (-SkipInstall)" }
else {
  Step "Installing the package from this commit"
  if ($DryRun) { Note "DRY RUN: would run 'uv tool install --reinstall .' in $RepoRoot" }
  else {
    Push-Location $RepoRoot
    try {
      uv tool install --reinstall . 2>&1 | ForEach-Object { Note $_ }
      if ($LASTEXITCODE -ne 0) { Die "uv tool install failed with exit code $LASTEXITCODE" }
    } finally { Pop-Location }
  }
}

# --------------------------------------------------------------------- restart
Step "Restarting the server"
if ($DryRun) { Note "DRY RUN: would restart scheduled task '$TaskName'" }
else {
  Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
  Get-Process | Where-Object { $_.Path -like "*uv\tools\openviking*" } | ForEach-Object {
    Note "killing orphan PID $($_.Id)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
  }
  Start-Sleep -Seconds 2
  Start-ScheduledTask -TaskName $TaskName

  Note "waiting for health (startup takes ~15-30s)"
  $ok = $false
  for ($i = 1; $i -le 20; $i++) {
    Start-Sleep -Seconds 6
    try {
      $r = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 8
      Note "healthy after $($i*6)s: $($r.Content)"
      $ok = $true; break
    } catch { }
  }
  if (-not $ok) { Die "Server did not become healthy within 120s. Check ~/.openviking/logs/server.log.err" }
}

# ---------------------------------------------------------------- verification
Step "Verifying the running configuration"
if ($DryRun) { Note "DRY RUN: would verify resolved models" }
else {
  $py = "$env:APPDATA/uv/tools/openviking/Scripts/python.exe"
  $check = @'
from openviking_cli.utils.config import get_openviking_config
vlm = get_openviking_config().vlm
wm = vlm.for_working_memory()
print(f"    extraction model : {vlm.model}")
print(f"    working memory   : {wm.model}")
assert vlm.api_key, "no api key resolved"
'@
  $out = $check | & $py - 2>&1 | Where-Object { $_ -notmatch 'RAGFSBinding' }
  $out | ForEach-Object { Write-Host $_ }
  if ($LASTEXITCODE -ne 0) { Die "Configuration verification failed" }

  $stamp = "$commit $branch $(Get-Date -Format o)"
  Set-Content -Path (Join-Path $OvHome ".deployed-commit") -Value $stamp -Encoding utf8
  Note "stamped $(Join-Path $OvHome '.deployed-commit')"
}

Write-Host "`nDeploy complete." -ForegroundColor Green
Write-Host "Verify at any time with: pwsh deploy/check-drift.ps1"
