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

  API keys are never stored in the repository. The VLM key is taken from
  $env:OPENVIKING_VLM_API_KEY, else carried over from the existing live config,
  else read from 1Password when -OpRef is supplied. The embedding provider is a
  different vendor and carries its own key, resolved the same way from
  $env:OPENVIKING_EMBED_API_KEY, the live config, or -EmbedOpRef.

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
  [string]$EmbedOpRef = "op://Personal/Voxel Forge API Credentials - OpenViking/credential",
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

# The embedding provider is a different vendor from the VLM, so it carries its
# own credential. Same resolution order: environment, then the live config, then
# 1Password.
Step "Resolving the embedding API key"
$embedKey = $env:OPENVIKING_EMBED_API_KEY
$embedSource = "environment"
if (-not $embedKey) {
  $liveConf = Join-Path $OvHome "ov.conf"
  if (Test-Path $liveConf) {
    try {
      $existing = (Get-Content $liveConf -Raw | ConvertFrom-Json).embedding.dense.api_key
      if ($existing -and $existing -notmatch '^\$\{') { $embedKey = $existing; $embedSource = "existing live config" }
    } catch { }
  }
}
if (-not $embedKey -and $EmbedOpRef) {
  $embedKey = (op read $EmbedOpRef 2>$null)
  if ($LASTEXITCODE -eq 0 -and $embedKey) { $embedSource = "1Password ($EmbedOpRef)" } else { $embedKey = $null }
}
if (-not $embedKey) {
  Die "No embedding API key available. Set OPENVIKING_EMBED_API_KEY, keep an existing ov.conf, or pass -EmbedOpRef."
}
Note "embedding key resolved from $embedSource ($($embedKey.Length) chars, not logged)"

# ------------------------------------------------------------------ render conf
Step "Rendering ov.conf from the tracked template"
$templatePath = Join-Path $PSScriptRoot "ov.conf.template"
if (-not (Test-Path $templatePath)) { Die "Missing $templatePath" }
$rendered = (Get-Content $templatePath -Raw).
  Replace('${OPENVIKING_VLM_API_KEY}', $apiKey).
  Replace('${OPENVIKING_EMBED_API_KEY}', $embedKey).
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

# ------------------------------------------------------------------ stop first
# The server must be down BEFORE installing. A running server holds its native
# extensions open, and Windows refuses to replace a loaded .pyd: the first real
# deploy failed on Crypto/Cipher/_raw_aes.pyd with "Access is denied" AFTER uv
# had already staged 166 packages. That left the worst possible state - a live
# server whose install was broken, still serving only because its modules were
# already in memory, and guaranteed to fail on the next restart.
Step "Stopping the server before touching the install"
if ($DryRun) { Note "DRY RUN: would stop scheduled task '$TaskName' and any orphans" }
else {
  Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
  Get-Process | Where-Object { $_.Path -like "*uv\tools\openviking*" } | ForEach-Object {
    Note "killing PID $($_.Id)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
  }
  Start-Sleep -Seconds 3
  $still = @(Get-Process | Where-Object { $_.Path -like "*uv\tools\openviking*" })
  if ($still.Count) { Die "$($still.Count) process(es) still holding the install; refusing to install over a live server." }
  Note "server stopped, nothing holding the files"
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
      if ($LASTEXITCODE -ne 0) {
        Die ("uv tool install failed with exit code $LASTEXITCODE. The install may be " +
             "partially applied - re-run this script before starting the server.")
      }
    } finally { Pop-Location }
  }
}

# --------------------------------------------------------------------- restart
Step "Starting the server"
if ($DryRun) { Note "DRY RUN: would start scheduled task '$TaskName'" }
else {
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
