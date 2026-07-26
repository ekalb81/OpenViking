<#
.SYNOPSIS
  Report whether the running OpenViking install matches a known fork commit.

.DESCRIPTION
  The server runs from an installed package, not from the repo. Nothing normally
  tells you when those two diverge, and a hand-copied file drifts silently until
  something fails hours later inside a background commit.

  This compares every tracked .py file in the installed package against the tree
  of the commit recorded at deploy time, and exits non-zero if they differ.

  Exit codes: 0 in sync, 1 drift detected, 2 cannot verify.

.EXAMPLE
  pwsh deploy/check-drift.ps1
  pwsh deploy/check-drift.ps1 -Detailed
#>
[CmdletBinding()]
param(
  [string]$RepoRoot  = (Resolve-Path "$PSScriptRoot/..").Path,
  [string]$SitePkgs  = (Join-Path $env:APPDATA 'uv/tools/openviking/Lib/site-packages'),
  [string]$StampFile = (Join-Path $env:USERPROFILE '.openviking/.deployed-commit'),
  [switch]$Detailed
)

$ErrorActionPreference = 'Stop'

# PowerShell decodes a native command's stdout using the console codepage, which
# on this machine is not UTF-8. Without this, every source file containing a
# non-ASCII character (an em dash is enough) comes back as mojibake and is
# reported as drift - 124 false positives on the first run, which is worse than
# no checker at all because it trains you to ignore the output.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Test-Path $StampFile)) {
  Write-Host "NO STAMP: $StampFile is missing." -ForegroundColor Yellow
  Write-Host "The running install was not produced by deploy.ps1, so its provenance is unknown."
  exit 2
}

$commit = ((Get-Content $StampFile -Raw).Trim() -split '\s+')[0]
Write-Host "Deployed commit: $commit"

Push-Location $RepoRoot
try {
  git cat-file -e "${commit}^{commit}" 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit $commit is not in this repository - cannot verify." -ForegroundColor Yellow
    exit 2
  }

  # Files the commit tracks, so an install-only file is not mistaken for drift.
  $tracked = @{}
  git ls-tree -r --name-only $commit | ForEach-Object { $tracked[$_] = $true }

  $differs   = New-Object System.Collections.ArrayList
  $installOnly = 0
  $checked   = 0

  foreach ($pkg in @('openviking', 'openviking_cli')) {
    $root = Join-Path $SitePkgs $pkg
    if (-not (Test-Path $root)) { continue }

    Get-ChildItem $root -Recurse -Filter *.py -File |
      Where-Object { $_.FullName -notlike '*__pycache__*' } |
      ForEach-Object {
        $rel = $_.FullName.Substring($SitePkgs.Length).TrimStart('\', '/').Replace('\', '/')
        if (-not $tracked.ContainsKey($rel)) { $installOnly++; return }

        $checked++
        $blob = (git show "${commit}:$rel" 2>$null) -join "`n"
        $disk = Get-Content $_.FullName -Raw
        if ($null -eq $disk) { $disk = '' }

        # Compare on content, ignoring line-ending and trailing-whitespace noise:
        # git normalises CRLF on checkout, so a byte comparison reports every file.
        $a = ($disk -replace "`r`n", "`n").TrimEnd()
        $b = ($blob -replace "`r`n", "`n").TrimEnd()
        if ($a -ne $b) { [void]$differs.Add($rel) }
      }
  }
}
finally { Pop-Location }

Write-Host "Compared $checked tracked module(s); $installOnly install-only file(s) ignored."

if ($differs.Count -eq 0) {
  Write-Host "IN SYNC: every tracked module matches $commit." -ForegroundColor Green
  exit 0
}

Write-Host "DRIFT: $($differs.Count) module(s) differ from $commit." -ForegroundColor Red
$show = if ($Detailed) { $differs } else { $differs | Select-Object -First 20 }
$show | ForEach-Object { Write-Host "  $_" }
if (-not $Detailed -and $differs.Count -gt 20) {
  Write-Host "  ... and $($differs.Count - 20) more (use -Detailed)"
}
Write-Host ""
Write-Host "Re-run deploy/deploy.ps1 to restore a known state."
exit 1
