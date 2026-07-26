<#
.SYNOPSIS
  Report whether the running OpenViking install matches a known fork commit.

.DESCRIPTION
  The server runs from an installed package, not from the repo. Nothing
  normally tells you when those two diverge, and a hand-copied file drifts
  silently until something fails hours later inside a background commit.

  This compares every .py file in the installed package against the tree of the
  commit recorded at deploy time, and exits non-zero if they differ.

.EXAMPLE
  pwsh deploy/check-drift.ps1
  pwsh deploy/check-drift.ps1 -Verbose   # list every differing file
#>
[CmdletBinding()]
param(
  [string]$RepoRoot  = (Resolve-Path "$PSScriptRoot/..").Path,
  [string]$SitePkgs  = "$env:APPDATA/uv/tools/openviking/Lib/site-packages",
  [string]$StampFile = "$env:USERPROFILE/.openviking/.deployed-commit"
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $StampFile)) {
  Write-Host "NO STAMP: $StampFile is missing." -ForegroundColor Yellow
  Write-Host "The running install was not produced by deploy.ps1, so its provenance is unknown."
  exit 2
}

$stamp  = (Get-Content $StampFile -Raw).Trim()
$commit = ($stamp -split '\s+')[0]
Write-Host "Deployed commit: $commit"

Push-Location $RepoRoot
try {
  if (-not (git cat-file -e "$commit^{commit}" 2>$null; $?)) {
    Write-Host "Commit $commit is not in this repository - cannot verify." -ForegroundColor Yellow
    exit 2
  }

  $drift = @()
  foreach ($pkg in @('openviking', 'openviking_cli')) {
    $root = Join-Path $SitePkgs $pkg
    if (-not (Test-Path $root)) { continue }
    Get-ChildItem $root -Recurse -Filter *.py -File |
      Where-Object { $_.FullName -notmatch '__pycache__' } | ForEach-Object {
        $rel = $_.FullName.Substring($SitePkgs.Length).TrimStart('\','/') -replace '\','/'
        $tracked = git show "${commit}:$rel" 2>$null
        if ($LASTEXITCODE -ne 0) {
          # Present in the install but absent from the commit. Expected for
          # files the wheel ships that the repo does not track.
          $drift += [pscustomobject]@{ File = $rel; Kind = 'install-only' }
        }
        else {
          $installed = (Get-Content $_.FullName -Raw) -replace "`r`n", "`n"
          $expected  = ($tracked -join "`n") -replace "`r`n", "`n"
          if ($installed.TrimEnd() -ne $expected.TrimEnd()) {
            $drift += [pscustomobject]@{ File = $rel; Kind = 'differs' }
          }
        }
      }
  }
}
finally { Pop-Location }

$differs = @($drift | Where-Object Kind -eq 'differs')
$only    = @($drift | Where-Object Kind -eq 'install-only')

if ($differs.Count -eq 0) {
  Write-Host "IN SYNC: every tracked module matches $commit." -ForegroundColor Green
  if ($only.Count) { Write-Host "  ($($only.Count) install-only files, expected: shipped by the wheel, not tracked)" }
  exit 0
}

Write-Host "DRIFT: $($differs.Count) module(s) differ from $commit." -ForegroundColor Red
$differs | Select-Object -First 20 | ForEach-Object { Write-Host "  $($_.File)" }
if ($differs.Count -gt 20) { Write-Host "  ... and $($differs.Count - 20) more" }
Write-Host "`nRe-run deploy/deploy.ps1 to restore a known state."
exit 1
