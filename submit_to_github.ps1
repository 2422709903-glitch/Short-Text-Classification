param(
    [string]$Message = ""
)

$ErrorActionPreference = 'Stop'
$BranchName = 'codex-project-upload'

function Invoke-Git {
    param([string[]]$Arguments)
    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($Arguments -join ' ')"
    }
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

if (-not (Test-Path (Join-Path $RepoRoot '.git'))) {
    throw 'This folder is not a Git repository.'
}

$CurrentBranch = (& git branch --show-current).Trim()
if ($CurrentBranch -ne $BranchName) {
    throw "Current branch is '$CurrentBranch'. Expected '$BranchName'."
}

Write-Host 'Fetching the GitHub branch. If GitHub asks you to sign in, finish the browser sign-in...'
$env:GIT_TERMINAL_PROMPT = '1'
$env:GIT_HTTP_LOW_SPEED_LIMIT = '1000'
$env:GIT_HTTP_LOW_SPEED_TIME = '30'
try {
    Invoke-Git -Arguments @('-c', 'credential.interactive=always', 'fetch', '--progress', '--verbose', 'origin', $BranchName)
}
catch {
    Write-Host 'Could not reach GitHub. Check your network or Git proxy settings, then run the script again.' -ForegroundColor Red
    exit 1
}

$RemoteRef = "origin/$BranchName"
$MergeBase = & git merge-base HEAD $RemoteRef 2>$null
if ($LASTEXITCODE -ne 0 -or -not $MergeBase) {
    $BackupBranch = "backup-before-upload-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Invoke-Git -Arguments @('branch', $BackupBranch, 'HEAD')
    Write-Host "Local history differs from GitHub. Backup created: $BackupBranch"
    # --mixed updates the branch and index while keeping working files unchanged.
    Invoke-Git -Arguments @('reset', '--mixed', $RemoteRef)
}

Invoke-Git -Arguments @('add', '-A', '--', '.', ':(exclude)1.data/**', ':(exclude).gitmodules')
$StagedChanges = & git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host 'No changes to commit.'
    exit 0
}

Invoke-Git -Arguments @('diff', '--cached', '--check')
if ([string]::IsNullOrWhiteSpace($Message)) {
    $Message = Read-Host 'Enter a commit message'
    if ([string]::IsNullOrWhiteSpace($Message)) {
        $Message = "Update project $(Get-Date -Format 'yyyy-MM-dd-HH-mm')"
    }
}
Invoke-Git -Arguments @('commit', '-m', $Message)
Invoke-Git -Arguments @('push', 'origin', "HEAD:$BranchName")

Write-Host "Committed and pushed to GitHub branch $BranchName."

