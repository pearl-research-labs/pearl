# Install pearld, prlctl, and oyster from Pearl GitHub Releases (Windows).
# Prefer: download, inspect, then run.
#   irm https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.ps1 -OutFile install.ps1
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
# Convenience:
#   irm https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.ps1 | iex

[CmdletBinding()]
param(
	[string]$Version = "",
	[string]$BinDir = "",
	[switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Repo = "pearl-research-labs/pearl"
$GitHubBase = "https://github.com/$Repo"
$ReleaseBase = "$GitHubBase/releases"
$Binaries = @("pearld", "prlctl", "oyster")
$Configs = @("pearld", "oyster", "prlctl")

$script:RpcUser = ""
$script:RpcPass = ""
$script:TempDir = $null

function Write-Info {
	param([string]$Message)
	Write-Host "pearl-install: $Message"
}

function Die {
	param([string]$Message)
	Write-Error "pearl-install: $Message"
	exit 1
}

function Show-Usage {
	@"
Install Pearl release binaries (pearld, prlctl, oyster) and mainnet configs.

Usage:
  install.ps1 [-Version vX.Y.Z] [-BinDir PATH]

Parameters:
  -Version vX.Y.Z   Install a specific release (default: latest stable)
  -BinDir PATH      Install directory (default: `$env:LOCALAPPDATA\Pearl\bin)
  -Help             Show this help

Examples:
  .\install.ps1
  .\install.ps1 -Version v0.1.0
  .\install.ps1 -BinDir "`$env:USERPROFILE\bin"
"@
}

function Test-Version {
	param([string]$V)
	if ($V -notmatch '^v[0-9]+\.[0-9]+\.[0-9]+([A-Za-z0-9._-]*)$') {
		return $false
	}
	if ($V -match '[/\\]' -or $V -match '\.\.') {
		return $false
	}
	return $true
}

function Get-PearlArch {
	$arch = $env:PROCESSOR_ARCHITECTURE
	if ($env:PROCESSOR_ARCHITEW6432) {
		$arch = $env:PROCESSOR_ARCHITEW6432
	}
	switch ($arch.ToUpperInvariant()) {
		"AMD64" { return "amd64" }
		"ARM64" {
			Die "unsupported architecture: ARM64 (Windows releases are amd64 only)"
		}
		default {
			Die "unsupported architecture: $arch (supported: amd64)"
		}
	}
}

# Match node/btcutil.AppDataDir(app, roaming=false): %LOCALAPPDATA%\AppName
function Get-AppDataDir {
	param([string]$App)
	$local = $env:LOCALAPPDATA
	if (-not $local) {
		$local = $env:APPDATA
	}
	if (-not $local) {
		Die "cannot determine LOCALAPPDATA / APPDATA"
	}
	$upper = $App.Substring(0, 1).ToUpperInvariant() + $App.Substring(1)
	return (Join-Path $local $upper)
}

function Get-ConfigPath {
	param([string]$Name)
	return (Join-Path (Get-AppDataDir $Name) "$Name.conf")
}

function Get-DefaultBinDir {
	$local = $env:LOCALAPPDATA
	if (-not $local) {
		Die "LOCALAPPDATA is not set"
	}
	return (Join-Path $local "Pearl\bin")
}

function Ensure-BinDir {
	param([string]$Dir)
	if (-not (Test-Path -LiteralPath $Dir)) {
		try {
			New-Item -ItemType Directory -Path $Dir -Force | Out-Null
		} catch {
			Die "cannot create install directory: $Dir`ntry: .\install.ps1 -BinDir `"$env:LOCALAPPDATA\Pearl\bin`""
		}
	}
	try {
		$probe = Join-Path $Dir ".pearl-install-write-test"
		[IO.File]::WriteAllText($probe, "ok")
		Remove-Item -LiteralPath $probe -Force
	} catch {
		Die "install directory is not writable: $Dir`ntry: .\install.ps1 -BinDir `"$env:LOCALAPPDATA\Pearl\bin`""
	}
}

function Get-Https {
	param(
		[string]$Url,
		[string]$Dest
	)
	if ($Url -notmatch '^https://') {
		Die "refusing non-HTTPS download URL: $Url"
	}
	Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
}

function Resolve-LatestVersion {
	$rel = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
	$tag = [string]$rel.tag_name
	if (-not (Test-Version $tag)) {
		Die "could not parse latest release version from API: $tag"
	}
	return $tag
}

function Assert-Checksum {
	param(
		[string]$Archive,
		[string]$ChecksumsFile
	)
	$name = Split-Path -Leaf $Archive
	$expected = $null
	Get-Content -LiteralPath $ChecksumsFile | ForEach-Object {
		$line = $_.Trim()
		if (-not $line) { return }
		$parts = $line -split '\s+', 2
		if ($parts.Count -eq 2 -and $parts[1] -eq $name) {
			$expected = $parts[0].ToLowerInvariant()
		}
	}
	if (-not $expected) {
		Die "no checksum entry for $name in checksums.txt"
	}
	if ($expected -notmatch '^[0-9a-f]{64}$') {
		Die "malformed checksum for $name"
	}
	$actual = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
	if ($actual -ne $expected) {
		Die "checksum mismatch for $name`n  expected: $expected`n  actual:   $actual"
	}
}

function Expand-PearlArchive {
	param(
		[string]$Archive,
		[string]$Staging
	)
	New-Item -ItemType Directory -Path $Staging -Force | Out-Null
	$extract = Join-Path $Staging "_extract"
	New-Item -ItemType Directory -Path $extract -Force | Out-Null
	Expand-Archive -LiteralPath $Archive -DestinationPath $extract -Force

	Get-ChildItem -LiteralPath $extract -Force | ForEach-Object {
		$name = $_.Name
		$base = [IO.Path]::GetFileNameWithoutExtension($name)
		$ext = [IO.Path]::GetExtension($name).ToLowerInvariant()
		if ($_.PSIsContainer) {
			Die "refusing nested archive path: $name"
		}
		if ($ext -ne ".exe") {
			Die "unexpected archive member: $name"
		}
		switch ($base) {
			{ $_ -in @("pearld", "prlctl", "oyster", "prlmon") } { }
			default { Die "unexpected archive member: $name" }
		}
	}

	foreach ($bin in $Binaries) {
		$src = Join-Path $extract "$bin.exe"
		if (-not (Test-Path -LiteralPath $src)) {
			Die "archive is missing $bin.exe"
		}
		if ((Get-Item -LiteralPath $src).Attributes -band [IO.FileAttributes]::ReparsePoint) {
			Die "refusing symlink/reparse point in archive: $bin.exe"
		}
		Copy-Item -LiteralPath $src -Destination (Join-Path $Staging "$bin.exe") -Force
	}
}

function Install-Binary {
	param(
		[string]$Src,
		[string]$DestDir
	)
	$name = Split-Path -Leaf $Src
	$tmp = Join-Path $DestDir ".$name.tmp.$PID"
	$dest = Join-Path $DestDir $name
	Copy-Item -LiteralPath $Src -Destination $tmp -Force
	Move-Item -LiteralPath $tmp -Destination $dest -Force
}

function New-Secret {
	$bytes = New-Object byte[] 24
	$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
	try {
		$rng.GetBytes($bytes)
	} finally {
		$rng.Dispose()
	}
	return [Convert]::ToBase64String($bytes).TrimEnd("=")
}

function Write-AtomicFile {
	param(
		[string]$Dest,
		[string]$Content
	)
	$dir = Split-Path -Parent $Dest
	if (-not (Test-Path -LiteralPath $dir)) {
		New-Item -ItemType Directory -Path $dir -Force | Out-Null
	}
	$tmp = "$Dest.tmp.$PID"
	[IO.File]::WriteAllText($tmp, $Content)
	Move-Item -LiteralPath $tmp -Destination $Dest -Force
}

function Get-ConfigText {
	param([string]$Name)
	switch ($Name) {
		"pearld" {
			return @"
[Application Options]

; Default mainnet configuration for pearld.
; RPC is bound to localhost only. TLS remains enabled by default.
; P2P still listens on all interfaces so the node can sync with the network.
; txindex helps prlctl / explorers; oyster defaults to SPV and does not require it.

rpcuser=$($script:RpcUser)
rpcpass=$($script:RpcPass)
rpclisten=127.0.0.1:44107
rpclisten=[::1]:44107
txindex=1
"@
		}
		"oyster" {
			return @"
[Application Options]

; Default mainnet configuration for oyster.
; Syncs via SPV (neutrino) by default — no local pearld required for chain data.
; Wallet RPC is bound to localhost only. TLS remains enabled by default.
; username/password authenticate wallet RPC (prlctl --wallet) and optional pearld RPC.

usespv=1
username=$($script:RpcUser)
password=$($script:RpcPass)
pearldusername=$($script:RpcUser)
pearldpassword=$($script:RpcPass)
rpcconnect=127.0.0.1:44107
rpclisten=127.0.0.1:44207
rpclisten=[::1]:44207
"@
		}
		"prlctl" {
			return @"
; Default mainnet configuration for prlctl.
; Shared credentials work for both:
;   prlctl getinfo           -> local pearld (port 44107)
;   prlctl --wallet getinfo  -> local oyster (port 44207)
; TLS cert defaults: pearld's rpc.cert, or oyster's when --wallet is set.

rpcuser=$($script:RpcUser)
rpcpass=$($script:RpcPass)
rpcserver=127.0.0.1
"@
		}
		default { Die "unknown config: $Name" }
	}
}

function Get-ConfValue {
	param(
		[string]$Key,
		[string]$File
	)
	if (-not (Test-Path -LiteralPath $File)) {
		return ""
	}
	foreach ($line in Get-Content -LiteralPath $File) {
		if ($line -match "^$([regex]::Escape($Key))=(.*)$") {
			$val = $Matches[1]
			if ($val -ne "") {
				return $val
			}
		}
	}
	return ""
}

function Test-ConfigHasSecrets {
	param(
		[string]$Name,
		[string]$File
	)
	if (-not (Test-Path -LiteralPath $File)) {
		return $false
	}
	switch ($Name) {
		{ $_ -in @("pearld", "prlctl") } {
			$user = Get-ConfValue "rpcuser" $File
			$pass = Get-ConfValue "rpcpass" $File
		}
		"oyster" {
			$user = Get-ConfValue "username" $File
			$pass = Get-ConfValue "password" $File
		}
		default { return $false }
	}
	return ($user -ne "" -and $pass -ne "")
}

function Ensure-RpcCredentials {
	if ($script:RpcUser -and $script:RpcPass) {
		return
	}

	foreach ($pair in @(
			@{ File = (Get-ConfigPath "pearld"); UserKey = "rpcuser"; PassKey = "rpcpass" },
			@{ File = (Get-ConfigPath "oyster"); UserKey = "username"; PassKey = "password" },
			@{ File = (Get-ConfigPath "prlctl"); UserKey = "rpcuser"; PassKey = "rpcpass" }
		)) {
		$u = Get-ConfValue $pair.UserKey $pair.File
		$p = Get-ConfValue $pair.PassKey $pair.File
		if ($u -and $p) {
			$script:RpcUser = $u
			$script:RpcPass = $p
			return
		}
	}

	$script:RpcUser = New-Secret
	$script:RpcPass = New-Secret
}

function Install-DefaultConfigs {
	param([string]$InstallBinDir)
	$created = $false
	Ensure-RpcCredentials

	$sample = Join-Path $InstallBinDir "sample-pearld.conf"
	Write-AtomicFile $sample (Get-ConfigText "pearld")
	Write-Info "default config template -> $sample"

	foreach ($name in $Configs) {
		$dest = Get-ConfigPath $name
		if (Test-ConfigHasSecrets $name $dest) {
			Write-Info "kept existing config -> $dest"
			continue
		}
		Write-AtomicFile $dest (Get-ConfigText $name)
		Write-Info "wrote config -> $dest"
		$created = $true
	}

	if ($created) {
		Write-Info "RPC username/password were auto-generated; no -u/-P flags needed"
	}
}

function Test-PathContains {
	param([string]$Dir)
	$parts = ($env:PATH -split ";") | Where-Object { $_ -ne "" }
	foreach ($p in $parts) {
		try {
			if ([IO.Path]::GetFullPath($p).TrimEnd("\") -eq [IO.Path]::GetFullPath($Dir).TrimEnd("\")) {
				return $true
			}
		} catch {
			# ignore invalid PATH entries
		}
	}
	return $false
}

function Show-Summary {
	param(
		[string]$Ver,
		[string]$InstallBinDir
	)
	Write-Host ""
	Write-Host "pearl-install: done ($Ver)"
	Write-Host ""
	Write-Host "Binaries:"
	Write-Host "  $(Join-Path $InstallBinDir 'pearld.exe')"
	Write-Host "  $(Join-Path $InstallBinDir 'prlctl.exe')"
	Write-Host "  $(Join-Path $InstallBinDir 'oyster.exe')"
	Write-Host "  $(Join-Path $InstallBinDir 'sample-pearld.conf')"
	Write-Host ""
	Write-Host "Configs (OS default locations, shared RPC credentials):"
	Write-Host "  $(Get-ConfigPath 'pearld')"
	Write-Host "  $(Get-ConfigPath 'oyster')"
	Write-Host "  $(Get-ConfigPath 'prlctl')"
	Write-Host ""
	Write-Host "Next steps (no -u/-P/-C needed):"
	Write-Host "  pearld"
	Write-Host "  oyster --create"
	Write-Host "  oyster                 # SPV sync by default"
	Write-Host "  prlctl getinfo"
	Write-Host "  prlctl --wallet getinfo"

	if (-not (Test-PathContains $InstallBinDir)) {
		Write-Host ""
		Write-Info "note: $InstallBinDir is not on your PATH"
		Write-Host "  add it for this session: `$env:PATH = `"$InstallBinDir;`$env:PATH`""
		Write-Host "  or set a permanent User PATH entry in System Properties"
	}
}

function Clear-TempDir {
	if ($script:TempDir -and (Test-Path -LiteralPath $script:TempDir)) {
		Remove-Item -LiteralPath $script:TempDir -Recurse -Force -ErrorAction SilentlyContinue
	}
}

try {
	if ($Help) {
		Show-Usage
		exit 0
	}

	if ($env:OS -ne "Windows_NT") {
		Die "this installer is for Windows; use install.sh on macOS/Linux"
	}

	$arch = Get-PearlArch
	if (-not $BinDir) {
		$BinDir = Get-DefaultBinDir
	}
	Ensure-BinDir $BinDir

	if (-not $Version) {
		Write-Info "resolving latest release..."
		$Version = Resolve-LatestVersion
	}
	if (-not (Test-Version $Version)) {
		Die "invalid version '$Version' (expected vX.Y.Z)"
	}

	$archiveName = "pearl-windows-$arch-$Version.zip"
	$assetBase = "$ReleaseBase/download/$Version"

	$script:TempDir = Join-Path ([IO.Path]::GetTempPath()) ("pearl-install." + [guid]::NewGuid().ToString("N"))
	New-Item -ItemType Directory -Path $script:TempDir -Force | Out-Null

	$archivePath = Join-Path $script:TempDir $archiveName
	$checksumsPath = Join-Path $script:TempDir "checksums.txt"
	$staging = Join-Path $script:TempDir "staging"

	Write-Info "downloading $archiveName"
	Get-Https "$assetBase/$archiveName" $archivePath
	Write-Info "downloading checksums.txt"
	Get-Https "$assetBase/checksums.txt" $checksumsPath

	Write-Info "verifying SHA-256..."
	Assert-Checksum $archivePath $checksumsPath

	Write-Info "extracting binaries..."
	Expand-PearlArchive $archivePath $staging
	foreach ($bin in $Binaries) {
		$destName = "$bin.exe"
		Write-Info "installing binary -> $(Join-Path $BinDir $destName)"
		Install-Binary (Join-Path $staging $destName) $BinDir
	}

	Install-DefaultConfigs $BinDir
	Show-Summary $Version $BinDir
} finally {
	Clear-TempDir
}
