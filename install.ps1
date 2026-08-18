# Install pearld, prlctl, oyster, and oystercli from Pearl GitHub Releases (Windows).
# Prefer: download, inspect, then run.
#   irm https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.ps1 -OutFile install.ps1
#   pwsh -File .\install.ps1
# Convenience:
#   irm https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.ps1 | iex

#Requires -Version 5.1

<#
.SYNOPSIS
  Install Pearl release binaries and mainnet configs on Windows.

.PARAMETER Version
  Release tag to install (default: latest stable), e.g. v1.1.5

.PARAMETER BinDir
  Install directory (default: $env:LOCALAPPDATA\Pearl\bin)

.EXAMPLE
  .\install.ps1

.EXAMPLE
  .\install.ps1 -Version v1.1.5 -BinDir $env:USERPROFILE\bin
#>
[CmdletBinding()]
param(
	[string]$Version,
	[string]$BinDir,
	[switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Repo = 'pearl-research-labs/pearl'
$Binaries = @('pearld', 'prlctl', 'oyster')
# Present only in releases that ship the interactive wallet CLI; skipped
# gracefully when installing older versions.
$OptionalBinaries = @('oystercli')
$AllowedArchiveBins = $Binaries + $OptionalBinaries + 'prlmon'
$TempDir = $null
$RpcUser = $null
$RpcPass = $null

function Write-InstallLog([string]$Message) {
	Write-Host "pearl-install: $Message"
}

function Get-LocalAppData {
	$local = $env:LOCALAPPDATA
	if (-not $local) { $local = $env:APPDATA }
	if (-not $local) { throw 'cannot determine LOCALAPPDATA / APPDATA' }
	$local
}

# Match node/btcutil.AppDataDir(app, roaming=false): %LOCALAPPDATA%\AppName
function Get-AppConfigPath([string]$Name) {
	$app = $Name.Substring(0, 1).ToUpperInvariant() + $Name.Substring(1)
	Join-Path (Join-Path (Get-LocalAppData) $app) "$Name.conf"
}

function Test-ReleaseVersion([string]$V) {
	$V -match '^v[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9._-]*$' -and $V -notmatch '[/\\]|\.\.'
}

function Get-WindowsArch {
	$arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
	switch ($arch.ToUpperInvariant()) {
		'AMD64' { 'amd64' }
		'ARM64' { throw 'unsupported architecture: ARM64 (Windows releases are amd64 only)' }
		default { throw "unsupported architecture: $arch (supported: amd64)" }
	}
}

function Save-HttpsFile([string]$Uri, [string]$OutFile) {
	if ($Uri -notmatch '^https://') { throw "refusing non-HTTPS download URL: $Uri" }
	Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
}

function Install-FileAtomic([string]$Source, [string]$Destination) {
	$dir = Split-Path -Parent $Destination
	if ($dir) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
	$tmp = "$Destination.tmp.$PID"
	Copy-Item -LiteralPath $Source -Destination $tmp -Force
	Move-Item -LiteralPath $tmp -Destination $Destination -Force
}

function Write-FileAtomic([string]$Destination, [string]$Content) {
	$dir = Split-Path -Parent $Destination
	if ($dir) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
	$tmp = "$Destination.tmp.$PID"
	[IO.File]::WriteAllText($tmp, $Content)
	Move-Item -LiteralPath $tmp -Destination $Destination -Force
}

function New-RpcSecret {
	$bytes = [byte[]]::new(24)
	$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
	try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
	[Convert]::ToBase64String($bytes).TrimEnd('=')
}

function Get-ConfValue([string]$Key, [string]$File) {
	if (-not (Test-Path -LiteralPath $File)) { return $null }
	foreach ($line in Get-Content -LiteralPath $File) {
		if ($line -match "^$([regex]::Escape($Key))=(.+)$") { return $Matches[1] }
	}
	$null
}

function Test-ConfigHasSecrets([string]$Name, [string]$File) {
	$keys = switch ($Name) {
		'oyster' { @('username', 'password') }
		{ $_ -in 'pearld', 'prlctl' } { @('rpcuser', 'rpcpass') }
		default { return $false }
	}
	(Get-ConfValue $keys[0] $File) -and (Get-ConfValue $keys[1] $File)
}

function Get-ConfigBody([string]$Name) {
	switch ($Name) {
		'pearld' {
			@"
[Application Options]

; Default mainnet configuration for pearld.
; RPC is bound to localhost only. TLS remains enabled by default.
; P2P still listens on all interfaces so the node can sync with the network.
; txindex helps prlctl / explorers; oyster defaults to SPV and does not require it.

rpcuser=$RpcUser
rpcpass=$RpcPass
rpclisten=127.0.0.1:44107
rpclisten=[::1]:44107
txindex=1
"@
		}
		'oyster' {
			@"
[Application Options]

; Default mainnet configuration for oyster.
; Syncs via SPV (neutrino) by default — no local pearld required for chain data.
; Wallet RPC is bound to localhost only. TLS remains enabled by default.
; username/password authenticate wallet RPC (prlctl --wallet) and optional pearld RPC.

usespv=1
username=$RpcUser
password=$RpcPass
pearldusername=$RpcUser
pearldpassword=$RpcPass
rpcconnect=127.0.0.1:44107
rpclisten=127.0.0.1:44207
rpclisten=[::1]:44207
"@
		}
		'prlctl' {
			@"
; Default mainnet configuration for prlctl.
; Shared credentials work for both:
;   prlctl getinfo           -> local pearld (port 44107)
;   prlctl --wallet getinfo  -> local oyster (port 44207)
; TLS cert defaults: pearld's rpc.cert, or oyster's when --wallet is set.

rpcuser=$RpcUser
rpcpass=$RpcPass
rpcserver=127.0.0.1
"@
		}
		default { throw "unknown config: $Name" }
	}
}

function Initialize-RpcCredentials {
	foreach ($probe in @(
			@{ Path = (Get-AppConfigPath 'pearld'); User = 'rpcuser'; Pass = 'rpcpass' }
			@{ Path = (Get-AppConfigPath 'oyster'); User = 'username'; Pass = 'password' }
			@{ Path = (Get-AppConfigPath 'prlctl'); User = 'rpcuser'; Pass = 'rpcpass' }
		)) {
		$u = Get-ConfValue $probe.User $probe.Path
		$p = Get-ConfValue $probe.Pass $probe.Path
		if ($u -and $p) {
			$script:RpcUser = $u
			$script:RpcPass = $p
			return
		}
	}
	$script:RpcUser = New-RpcSecret
	$script:RpcPass = New-RpcSecret
}

try {
	if ($Help) {
		@'
Install Pearl release binaries (pearld, prlctl, oyster) and mainnet configs.

Usage:
  install.ps1 [-Version vX.Y.Z] [-BinDir PATH]

Examples:
  .\install.ps1
  .\install.ps1 -Version v1.1.5
  .\install.ps1 -BinDir "$env:USERPROFILE\bin"
'@
		exit 0
	}

	if ($env:OS -ne 'Windows_NT') {
		throw 'this installer is for Windows; use install.sh on macOS/Linux'
	}

	if (-not $BinDir) { $BinDir = Join-Path (Get-LocalAppData) 'Pearl\bin' }
	try {
		New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
		$probe = Join-Path $BinDir ".pearl-install-write-test.$PID"
		[IO.File]::WriteAllText($probe, 'ok')
		Remove-Item -LiteralPath $probe -Force
	} catch {
		throw "install directory is not writable: $BinDir`ntry: .\install.ps1 -BinDir `"$env:LOCALAPPDATA\Pearl\bin`""
	}

	if (-not $Version) {
		Write-InstallLog 'resolving latest release...'
		$Version = [string](Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest").tag_name
	}
	if (-not (Test-ReleaseVersion $Version)) {
		throw "invalid version '$Version' (expected vX.Y.Z)"
	}

	$arch = Get-WindowsArch
	$archiveName = "pearl-windows-$arch-$Version.zip"
	$assetBase = "https://github.com/$Repo/releases/download/$Version"

	$TempDir = Join-Path ([IO.Path]::GetTempPath()) ("pearl-install." + [guid]::NewGuid().ToString('N'))
	$extract = Join-Path $TempDir 'extract'
	New-Item -ItemType Directory -Path $extract -Force | Out-Null

	$archivePath = Join-Path $TempDir $archiveName
	$checksumsPath = Join-Path $TempDir 'checksums.txt'

	Write-InstallLog "downloading $archiveName"
	Save-HttpsFile "$assetBase/$archiveName" $archivePath
	Write-InstallLog 'downloading checksums.txt'
	Save-HttpsFile "$assetBase/checksums.txt" $checksumsPath

	Write-InstallLog 'verifying SHA-256...'
	$expected = $null
	foreach ($line in Get-Content -LiteralPath $checksumsPath) {
		if ($line -match '^\s*([0-9a-fA-F]{64})\s+(\S+)\s*$' -and $Matches[2] -eq $archiveName) {
			$expected = $Matches[1].ToLowerInvariant()
			break
		}
	}
	if (-not $expected) { throw "no checksum entry for $archiveName in checksums.txt" }
	$actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
	if ($actual -ne $expected) {
		throw "checksum mismatch for $archiveName`n  expected: $expected`n  actual:   $actual"
	}

	Write-InstallLog 'extracting binaries...'
	Expand-Archive -LiteralPath $archivePath -DestinationPath $extract -Force
	foreach ($item in Get-ChildItem -LiteralPath $extract -Force) {
		if ($item.PSIsContainer) { throw "refusing nested archive path: $($item.Name)" }
		$base = [IO.Path]::GetFileNameWithoutExtension($item.Name)
		if ($item.Extension -ne '.exe' -or $base -notin $AllowedArchiveBins) {
			throw "unexpected archive member: $($item.Name)"
		}
	}

	foreach ($bin in $Binaries) {
		$src = Join-Path $extract "$bin.exe"
		if (-not (Test-Path -LiteralPath $src)) { throw "archive is missing $bin.exe" }
		if ((Get-Item -LiteralPath $src).Attributes -band [IO.FileAttributes]::ReparsePoint) {
			throw "refusing symlink/reparse point in archive: $bin.exe"
		}
		$dest = Join-Path $BinDir "$bin.exe"
		Write-InstallLog "installing binary -> $dest"
		Install-FileAtomic $src $dest
	}
	foreach ($bin in $OptionalBinaries) {
		$src = Join-Path $extract "$bin.exe"
		if (-not (Test-Path -LiteralPath $src)) {
			Write-InstallLog "$bin.exe is not part of $Version; skipping"
			continue
		}
		if ((Get-Item -LiteralPath $src).Attributes -band [IO.FileAttributes]::ReparsePoint) {
			throw "refusing symlink/reparse point in archive: $bin.exe"
		}
		$dest = Join-Path $BinDir "$bin.exe"
		Write-InstallLog "installing binary -> $dest"
		Install-FileAtomic $src $dest
	}

	Initialize-RpcCredentials

	$created = $false
	foreach ($name in @('pearld', 'oyster', 'prlctl')) {
		$dest = Get-AppConfigPath $name
		if (Test-ConfigHasSecrets $name $dest) {
			Write-InstallLog "kept existing config -> $dest"
			continue
		}
		Write-FileAtomic $dest (Get-ConfigBody $name)
		Write-InstallLog "wrote config -> $dest"
		$created = $true
	}
	if ($created) {
		Write-InstallLog 'RPC username/password were auto-generated; no -u/-P flags needed'
	}

	$installedBins = $Binaries + ($OptionalBinaries | Where-Object { Test-Path -LiteralPath (Join-Path $BinDir "$_.exe") })
	$binLines = ($installedBins | ForEach-Object { "  $(Join-Path $BinDir "$_.exe")" }) -join "`n"
	$configLines = (@('pearld', 'oyster', 'prlctl') | ForEach-Object { "  $(Get-AppConfigPath $_)" }) -join "`n"
	$oystercliLine = ''
	if (Test-Path -LiteralPath (Join-Path $BinDir 'oystercli.exe')) {
		$oystercliLine = "`n  oystercli              # interactive wallet UI"
	}
	@"

pearl-install: done ($Version)

Binaries:
$binLines

Configs (OS default locations, shared RPC credentials):
$configLines

Next steps (no -u/-P/-C needed):
  pearld
  oyster --create
  oyster                 # SPV sync by default
  prlctl getinfo
  prlctl --wallet getinfo$oystercliLine
"@ | Write-Host

	$normalizedBin = [IO.Path]::GetFullPath($BinDir).TrimEnd('\')
	$onPath = $false
	foreach ($entry in ($env:PATH -split ';')) {
		if (-not $entry) { continue }
		try {
			if ([IO.Path]::GetFullPath($entry).TrimEnd('\') -eq $normalizedBin) {
				$onPath = $true
				break
			}
		} catch { }
	}
	if (-not $onPath) {
		Write-InstallLog "note: $BinDir is not on your PATH"
		Write-Host "  add it for this session: `$env:PATH = `"$BinDir;`$env:PATH`""
		Write-Host '  or set a permanent User PATH entry in System Properties'
	}
} catch {
	[Console]::Error.WriteLine("pearl-install: $($_.Exception.Message)")
	exit 1
} finally {
	if ($TempDir -and (Test-Path -LiteralPath $TempDir)) {
		Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
	}
}
