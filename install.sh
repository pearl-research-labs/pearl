#!/bin/sh
# Install pearld, prlctl, oyster, and oystercli from Pearl GitHub Releases.
# Usage: curl -fsSL https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.sh | sh
# Prefer: download, inspect, then run with sh.

set -eu

REPO="pearl-research-labs/pearl"
GITHUB_BASE="https://github.com/${REPO}"
RELEASE_BASE="${GITHUB_BASE}/releases"
BINARIES="pearld prlctl oyster"
# Present only in releases that ship the interactive wallet CLI; skipped
# gracefully when installing older versions.
OPTIONAL_BINARIES="oystercli"
CONFIGS="pearld oyster prlctl"

PRINT_HELP=0
VERSION=""
BIN_DIR=""
TMP_DIR=""
OS=""
RPC_USER=""
RPC_PASS=""

usage() {
	cat << 'EOF'
Install Pearl release binaries (pearld, prlctl, oyster, oystercli) and mainnet configs.

Usage:
  install.sh [options]

Options:
  --version vX.Y.Z   Install a specific release (default: latest stable)
  --bin-dir PATH     Install directory (default: ${XDG_BIN_HOME:-$HOME/.local/bin})
  -h, --help         Show this help

Examples:
  install.sh
  install.sh --version v0.1.0
  install.sh --bin-dir "$HOME/bin"
EOF
}

err() {
	printf 'pearl-install: %s\n' "$*" >&2
}

die() {
	err "$*"
	exit 1
}

info() {
	printf 'pearl-install: %s\n' "$*"
}

need_cmd() {
	command -v "$1" > /dev/null 2>&1 || die "required command not found: $1"
}

cleanup() {
	if [ -n "${TMP_DIR:-}" ] && [ -d "${TMP_DIR}" ]; then
		rm -rf "${TMP_DIR}"
	fi
}

parse_args() {
	while [ "$#" -gt 0 ]; do
		case "$1" in
			--version)
				[ "$#" -ge 2 ] || die "--version requires an argument"
				VERSION=$2
				shift 2
				;;
			--version=*)
				VERSION=${1#--version=}
				shift
				;;
			--bin-dir)
				[ "$#" -ge 2 ] || die "--bin-dir requires an argument"
				BIN_DIR=$2
				shift 2
				;;
			--bin-dir=*)
				BIN_DIR=${1#--bin-dir=}
				shift
				;;
			-h | --help)
				PRINT_HELP=1
				shift
				;;
			*)
				die "unknown argument: $1 (try --help)"
				;;
		esac
	done
}

validate_version() {
	case "$1" in
		v[0-9]*.[0-9]*.[0-9]*) ;;
		*) return 1 ;;
	esac
	case "$1" in
		*[!A-Za-z0-9._-]* | */* | *..*) return 1 ;;
	esac
}

detect_os() {
	case "$(uname -s)" in
		Linux) printf 'linux\n' ;;
		Darwin) printf 'darwin\n' ;;
		*) die "unsupported operating system: $(uname -s) (supported: Linux, macOS)" ;;
	esac
}

detect_arch() {
	# Prefer hardware arch so Rosetta does not select the wrong binary.
	if [ "$1" = "darwin" ] && [ "$(sysctl -n hw.optional.arm64 2> /dev/null || true)" = "1" ]; then
		printf 'arm64\n'
		return 0
	fi
	case "$(uname -m)" in
		x86_64 | amd64) printf 'amd64\n' ;;
		aarch64 | arm64) printf 'arm64\n' ;;
		*) die "unsupported architecture: $(uname -m) (supported: amd64, arm64)" ;;
	esac
}

# Home directory the Pearl binaries resolve (passwd entry), matching
# node/btcutil.AppDataDir. $HOME alone is not enough because the Go binaries
# prefer the passwd home over the HOME environment variable.
user_home() {
	if command -v python3 > /dev/null 2>&1; then
		python3 -c 'import os, pwd; print(pwd.getpwuid(os.getuid()).pw_dir)'
		return 0
	fi
	if command -v dscl > /dev/null 2>&1; then
		# macOS fallback when python3 is unavailable.
		_home=$(dscl . -read "/Users/$(id -un)" NFSHomeDirectory 2> /dev/null | awk '{print $2}')
		if [ -n "$_home" ]; then
			printf '%s\n' "$_home"
			return 0
		fi
	fi
	if [ -n "${HOME:-}" ]; then
		printf '%s\n' "$HOME"
		return 0
	fi
	die "cannot determine home directory"
}

# Match node/btcutil.AppDataDir for pearld / oyster / prlctl.
app_data_dir() {
	app=$1
	_home=$(user_home)
	case "$OS" in
		darwin)
			first=$(printf '%s' "$app" | cut -c1 | tr '[:lower:]' '[:upper:]')
			rest=$(printf '%s' "$app" | cut -c2-)
			printf '%s\n' "${_home}/Library/Application Support/${first}${rest}"
			;;
		*)
			printf '%s\n' "${_home}/.${app}"
			;;
	esac
}

config_path() {
	printf '%s/%s.conf\n' "$(app_data_dir "$1")" "$1"
}

default_bin_dir() {
	if [ -n "${XDG_BIN_HOME:-}" ]; then
		printf '%s\n' "$XDG_BIN_HOME"
	else
		printf '%s\n' "${HOME}/.local/bin"
	fi
}

ensure_bin_dir() {
	if [ ! -d "$1" ]; then
		mkdir -p "$1" 2> /dev/null || die "cannot create install directory: $1
try: install.sh --bin-dir \"\$HOME/.local/bin\""
	fi
	[ -w "$1" ] || die "install directory is not writable: $1
try: install.sh --bin-dir \"\$HOME/.local/bin\""
}

download() {
	_url=$1
	_dest=$2
	case "$_url" in
		https://*) ;;
		*) die "refusing non-HTTPS download URL: $_url" ;;
	esac

	if command -v curl > /dev/null 2>&1; then
		curl --fail --show-error --silent --location --proto '=https' --tlsv1.2 \
			--output "$_dest" "$_url" || die "download failed: $_url"
		return 0
	fi
	if command -v wget > /dev/null 2>&1; then
		wget --quiet --https-only --output-document="$_dest" "$_url" || die "download failed: $_url"
		return 0
	fi
	die "required command not found: curl or wget"
}

resolve_latest_version() {
	need_cmd curl
	url=$(curl --fail --silent --show-error --location --head \
		--proto '=https' --tlsv1.2 \
		--output /dev/null --write-out '%{url_effective}' \
		"${GITHUB_BASE}/releases/latest") \
		|| die "failed to resolve latest release from GitHub"
	version=${url##*/}
	validate_version "$version" || die "could not parse latest release version from: $url"
	printf '%s\n' "$version"
}

sha256_file() {
	if command -v sha256sum > /dev/null 2>&1; then
		sha256sum "$1" | awk '{print $1}'
		return 0
	fi
	if command -v shasum > /dev/null 2>&1; then
		shasum -a 256 "$1" | awk '{print $1}'
		return 0
	fi
	die "required command not found: sha256sum or shasum"
}

verify_checksum() {
	_archive=$1
	_checksums=$2
	_name=$(basename "$_archive")
	_expected=$(awk -v name="$_name" '
		$2 == name { print $1; found=1; exit }
		END { if (!found) exit 1 }
	' "$_checksums") || die "no checksum entry for ${_name} in checksums.txt"

	case "$_expected" in
		[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]*)
			[ "${#_expected}" -eq 64 ] || die "malformed checksum for ${_name}"
			;;
		*) die "malformed checksum for ${_name}" ;;
	esac

	_actual=$(sha256_file "$_archive")
	[ "$_actual" = "$_expected" ] || die "checksum mismatch for ${_name}
  expected: ${_expected}
  actual:   ${_actual}"
}

extract_binaries() {
	_archive=$1
	_staging=$2
	_list="${_staging}.members"
	mkdir -p "$_staging"
	tar -tzf "$_archive" > "$_list" || die "failed to list archive contents"

	while IFS= read -r _member; do
		[ -n "$_member" ] || continue
		_name=${_member#./}
		case "$_name" in
			'' | */) continue ;;
			*/*) die "refusing nested archive path: $_member" ;;
			pearld | prlctl | oyster | oystercli | prlmon) ;;
			*) die "unexpected archive member: $_member" ;;
		esac
	done < "$_list"

	if ! tar -xzf "$_archive" -C "$_staging" pearld prlctl oyster 2> /dev/null; then
		tar -xzf "$_archive" -C "$_staging" ./pearld ./prlctl ./oyster \
			|| die "archive is missing one or more of: ${BINARIES}"
	fi
	for _bin in $OPTIONAL_BINARIES; do
		tar -xzf "$_archive" -C "$_staging" "$_bin" 2> /dev/null \
			|| tar -xzf "$_archive" -C "$_staging" "./${_bin}" 2> /dev/null \
			|| true
	done

	for _bin in $BINARIES $OPTIONAL_BINARIES; do
		_path="${_staging}/${_bin}"
		if [ ! -e "$_path" ]; then
			case " ${OPTIONAL_BINARIES} " in
				*" ${_bin} "*) continue ;;
			esac
			die "missing ${_bin} after extraction"
		fi
		[ ! -L "$_path" ] || die "refusing symlink in archive: ${_bin}"
		[ -f "$_path" ] || die "refusing non-regular file: ${_bin}"
	done
}

install_binary() {
	_src=$1
	_dest_dir=$2
	_name=$(basename "$_src")
	_tmp="${_dest_dir}/.${_name}.tmp.$$"
	cp "$_src" "$_tmp"
	chmod 755 "$_tmp"
	mv -f "$_tmp" "${_dest_dir}/${_name}"
}

path_contains() {
	case ":${PATH}:" in
		*":$1:"*) return 0 ;;
		*) return 1 ;;
	esac
}

gen_secret() {
	if command -v openssl > /dev/null 2>&1; then
		openssl rand -base64 24 | tr -d '\n='
		return 0
	fi
	if command -v base64 > /dev/null 2>&1; then
		dd if=/dev/urandom bs=24 count=1 2> /dev/null | base64 | tr -d '\n='
		return 0
	fi
	die "required command not found: openssl or base64 (needed to generate RPC credentials)"
}

write_file() {
	_dest=$1
	_mode=$2
	_dir=$(dirname "$_dest")
	mkdir -p "$_dir"
	chmod 700 "$_dir" 2> /dev/null || true
	_tmp="${_dest}.tmp.$$"
	cat > "$_tmp"
	chmod "$_mode" "$_tmp"
	mv -f "$_tmp" "$_dest"
}

# Mainnet defaults. Uses RPC_USER / RPC_PASS (always set before calling).
write_config() {
	_name=$1
	_dest=$2
	_mode=$3
	case "$_name" in
		pearld)
			write_file "$_dest" "$_mode" << EOF
[Application Options]

; Default mainnet configuration for pearld.
; RPC is bound to localhost only. TLS remains enabled by default.
; P2P still listens on all interfaces so the node can sync with the network.
; txindex helps prlctl / explorers; oyster defaults to SPV and does not require it.

rpcuser=${RPC_USER}
rpcpass=${RPC_PASS}
rpclisten=127.0.0.1:44107
rpclisten=[::1]:44107
txindex=1
EOF
			;;
		oyster)
			write_file "$_dest" "$_mode" << EOF
[Application Options]

; Default mainnet configuration for oyster.
; Syncs via SPV (neutrino) by default — no local pearld required for chain data.
; Wallet RPC is bound to localhost only. TLS remains enabled by default.
; username/password authenticate wallet RPC (prlctl --wallet) and optional pearld RPC.

usespv=1
username=${RPC_USER}
password=${RPC_PASS}
pearldusername=${RPC_USER}
pearldpassword=${RPC_PASS}
rpcconnect=127.0.0.1:44107
rpclisten=127.0.0.1:44207
rpclisten=[::1]:44207
EOF
			;;
		prlctl)
			write_file "$_dest" "$_mode" << EOF
; Default mainnet configuration for prlctl.
; Shared credentials work for both:
;   prlctl getinfo           -> local pearld (port 44107)
;   prlctl --wallet getinfo  -> local oyster (port 44207)
; TLS cert defaults: pearld's rpc.cert, or oyster's when --wallet is set.

rpcuser=${RPC_USER}
rpcpass=${RPC_PASS}
rpcserver=127.0.0.1
EOF
			;;
		*)
			die "unknown config: $_name"
			;;
	esac
}

conf_get() {
	[ -f "$2" ] || return 0
	awk -F= -v key="$1" '$0 ~ "^" key "=" {
		val = substr($0, index($0, "=") + 1)
		if (val != "") print val
		exit
	}' "$2"
}

# True when the installed config already has non-empty RPC credentials.
config_has_secrets() {
	_name=$1
	_file=$2
	[ -f "$_file" ] || return 1
	case "$_name" in
		pearld | prlctl)
			_user=$(conf_get rpcuser "$_file")
			_pass=$(conf_get rpcpass "$_file")
			;;
		oyster)
			_user=$(conf_get username "$_file")
			_pass=$(conf_get password "$_file")
			;;
		*)
			return 1
			;;
	esac
	[ -n "$_user" ] && [ -n "$_pass" ]
}

ensure_rpc_credentials() {
	[ -n "$RPC_USER" ] && [ -n "$RPC_PASS" ] && return 0

	RPC_USER=$(conf_get rpcuser "$(config_path pearld)")
	RPC_PASS=$(conf_get rpcpass "$(config_path pearld)")
	if [ -n "$RPC_USER" ] && [ -n "$RPC_PASS" ]; then
		return 0
	fi

	RPC_USER=$(conf_get username "$(config_path oyster)")
	RPC_PASS=$(conf_get password "$(config_path oyster)")
	if [ -n "$RPC_USER" ] && [ -n "$RPC_PASS" ]; then
		return 0
	fi

	RPC_USER=$(conf_get rpcuser "$(config_path prlctl)")
	RPC_PASS=$(conf_get rpcpass "$(config_path prlctl)")
	if [ -n "$RPC_USER" ] && [ -n "$RPC_PASS" ]; then
		return 0
	fi

	RPC_USER=$(gen_secret)
	RPC_PASS=$(gen_secret)
}

install_default_configs() {
	created=0
	ensure_rpc_credentials

	for name in $CONFIGS; do
		dest=$(config_path "$name")
		if config_has_secrets "$name" "$dest"; then
			info "kept existing config -> ${dest}"
			continue
		fi
		write_config "$name" "$dest" 600
		info "wrote config -> ${dest}"
		created=1
	done

	if [ "$created" -eq 1 ]; then
		info "RPC username/password were auto-generated; no -u/-P flags needed"
	fi
}

print_summary() {
	cat << EOF

pearl-install: done (${VERSION})

Binaries:
  ${BIN_DIR}/pearld
  ${BIN_DIR}/prlctl
  ${BIN_DIR}/oyster$([ -f "${BIN_DIR}/oystercli" ] && printf '\n  %s' "${BIN_DIR}/oystercli")

Configs (OS default locations, shared RPC credentials):
  $(config_path pearld)
  $(config_path oyster)
  $(config_path prlctl)

Next steps (no -u/-P/-C needed):
  pearld
  oyster --create
  oyster                 # SPV sync by default
  prlctl getinfo
  prlctl --wallet getinfo$([ -f "${BIN_DIR}/oystercli" ] && printf '\n  %s' 'oystercli              # interactive wallet UI')
EOF

	if ! path_contains "$BIN_DIR"; then
		printf '\n'
		info "note: ${BIN_DIR} is not on your PATH"
		# shellcheck disable=SC2016
		printf '  add it with: export PATH="%s:%s"\n' "$BIN_DIR" '$PATH'
	fi
}

main() {
	parse_args "$@"
	if [ "$PRINT_HELP" -eq 1 ]; then
		usage
		exit 0
	fi

	need_cmd uname
	need_cmd mktemp
	need_cmd awk
	need_cmd basename
	need_cmd dirname
	need_cmd tar
	need_cmd cut
	need_cmd tr
	need_cmd cp

	# Binaries install under $HOME/.local/bin by default; configs use passwd home.
	: "${HOME:=$(user_home)}"
	[ -n "$HOME" ] || die "HOME is not set"

	OS=$(detect_os)
	arch=$(detect_arch "$OS")
	BIN_DIR=${BIN_DIR:-$(default_bin_dir)}
	ensure_bin_dir "$BIN_DIR"

	if [ -z "$VERSION" ]; then
		info "resolving latest release..."
		VERSION=$(resolve_latest_version)
	fi
	validate_version "$VERSION" || die "invalid version '${VERSION}' (expected vX.Y.Z)"

	archive_name="pearl-${OS}-${arch}-${VERSION}.tar.gz"
	asset_base="${RELEASE_BASE}/download/${VERSION}"

	TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/pearl-install.XXXXXX")
	trap cleanup EXIT INT HUP TERM

	info "downloading ${archive_name}"
	download "${asset_base}/${archive_name}" "${TMP_DIR}/${archive_name}"
	info "downloading checksums.txt"
	download "${asset_base}/checksums.txt" "${TMP_DIR}/checksums.txt"

	info "verifying SHA-256..."
	verify_checksum "${TMP_DIR}/${archive_name}" "${TMP_DIR}/checksums.txt"

	info "extracting binaries..."
	extract_binaries "${TMP_DIR}/${archive_name}" "${TMP_DIR}/staging"
	for bin in $BINARIES; do
		info "installing binary -> ${BIN_DIR}/${bin}"
		install_binary "${TMP_DIR}/staging/${bin}" "$BIN_DIR"
	done
	for bin in $OPTIONAL_BINARIES; do
		if [ -f "${TMP_DIR}/staging/${bin}" ]; then
			info "installing binary -> ${BIN_DIR}/${bin}"
			install_binary "${TMP_DIR}/staging/${bin}" "$BIN_DIR"
		else
			info "${bin} is not part of ${VERSION}; skipping"
		fi
	done

	install_default_configs
	print_summary
}

main "$@"
