#!/usr/bin/env bash
set -euo pipefail

mode="${1:-pr}"
workers="${PYTEST_WORKERS:-8}"
repo_root="$(git rev-parse --show-toplevel)"
package_root="${repo_root}/miner/pearl-gemm"
uv_cmd=(uv --directory "$repo_root" run --locked --package pearl-gemm --extra cuda)

"${uv_cmd[@]}" ruff format --check "$package_root/src" "$package_root/tests"
"${uv_cmd[@]}" ruff check "$package_root/src" "$package_root/tests"

if [[ "$mode" == "pr" ]]; then
  "${uv_cmd[@]}" pytest -n "$workers" -m "not slow" "$package_root/tests"
elif [[ "$mode" == "full" ]]; then
  "${uv_cmd[@]}" pytest -n "$workers" "$package_root/tests"
else
  echo "usage: $0 [pr|full]" >&2
  exit 2
fi
