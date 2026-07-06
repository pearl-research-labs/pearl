# CI Security Tiers

This repository uses a 3-tier CI system to safely support open-source contributions.

## Tier Overview

| Tier | Trigger | Secrets | Runners | Purpose |
|------|---------|---------|---------|---------|
| 1 — Untrusted PRs | `pull_request` | None | `ubuntu-latest` | Lint, format, basic unit tests |
| 2 — Maintainer-Approved | `merge_group` or `pull_request_target` + `safe-to-test` label | EC2, PAT, HF_TOKEN | GPU (EC2), `large-runner` | GPU tests, Docker builds, integration tests, perf tests |
| 3 — Protected Release | `workflow_dispatch` + `release` environment | GITHUB_TOKEN | Various (matrix) | Binary builds, GitHub Releases |

## Tier 1 Workflows (run on every PR, including forks)

- **`miner_ci.yml`** — Ruff/clang-format checks, Pearl Gateway + Miner Base pytest
- **`rust_ci.yml`** — pearl-blake3, zk-pow, py-pearl-mining (fmt, clippy, tests)
- **`plonky2_ci.yml`** — Plonky2 fmt, clippy, tests
- **`blockchain_ci.yml`** — Go fmt/tidy, blockchain build + tests

## Tier 2 Workflows (require maintainer approval)

- **`miner_gpu_ci.yml`** — GPU tests on EC2 H100 runner
- **`integration_tests_ci.yml`** — Full integration tests (pearld + vLLM + miner)
- **`miner_heavy_ci.yml`** — Performance tests + vLLM Docker image build
- **`pearl-desktop-wallet.yml`** — Cross-platform Electron + Go builds
- **`blockchain_build.yml`** — Continuous per-commit build (on `push` to master +
  manual `workflow_dispatch`, including side branches). Publishes commit-hash-only
  `pearld` images (`ghcr.io/.../pearld:sha-<commit>`, needs `packages: write`)
  and commit-hash-named binary artifacts for internal/testnet testing. Never
  produces version/`latest` tags — those are release-only. Because it holds
  `packages: write`, keep `workflow_dispatch` maintainer-triggered so untrusted
  forks cannot push to GHCR.

### How Tier 2 gating works

1. **Merge queue** (`merge_group`): Tier 2 workflows run automatically when a PR enters the merge queue. This requires branch protection approval first.

2. **Label-based** (`pull_request_target` + `safe-to-test`): A maintainer adds the `safe-to-test` label to run Tier 2 checks before queueing. When new commits are pushed to the PR, the label is automatically removed (by `remove_safe_label.yml`), requiring re-approval.

## Tier 3 Workflows (release only)

- **`blockchain_release.yml`** — Promotion-only (workflow_dispatch, `release`
  environment). Does not rebuild: it promotes the artifacts `blockchain_build.yml`
  already produced for the release commit — re-tagging the `sha-<commit>` image to
  `:vX.Y.Z` + `:latest`, and renaming the tested binaries to their version names
  for the GitHub Release.
- **`pearl-desktop-wallet.yml`** release job — Wallet release (protected by `release` environment)