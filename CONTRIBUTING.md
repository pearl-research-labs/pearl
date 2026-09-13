# Contributing to Pearl

## Getting Started

1. Clone the repository
2. Install prerequisites: Go 1.26+, Rust toolchain, C compiler, Python 3.12, [uv](https://docs.astral.sh/uv/), [Task](https://taskfile.dev), CUDA 13 toolkit and an SM100 (B200) GPU (for the vLLM miner and `miner/pearl-gemm`)
3. Build: `task build`
4. Test: `task test`

You can also build or test specific components:

```
task build:blockchain   # pearld, prlctl, oyster
task build:miner        # Python miner packages (pearl-gateway, miner-base, vLLM miner, pearl_gemm)
task test:go            # Go tests only
task test:python        # full Python test suite
task test:python:basic  # Python tests (excludes integration/perf/slow)
```

## Submitting Changes

1. Create a branch from `master`
2. Keep PRs focused — one fix or feature per PR
3. Run `task fmt lint:python tidy` before pushing
4. All CI checks must pass
5. Bug fixes should include a test that reproduces the issue

## Commit Messages

Use the format: `type(scope): description`

```
fix(node): correct block validation for edge case
feat(miner): add GPU memory monitoring
docs: update build instructions
```

## Sign-Off

By submitting a PR you certify that your contribution is your own work
and you have the right to submit it under the project's ISC License.

## Security

Do **not** open public issues for security vulnerabilities.
See [SECURITY.md](SECURITY.md).
