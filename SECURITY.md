# Security Policy

## Scope

This policy covers all components in the Pearl monorepo:

- **pearld** — full node (`node/`)
- **Oyster** — wallet daemon (`wallet/`)
- **SPV client** (`spv/`)
- **ZK proof-of-work** circuits and verifier (`zk-pow/`, `plonky2/`)
- **Mining infrastructure** (`miner/`, `py-pearl-mining/`)
- **XMSS** post-quantum signatures (`xmss/`)
- **DNS seeder** (`dnsseeder/`)
- **Frontend applications** (`apps/`)

## Supported Versions

Only the latest release is actively supported. Critical fixes may be
backported to prior releases at the team's discretion.

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Use [GitHub's private vulnerability reporting](https://github.com/pearl-research-labs/pearl/security/advisories/new)
to submit a report. Include:

- Description of the vulnerability
- Steps to reproduce or a proof-of-concept
- Affected component(s) and version(s)
- Potential impact

## Disclosure Policy

We follow coordinated disclosure. Please allow reasonable time from the
initial report before publicly disclosing any findings, so we have time to
develop and release a fix. We will credit reporters in the release notes
unless anonymity is requested.

## Bug Bounty

Pearl Research Labs may, at its sole discretion, offer rewards in PRL for
qualifying security reports. This is a voluntary recognition program, not a
contractual offer, contest, or guarantee of payment.

### Reward guidelines

Indicative reward ranges by severity (as determined by us):

| Severity | Reward |
| --- | --- |
| Low | 500 PRL |
| Medium | 1,000 PRL |
| High | 2,000 PRL |
| Critical | 50,000 PRL |

### Terms

- **Internal triage.** We assess severity, validity, novelty, and impact
  internally. Our classification is final.
- **Discretionary awards.** Whether to reward a report, at what severity
  tier, and in what amount is entirely at our discretion. The guidelines
  above are non-binding and may be adjusted or withheld for any reason.
- **Basis for reward.** An award, if any, may be based on responsible
  disclosure, assistance with remediation, demonstration of a fix, or any
  other contribution we consider valuable. There is no single required
  trigger for payment.
- **No entitlement.** Submission of a report does not create any right,
  claim, or expectation to a reward. Duplicate, out-of-scope, low-quality,
  or previously known issues may receive no award.

Submit reports through the process described in
[Reporting a Vulnerability](#reporting-a-vulnerability).

## Contact

- [Report a vulnerability](https://github.com/pearl-research-labs/pearl/security/advisories/new)
- Website: https://pearlresearch.ai
