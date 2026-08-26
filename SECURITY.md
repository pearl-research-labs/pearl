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

All components are covered by this policy for vulnerability reporting and
coordinated disclosure. Bounty rewards, however, apply only to a subset of
components — see [Bounty scope](#bounty-scope).

## Supported Versions

Only the latest release is actively supported. Critical fixes may be
backported to prior releases at the team's discretion.

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Private vulnerability reporting is for **security vulnerabilities only**.
Non-security bugs, crashes without security impact, and feature requests
should be filed as regular [GitHub issues](https://github.com/pearl-research-labs/pearl/issues)
instead.

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

### Bounty scope

Bounty rewards apply only to vulnerabilities in the protocol, the node, and the
wallet components:

- **pearld** — full node and protocol (`node/`)
- **ZK proof-of-work** circuits and verifier (`zk-pow/`, `plonky2/`)
- **XMSS** post-quantum signatures (`xmss/`)
- **Oyster** - wallet daemon (`wallet/`)
- **SPV** - light client (`spv/`)

Reports against other components in the repo (`miner/`, `py-pearl-mining/`,
`dnsseeder/`, `apps/`) are **not bounty-eligible** and should be filed as
regular [GitHub issues](https://github.com/pearl-research-labs/pearl/issues).
Severe vulnerabilities in those components may still be reported privately
via [Reporting a Vulnerability](#reporting-a-vulnerability), but no reward
is implied.

### Reward guidelines

Indicative rewards by severity (as determined by us):

| Severity | Reward |
| --- | --- |
| Low | 1,000 PRL |
| Medium | 2,000 PRL |
| High | 10,000 PRL |
| Critical | 50,000+ PRL |

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
- **Identity verification.** To comply with applicable U.S. sanctions laws
  and regulations, Contributors may be required to disclose and verify their
  identity.

Submit reports through the process described in
[Reporting a Vulnerability](#reporting-a-vulnerability).

## Contact

- [Report a vulnerability](https://github.com/pearl-research-labs/pearl/security/advisories/new)
- Website: https://pearlresearch.ai
