# Security Policy

## Scope

This policy covers all components in the Pearl monorepo:

- **pearld** — full node (`node/`)
- **Oyster** — wallet daemon (`wallet/`)
- **SPV client** (`spv/`)
- **ZK proof-of-work** circuits and verifier (`zk-pow/`, `plonky2/`)
- **Mining infrastructure** (`miner/`, `py-pearl-mining/`)
- **XMSS** post-quantum signatures (`xmss/`)
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

The rewards mentioned above apply only to production code on the default
branch (`master`) or the latest supported release. Reports against feature
branches, work-in-progress, experimental PRs, or other unmerged code are
**not bounty-eligible**, even if the component would otherwise be in bounty
scope. Nevertheless, such findings may still be reported privately via
[Reporting a Vulnerability](#reporting-a-vulnerability), or as regular
[GitHub issues](https://github.com/pearl-research-labs/pearl/issues) for
non-security bugs.

Reports against other components in the repo (`miner/`, `py-pearl-mining/`,
`apps/`) are **not bounty-eligible** and should be filed as
regular [GitHub issues](https://github.com/pearl-research-labs/pearl/issues).
Severe vulnerabilities in those components may still be reported privately
via [Reporting a Vulnerability](#reporting-a-vulnerability), but no reward
is implied.

### Severity and rewards

Indicative rewards by severity (as determined by us; our decision is final):

| Severity | Reward |
| --- | --- |
| Low | 1,000 PRL |
| Medium | 2,000 PRL |
| High | 10,000 PRL |
| Critical | 50,000+ PRL |

The examples below illustrate each tier; they are not exhaustive.

**Critical** — Bugs that threaten the fundamental security and integrity of the entire network.
These are bugs that allow coin theft at the protocol level or the creation of coins outside the
specified issuance schedule. For example, compromising the ZK verifier so that blocks can be mined
without actual work.

**High** — Bugs with a significant impact on affected nodes or the network. These are typically
exploitable remotely under default configurations and can cause widespread disruption. For example,
halting block production or block relay network-wide at low cost.

**Medium** — Bugs that noticeably degrade the performance or functionality of the network or a
node. For example, crashing a node through a prolonged attack.

**Low** — Bugs that are hard to exploit or have a minor impact on a node's operation. For example,
slowing a single node in a way that clears when the attack stops.

### Reward adjustments

- **Publicly known issues.** Pearl is derived from
  [btcd](https://github.com/btcsuite/btcd),
  [btcwallet](https://github.com/btcsuite/btcwallet),
  [neutrino](https://github.com/lightninglabs/neutrino), and
  [plonky2](https://github.com/0xPolygonZero/plonky2). If a vulnerability was
  publicly disclosed in any of these projects or their forks before your
  report (in an advisory, CVE, issue, commit, or release note), the reward
  may be reduced or withheld entirely.
- **Exploitation.** Assessing and fixing a vulnerability takes time, and we
  will not reward anyone who exploits a bug, whether before or after reporting
  it. If the reporter, or anyone acting with them, exploited it, the reward is
  forfeited. If someone else exploits it after we have had reasonable time to
  act on your report, your reward will not be affected. If exploitation began
  before then, or was already under way when you reported it, the reward may
  be reduced or withheld. Test only on regtest, simnet, or private networks with
  nodes and wallets you control.
- **Premature disclosure.** Disclosing the vulnerability publicly, or to any
  third party, before we release a fix has the same effect as exploiting it.

### Not eligible

- Attacks that require a majority of hashrate or of a node's peers
- Volumetric flooding (bandwidth or connection floods) without an
  amplification or protocol flaw
- Theoretical issues with no proof of concept or concrete attack path
- Issues reachable only through authenticated RPC
- Missing best practices, scanner output, or reports without security impact
- Social engineering, phishing, or attacks on Pearl Research Labs
  infrastructure, staff, or users

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
