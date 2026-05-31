"""Env-driven config for the solo pool.

All knobs are environment variables, prefixed PEARL_SRV_, so the same
binary can be reconfigured per-deploy without code edits.

Required:
  PEARL_SRV_RPC_URL        e.g. http://127.0.0.1:18334  (pearld plaintext RPC)
  PEARL_SRV_RPC_USER       pearld rpcuser
  PEARL_SRV_RPC_PASSWORD   pearld rpcpass
  PEARL_SRV_MINING_ADDRESS prl1...  receiving address for the coinbase

Optional (defaults match mainnet alphapool):
  PEARL_SRV_LISTEN_HOST    default "0.0.0.0"
  PEARL_SRV_LISTEN_PORT    default 5566
  PEARL_SRV_POLL_INTERVAL  default 2.0 s (gbt poll cadence)
  PEARL_SRV_PARAM_M / _N / _K / _RANK  (mainnet defaults)
  PEARL_SRV_MMA_TYPE       default "Int7xInt7ToInt32"
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PEARL_SRV_", env_file=None)

    # pearld RPC (required)
    rpc_url: str
    rpc_user: str
    rpc_password: str
    mining_address: str

    # stratum listener
    listen_host: str = "0.0.0.0"
    listen_port: int = 5566

    # metrics + healthcheck HTTP listener
    metrics_host: str = "0.0.0.0"
    metrics_port: int = 9101
    """Set to 0 to disable the metrics endpoint entirely."""

    metrics_max_template_age_seconds: float = 60.0
    """Health endpoint returns 503 if the last template was minted longer ago than this."""

    # template fetching
    poll_interval: float = 2.0
    """Seconds to sleep between template fetches. Used as the fast-poll
    cadence when long-poll is disabled, and as the backoff window after RPC
    errors regardless of long-poll setting."""

    long_poll: bool = True
    """If True, use pearld's longpollid handshake — fetch blocks until the
    chain advances. Drops stale-share window from ~poll_interval to <100ms."""

    long_poll_timeout_secs: float = 30.0
    """Per-call long-poll timeout. pearld returns the (unchanged) template
    after this elapses even if the tip hasn't moved, which keeps the
    connection healthy through firewall idle timers."""

    # mining params pushed to clients (must match mainnet consensus)
    param_m: int = 131072
    param_n: int = 131072
    param_k: int = 4096
    param_rank: int = 128
    param_rows_pattern: tuple[int, ...] = (0, 32)
    param_cols_pattern: tuple[int, ...] = tuple(range(64))
    mma_type: str = "Int7xInt7ToInt32"

    # operational
    job_history_size: int = 16
    """How many recent jobs to keep for submit lookups. Old jobs get error[21]."""

    debug_verify: bool = False
    """If True, ProofGenerator runs verify_proof after generate_proof. Slow; off in prod."""

    # ---- public-pool mode ----------------------------------------------
    # When `public_pool=False` (default), all blocks pay 100% to mining_address
    # (solo mode). When True, miners' addresses (parsed from mining.authorize)
    # get PPLNS payouts, with operator_fee_percent going to mining_address.

    public_pool: bool = False
    """Enable multi-tenant mode: per-miner accounting, PPLNS payouts, vardiff,
    challenge handshake, IP rate-limits, banned-IP enforcement."""

    share_db_path: str = "/var/lib/pearl-stratum-srv/shares.sqlite3"
    """Where to persist shares + blocks + payouts in public-pool mode."""

    pool_fee_percent: float = 1.0
    """Operator's cut of every block reward. 1% is competitive."""

    pplns_window_difficulty: int = 100_000_000
    """Total difficulty units in the PPLNS lookback window. Larger = smoother
    payouts but slower to adapt to fleet changes. 100M ≈ 100 shares at d=2^20."""

    min_payout_sats: int = 100_000
    """Recipients owed less than this are rolled into the operator fee.
    Avoids spamming the chain with dust outputs."""

    vardiff_target_shares_per_min: float = 6.0
    """Per-worker share rate goal. 6/min = 1 share / 10s, matches alpha-pool."""

    challenge_difficulty: int = 0
    """`pearl.challenge` leading-zero bits. 0 = disabled. Production v1.5
    pools use 32. Disabled by default since most deploys are LAN; turn on
    when exposing publicly."""

    max_connections_per_ip: int = 200
    """Concurrent stratum connections per source IP. 200 supports a rig with
    up to ~200 GPUs; legit rigs rarely exceed 20."""

    max_new_connections_per_minute_per_ip: int = 60
    """New TCP connections per minute per source IP. 60 = 1/sec sustained;
    bursts above this look like connection-flood probes."""

    malformed_share_ban_threshold: int = 50
    """Auto-ban an IP if it sends this many `malformed` shares within 5 minutes.
    Legit miners send 0; sustained nonzero = buggy miner or attack."""

    malformed_share_ban_duration_secs: float = 3600.0
    """How long auto-bans last. 1 hour gives time to investigate/fix."""

    operator_dashboard_token: str = ""
    """Shared-secret token gating /op?token=... view. Empty = view disabled.
    Set to a long random string for production."""

    # ---- alerter ---------------------------------------------------------
    alert_log_path: str = "/var/log/pearl-stratum-srv/alerts.log"
    """Where to append JSON alert lines. Empty = disable file delivery."""

    alert_webhook_url: str = ""
    """Discord/Slack-compatible webhook to POST `{"content": "..."}` to.
    Empty = disable webhook delivery."""

    alert_template_age_seconds: float = 120.0
    """Fire `template_stale` if pearld stopped delivering work for this long."""

    alert_template_never_seconds: float = 300.0
    """Fire `template_never` if the pool has been up this long but pearld has
    never returned a single template — usually means pearld is in IBD or
    RPC creds are wrong."""

    alert_no_miners_seconds: float = 180.0
    """Fire `no_miners` if a public-pool stays at 0 connected for this long."""

    alert_rig_idle_seconds: float = 180.0
    """Fire `rig_idle:<worker>` if a worker submitted 0 shares in this window
    (and was active before — fresh joiners don't trigger)."""

    alert_malformed_total_threshold: int = 10
    """Fire `malformed_flood` once total malformed shares hits this count."""

    alert_tick_seconds: float = 15.0
    """How often the alerter evaluates rules."""

    def mining_params_payload(self) -> dict:
        """The dict pushed in pearl.set_mining_params after subscribe."""
        return {
            "m": self.param_m,
            "n": self.param_n,
            "k": self.param_k,
            "rank": self.param_rank,
            "rows_pattern": list(self.param_rows_pattern),
            "cols_pattern": list(self.param_cols_pattern),
            "mma_type": self.mma_type,
        }
