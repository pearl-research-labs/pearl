"""Settings defaults must match mainnet so miners don't desync."""


def test_mining_params_match_mainnet(settings):
    p = settings.mining_params_payload()
    assert p["m"] == 131072
    assert p["n"] == 131072
    assert p["k"] == 4096
    assert p["rank"] == 128
    assert p["mma_type"] == "Int7xInt7ToInt32"
    assert p["rows_pattern"] == [0, 32]
    assert p["cols_pattern"] == list(range(64))


def test_default_listen_is_5566(settings):
    assert settings.listen_port == 5566
    assert settings.listen_host == "0.0.0.0"


def test_default_poll_interval_is_2s(settings):
    assert settings.poll_interval == 2.0
