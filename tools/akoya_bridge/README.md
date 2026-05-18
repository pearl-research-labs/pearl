# Akoya Bridge Tools

Local gates for `P1K-131`, the acceptance-first branch.

These scripts do not touch CUDA mainloop files and do not require a GPU.

```bash
python3 tools/akoya_bridge/test_akoya_protocol.py

PYTHONPATH=/home/bereket/.local/lib/python3.12/site-packages \
  .venv/bin/python tools/akoya_bridge/verify_captured_share.py

PYTHONPATH=/home/bereket/.local/lib/python3.12/site-packages \
  .venv/bin/python tools/akoya_bridge/test_share_builder.py

python3 tools/akoya_bridge/live_no_submit_scaffold.py

python3 -m py_compile tools/akoya_bridge/direct_gpu_akoya_submit.py
```

`test_akoya_protocol.py` validates the captured Akoya MessagePack schema.
`verify_captured_share.py` rebuilds a local `pearl_mining.PlainProof` from
the first accepted Akoya type-3 share fixture and verifies it with the pool
share difficulty override.
`test_share_builder.py` proves the type-3 builder can reconstruct the captured
accepted share from a local `PlainProof.to_base64()` payload plus canonical
jackpot fields.
`live_no_submit_scaffold.py` registers to Akoya and receives one live job. By
default it does not mine and never submits a share.
`direct_gpu_akoya_submit.py` is the P1K-132 GPU runner for the exact Akoya
baseline shape. Run its `--force-target-max` mode first as a no-submit H200
canary, then use `--submit` only after that canary proves host-signal and local
PlainProof verification are alive.

`P1K-131` is only complete when our own submitted share receives Akoya type
`4` `ShareResult` outcome `0`. TMAD/s diagnostics do not close this ticket.
