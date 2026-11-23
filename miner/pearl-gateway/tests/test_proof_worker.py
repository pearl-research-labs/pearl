from types import SimpleNamespace

from pearl_gateway import proof_worker


def test_fp8_prover_is_reused_across_devices_and_shapes(monkeypatch):
    setups = []

    class FakeProver:
        @staticmethod
        def setup(device):
            setups.append(device)
            return FakeProver()

        def prove(self, _header, plain_proof):
            return bytes([plain_proof.common.k]), bytes([plain_proof.common.device])

    monkeypatch.setattr(proof_worker, "Fp8Prover", FakeProver)
    monkeypatch.setattr(proof_worker, "_fp8_prover", None)

    first = SimpleNamespace(common=SimpleNamespace(device=1, k=8))
    second = SimpleNamespace(common=SimpleNamespace(device=0, k=16))

    assert proof_worker._prove_fp8(object(), first, False) == (b"\x08", b"\x01")
    assert proof_worker._prove_fp8(object(), second, False) == (b"\x10", b"\x00")
    assert setups == [1]
