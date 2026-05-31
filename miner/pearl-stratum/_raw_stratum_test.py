"""Minimal raw stratum test - opens TCP, sends ONE message at a time, dumps bytes."""
import socket
import time
import sys

HOST = "us2.alphapool.tech"
PORT = 5566


def send_and_read(sock, payload: bytes, label: str, wait_s: float = 3.0):
    print(f"\n--- SEND {label} ({len(payload)} bytes) ---")
    print(payload.decode("utf-8", errors="replace").rstrip())
    sock.sendall(payload)
    sock.settimeout(wait_s)
    try:
        buf = b""
        t0 = time.time()
        while time.time() - t0 < wait_s:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    print(f"--- PEER CLOSED after {time.time()-t0:.2f}s ---")
                    return False
                buf += chunk
                if b"\n" in buf:
                    break
            except socket.timeout:
                break
        print(f"--- RECV ({len(buf)} bytes) ---")
        for line in buf.decode("utf-8", errors="replace").splitlines():
            if line:
                print(line)
        return True
    except Exception as e:
        print(f"--- RECV ERROR: {e} ---")
        return False


def main():
    sock = socket.create_connection((HOST, PORT), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"Connected to {HOST}:{PORT} via {sock.getsockname()}")

    # COMPACT JSON exactly matching alpha-miner's 64/66/141-byte wire format.
    configure = b'{"id":46,"method":"mining.configure","params":[["pearl/v1"],{}]}\n'
    if not send_and_read(sock, configure, "mining.configure", wait_s=30.0):
        return 1

    subscribe = b'{"id":47,"method":"mining.subscribe","params":["alpha-miner/0.1"]}\n'
    if not send_and_read(sock, subscribe, "mining.subscribe", wait_s=8.0):
        return 1

    authorize = (
        b'{"id":48,"method":"mining.authorize",'
        b'"params":["prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg.test","x;d=1048576"]}'
        b'\n'
    )
    if not send_and_read(sock, authorize, "mining.authorize", wait_s=5.0):
        return 1

    print("\n--- WAITING 30s for notifications ---")
    sock.settimeout(30)
    try:
        buf = b""
        t0 = time.time()
        while time.time() - t0 < 30:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    print(f"PEER CLOSED at t={time.time()-t0:.2f}s")
                    break
                buf += chunk
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    if line.strip():
                        print(f"[t={time.time()-t0:6.2f}s] {line.decode('utf-8', errors='replace')}")
            except socket.timeout:
                continue
    except Exception as e:
        print(f"Loop exit: {e}")

    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
