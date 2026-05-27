//! Shared test utilities for vulkan-miner integration tests.
//!
//! Provides a mock JSON-RPC 2.0 gateway server that can be spawned
//! in a background tokio task for protocol-level testing.

use base64::Engine;

/// Build a valid 80-byte mock block header and base64-encode it.
pub fn mock_header_b64() -> String {
    let mut header = [0u8; 80];
    header[0..4].copy_from_slice(&1u32.to_le_bytes());
    for i in 0..32 { header[4 + i] = (i + 1) as u8; }
    for i in 0..32 { header[36 + i] = (i + 0x11) as u8; }
    header[68..72].copy_from_slice(&0x12345678u32.to_le_bytes());
    header[72..76].copy_from_slice(&0x1D00FFFFu32.to_le_bytes());
    header[76..80].copy_from_slice(&0u32.to_le_bytes());
    base64::engine::general_purpose::STANDARD.encode(&header)
}

pub fn mock_target() -> u64 {
    0x00FFFFFF_00000000
}

/// Start a mock JSON-RPC 2.0 gateway on a random TCP port.
///
/// Returns the address string (e.g. `"127.0.0.1:54321"`).
pub async fn start_mock_gateway() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let header_b64 = mock_header_b64();
    let target = mock_target();

    tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            let header_b64 = header_b64.clone();
            tokio::spawn(handle_client(stream, header_b64, target));
        }
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    addr
}

async fn handle_client(stream: tokio::net::TcpStream, header_b64: String, target: u64) {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let (reader, mut writer) = tokio::io::split(stream);
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await.unwrap();
        if n == 0 { return; }
        let line = line.trim();
        if line.is_empty() { continue; }

        let req: serde_json::Value = serde_json::from_str(line).unwrap();
        let method = req["method"].as_str().unwrap_or("");
        let id = &req["id"];

        let response = match method {
            "getMiningInfo" => serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "incomplete_header_bytes": header_b64,
                    "target": target,
                },
                "id": id,
            }),
            "submitPlainProof" => serde_json::json!({
                "jsonrpc": "2.0",
                "result": "submitted",
                "id": id,
            }),
            _ => serde_json::json!({
                "jsonrpc": "2.0",
                "error": { "code": -32601, "message": "Method not found" },
                "id": id,
            }),
        };

        let resp_line = serde_json::to_string(&response).unwrap();
        writer.write_all(resp_line.as_bytes()).await.unwrap();
        writer.write_all(b"\n").await.unwrap();
        writer.flush().await.unwrap();
    }
}
