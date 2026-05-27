use anyhow::{Context, Result};
use base64::Engine;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[cfg(unix)]
use tokio::net::UnixStream;

#[cfg(windows)]
use tokio::net::TcpStream;

/// Mining job received from the pearl gateway via JSON-RPC 2.0.
///
/// The gateway sends `{ incomplete_header_bytes: "<base64>", target: <int> }`.
#[derive(Debug, Clone)]
pub struct MiningJob {
    /// Raw 80‑byte incomplete block header.
    pub incomplete_header_bytes: [u8; 80],
    /// PoW target as big‑endian u256.
    pub target: [u8; 32],
    /// Extracted fields for convenience.
    pub prev_block: [u8; 32],
    pub merkle_root: [u8; 32],
}

/// Gateway RPC client speaking line‑delimited JSON-RPC 2.0.
///
/// On Unix connects via UDS; on Windows via TCP loopback.
pub struct GatewayClient {
    reader: BufReader<Box<dyn tokio::io::AsyncRead + Unpin + Send>>,
    writer: Box<dyn tokio::io::AsyncWrite + Unpin + Send>,
}

impl GatewayClient {
    pub async fn connect(path: &str) -> Result<Self> {
        #[cfg(unix)]
        {
            let stream = UnixStream::connect(path).await?;
            let (r, w) = tokio::io::split(stream);
            Ok(Self {
                reader: BufReader::new(Box::new(r)),
                writer: Box::new(w),
            })
        }
        #[cfg(windows)]
        {
            let stream = TcpStream::connect(path).await?;
            let (r, w) = tokio::io::split(stream);
            Ok(Self {
                reader: BufReader::new(Box::new(r)),
                writer: Box::new(w),
            })
        }
    }

    /// Request a new mining job via `getMiningInfo`.
    pub async fn get_job(&mut self) -> Result<MiningJob> {
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "getMiningInfo",
            "params": {},
            "id": 1,
        });

        let mut line = String::new();
        self.writer
            .write_all(serde_json::to_string(&req)?.as_bytes())
            .await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;

        loop {
            line.clear();
            let n = self.reader.read_line(&mut line).await?;
            if n == 0 {
                anyhow::bail!("gateway connection closed");
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let resp: serde_json::Value =
                serde_json::from_str(line).context("failed to parse JSON-RPC response")?;

            if let Some(err) = resp.get("error") {
                let code = err.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
                let msg = err
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error");
                anyhow::bail!("gateway error {}: {}", code, msg);
            }

            let result = &resp["result"];
            let header_b64 = result["incomplete_header_bytes"]
                .as_str()
                .context("missing incomplete_header_bytes")?;
            let target_val = result["target"]
                .as_u64()
                .context("missing or invalid target")?;

            let incomplete_header_bytes: [u8; 80] =
                Self::decode_b64_header(header_b64)?;

            let mut prev_block = [0u8; 32];
            let mut merkle_root = [0u8; 32];
            prev_block.copy_from_slice(&incomplete_header_bytes[4..36]);
            merkle_root.copy_from_slice(&incomplete_header_bytes[36..68]);

            let mut target = [0u8; 32];
            target[24..32].copy_from_slice(&target_val.to_be_bytes());

            return Ok(MiningJob {
                incomplete_header_bytes,
                target,
                prev_block,
                merkle_root,
            });
        }
    }

    /// Submit a solved `PlainProof` via `submitPlainProof`.
    pub async fn submit_plain_proof(
        &mut self,
        plain_proof_bincode: &[u8],
        mining_job: &MiningJob,
    ) -> Result<()> {
        let proof_b64 = base64::engine::general_purpose::STANDARD.encode(plain_proof_bincode);
        let header_b64 =
            base64::engine::general_purpose::STANDARD.encode(&mining_job.incomplete_header_bytes);

        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "submitPlainProof",
            "params": {
                "plain_proof": proof_b64,
                "mining_job": {
                    "incomplete_header_bytes": header_b64,
                    "target": u64::from_be_bytes(mining_job.target[24..32].try_into().unwrap()),
                },
            },
            "id": 1,
        });

        self.writer
            .write_all(serde_json::to_string(&req)?.as_bytes())
            .await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;

        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line).await?;
            if n == 0 {
                anyhow::bail!("gateway connection closed on submit");
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let resp: serde_json::Value =
                serde_json::from_str(line).context("failed to parse submit response")?;
            if let Some(err) = resp.get("error") {
                let code = err.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
                let msg = err
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error");
                anyhow::bail!("gateway submit error {}: {}", code, msg);
            }
            return Ok(());
        }
    }

    fn decode_b64_header(s: &str) -> Result<[u8; 80]> {
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(s)
            .context("base64 decode failed")?;
        let mut out = [0u8; 80];
        if decoded.len() != out.len() {
            anyhow::bail!("expected {} bytes, got {}", out.len(), decoded.len());
        }
        out.copy_from_slice(&decoded);
        Ok(out)
    }
}
