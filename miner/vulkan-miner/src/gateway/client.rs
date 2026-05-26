use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

/// Mining job received from the pearl gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningJob {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub rank: u32,
    pub header_version: u32,
    pub header_prev_block: [u8; 32],
    pub header_merkle_root: [u8; 32],
    pub header_timestamp: u32,
    pub header_nbits: u32,
    pub target: [u8; 32],
}

/// Gateway RPC client over a Unix domain socket.
/// Communicates with the pearl gateway daemon using bincode-encoded messages.
pub struct GatewayClient {
    stream: UnixStream,
}

impl GatewayClient {
    pub async fn connect(path: &str) -> Result<Self> {
        let stream = UnixStream::connect(path).await?;
        Ok(Self { stream })
    }

    /// Request a new mining job from the gateway.
    pub async fn get_job(&mut self) -> Result<MiningJob> {
        // Send request (empty GET_JOB message)
        let req = b"GET_JOB";
        self.stream.write_all(req).await?;

        // Read the response: first 4 bytes = length prefix
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;

        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf).await?;

        let job: MiningJob = bincode::deserialize(&buf)?;
        Ok(job)
    }

    /// Submit a solved block proof.
    pub async fn submit_block(&mut self, proof_data: &[u8]) -> Result<()> {
        let msg = bincode::serialize(&SubmitRequest {
            msg_type: 1, // SUBMIT_BLOCK
            data: proof_data.to_vec(),
        })?;

        let len = (msg.len() as u32).to_le_bytes();
        self.stream.write_all(&len).await?;
        self.stream.write_all(&msg).await?;

        // Read acknowledgement
        let mut ack = [0u8; 4];
        self.stream.read_exact(&mut ack).await?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct SubmitRequest {
    msg_type: u32,
    data: Vec<u8>,
}
