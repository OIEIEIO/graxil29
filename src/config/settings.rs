// src/config/settings.rs - Updated for real pool mining

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Main configuration settings for the miner
pub struct Settings {
    /// Pool URL to connect to
    pub pool_url: String,
    /// Your Grin wallet address for mining rewards
    pub wallet_address: String,
    /// Worker name identifier
    pub worker_name: String,
    /// Mining difficulty target
    pub difficulty: u64,
    /// Number of GPU threads
    pub gpu_threads: u32,
    /// Enable/disable GPU mining
    pub use_gpu: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            // Active Grin pools (as of 2024-2025)
            pool_url: "stratum+tcp://grin.2miners.com:3030".to_string(),
            wallet_address: "grin1642yrdxlf5e00teen883vl0f4ezlp9kn52zv8dm7hpu7ytf5js9q63dts9".to_string(),
            worker_name: "graxil29_rtx4060ti".to_string(),
            difficulty: 1000000, // Starting difficulty
            gpu_threads: 1024,   // RTX 4060 Ti optimal
            use_gpu: true,
        }
    }
}

// Pool options (update as needed):
// 2Miners: grin.2miners.com:3030
// Grinmint: eu-west-stratum.grinmint.com:4416
// F2Pool: grin.f2pool.com:13654