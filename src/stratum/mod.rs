/// Stratum client implementation
/// Stratum protocol handlers
/// Stratum protocol types and errors
pub mod client;
/// Stratum protocol implementation module
/// Stratum types and error definitions module
pub mod protocol; 
/// Stratum types and error definitions module
pub mod types;

pub use client::{StratumClient, StratumSolution};
pub use types::StratumError;
