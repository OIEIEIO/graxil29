// src/lib.rs - Main library file for Graxil29 Cuckoo Cycle mining
// Tree location: ./src/lib.rs

//! Graxil29 v2.0 - Modern Rust Grin Miner
//! 
//! A high-performance, GPU-accelerated Grin miner implementing both Cuckaroo29 and Cuckatoo32 algorithms.
//! Built with modern Rust practices and cross-platform GPU compute via OpenCL.
//!
//! # Version History
//! - 0.1.0: Initial Cuckaroo29 implementation with WGSL mock solutions  
//! - 0.2.0: Added dual algorithm support (C29 + C32)
//! - 0.3.0: **MAJOR**: Complete rewrite using OpenCL backend for real mining
//! - 0.3.1: Fixed compilation errors, added OpenCL error handling
//! - 0.3.2: Enhanced error conversion, removed unsafe code restrictions
//! - 0.3.3: Added From<hex::FromHexError> for hex decode error handling

#![warn(missing_docs)]
// Note: OpenCL mining requires unsafe operations for GPU compute and memory management
#![allow(unsafe_code)]

pub mod algorithms;
/// Configuration module for miner settings
pub mod config;
/// GPU computation and memory management
pub mod gpu;
/// Performance metrics and monitoring
pub mod metrics;
/// Stratum protocol implementation for pool mining
pub mod stratum;

// Re-export main types for convenience
pub use algorithms::{Algorithm, Solution, AlgorithmError, CuckooMiner};
pub use config::Settings;
pub use stratum::{StratumClient, StratumError};
pub use gpu::compute::GpuCompute;

use thiserror::Error;
use hex::FromHexError;

/// Main error type for Graxil29
#[derive(Error, Debug)]
pub enum Graxil29Error {
    /// Algorithm-related errors
    #[error("Algorithm error: {0}")]
    Algorithm(#[from] AlgorithmError),
    
    /// GPU computation errors
    #[error("GPU error: {0}")]
    Gpu(String),
    
    /// OpenCL computation errors
    #[error("OpenCL error: {0}")]
    OpenCL(String),
    
    /// Stratum protocol errors
    #[error("Stratum error: {0}")]
    Stratum(#[from] StratumError),
    
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// IO operation errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Network communication errors
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    /// JSON serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// Implement conversion from OCL errors
impl From<ocl::Error> for Graxil29Error {
    fn from(err: ocl::Error) -> Self {
        Graxil29Error::OpenCL(format!("OpenCL operation failed: {}", err))
    }
}

// Implement conversion from hex decode errors
impl From<FromHexError> for Graxil29Error {
    fn from(err: FromHexError) -> Self {
        Graxil29Error::Gpu(format!("Hex decode error: {}", err))
    }
}

/// Result type alias for Graxil29 operations
pub type Result<T> = std::result::Result<T, Graxil29Error>;

/// Application version from Cargo.toml
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Application name from Cargo.toml
pub const NAME: &str = env!("CARGO_PKG_NAME");
/// Application description from Cargo.toml
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Initialize the miner with logging and metrics
pub fn init() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    
    tracing::info!("{} v{} - {}", NAME, VERSION, DESCRIPTION);
    tracing::info!("Initializing Graxil29 miner with OpenCL backend...");
    tracing::info!("🔥 Real Cuckoo Cycle mining - no more mock solutions!");
    
    Ok(())
}