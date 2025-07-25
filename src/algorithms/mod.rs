// src/algorithms/mod.rs - Unified algorithms module for C29 and C32
// Tree location: ./src/algorithms/mod.rs

//! Algorithms module for Graxil29
//! 
//! Provides unified access to Cuckaroo29 and Cuckatoo32 implementations.
//! Defines common types and traits for mining across algorithms.
//! 
//! # Version History
//! - 0.1.0: Initial unified structure
//! - 0.2.0: Added utility methods for main.rs integration
//! - 0.2.1: Fixed naming and format consistency

pub mod cuckaroo29;
pub mod cuckatoo32;

use thiserror::Error;

/// Supported mining algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// ASIC-resistant variant (primary for Grin)
    Cuckaroo29,
    /// ASIC-friendly variant
    Cuckatoo32,
}

impl Algorithm {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Algorithm::Cuckaroo29 => "Cuckaroo29",
            Algorithm::Cuckatoo32 => "Cuckatoo32",
        }
    }

    /// Minimum memory required in GB
    pub fn min_memory_gb(&self) -> usize {
        match self {
            Algorithm::Cuckaroo29 => 6,
            Algorithm::Cuckatoo32 => 11,
        }
    }

    /// Get edge bits parameter
    pub fn edge_bits(&self) -> u32 {
        match self {
            Algorithm::Cuckaroo29 => cuckaroo29::EDGE_BITS,
            Algorithm::Cuckatoo32 => cuckatoo32::EDGE_BITS,
        }
    }

    /// Get graph size
    pub fn graph_size(&self) -> u64 {
        match self {
            Algorithm::Cuckaroo29 => cuckaroo29::GRAPH_SIZE,
            Algorithm::Cuckatoo32 => cuckatoo32::GRAPH_SIZE,
        }
    }

    /// Check if active on Grin mainnet
    pub fn is_mainnet_active(&self) -> bool {
        match self {
            Algorithm::Cuckaroo29 => true,  // Primary
            Algorithm::Cuckatoo32 => true,  // Supported
        }
    }

    /// Detect best algorithm based on available memory (GB)
    pub fn detect_best(memory_gb: usize) -> Self {
        if memory_gb >= 11 {
            Algorithm::Cuckatoo32
        } else if memory_gb >= 6 {
            Algorithm::Cuckaroo29
        } else {
            // Default to Cuckaroo29 if low memory
            Algorithm::Cuckaroo29
        }
    }
}

/// Unified solution type (shared structure between algorithms)
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Nonce that generated the solution
    pub nonce: u64,
    /// 42-cycle as edge indices
    pub cycle: [u32; 42],
}

impl Solution {
    /// Create a new solution
    pub fn new(nonce: u64, cycle: [u32; 42]) -> Self {
        Self { nonce, cycle }
    }

    /// Get difficulty (placeholder; implement per-algorithm if needed)
    pub fn difficulty(&self) -> u32 {
        0
    }
}

/// Required cycle length (shared)
pub const CYCLE_LENGTH: usize = 42;

/// Unified algorithm errors
#[derive(Error, Debug)]
pub enum AlgorithmError {
    /// Cuckaroo29-specific error
    #[error("Cuckaroo29 error: {0}")]
    Cuckaroo29(#[from] cuckaroo29::Cuckaroo29Error),
    /// Cuckatoo32-specific error
    #[error("Cuckatoo32 error: {0}")]
    Cuckatoo32(#[from] cuckatoo32::Cuckatoo32Error),
}

/// Trait for cuckoo mining operations (implemented by per-algorithm structs)
pub trait CuckooMiner {
    /// Create new miner context
    fn new(header_hash: [u8; 32]) -> Self;

    /// Set nonce
    fn set_nonce(&mut self, nonce: u64);

    /// Attempt to solve for a cycle
    fn solve(&self) -> Result<Option<Solution>, AlgorithmError>;

    /// Verify a solution
    fn verify(&self, solution: &Solution) -> Result<bool, AlgorithmError>;
}

// Implement for Cuckaroo29
impl CuckooMiner for cuckaroo29::Cuckaroo29 {
    fn new(header_hash: [u8; 32]) -> Self {
        cuckaroo29::Cuckaroo29::new(header_hash)
    }

    fn set_nonce(&mut self, nonce: u64) {
        self.set_nonce(nonce);
    }

    fn solve(&self) -> Result<Option<Solution>, AlgorithmError> {
        self.solve()
            .map_err(AlgorithmError::from)
            .map(|opt| opt.map(|s| Solution::new(s.nonce, s.cycle)))
    }

    fn verify(&self, solution: &Solution) -> Result<bool, AlgorithmError> {
        let inner_sol = cuckaroo29::Solution::new(solution.nonce, solution.cycle);
        inner_sol.verify(self.header_hash())
            .map_err(AlgorithmError::from)
    }
}

// Implement for Cuckatoo32
impl CuckooMiner for cuckatoo32::Cuckatoo32 {
    fn new(header_hash: [u8; 32]) -> Self {
        cuckatoo32::Cuckatoo32::new(header_hash)
    }

    fn set_nonce(&mut self, nonce: u64) {
        self.set_nonce(nonce);
    }

    fn solve(&self) -> Result<Option<Solution>, AlgorithmError> {
        self.solve()
            .map_err(AlgorithmError::from)
            .map(|opt| opt.map(|s| Solution::new(s.nonce, s.cycle)))
    }

    fn verify(&self, solution: &Solution) -> Result<bool, AlgorithmError> {
        let inner_sol = cuckatoo32::Solution::new(solution.nonce, solution.cycle);
        inner_sol.verify(self.header_hash())
            .map_err(AlgorithmError::from)
    }
}

// Bottom comments: Rewritten with consistent naming (e.g., Cuckaroo29, Cuckatoo32). Added format as per request. No functional changes, but ensured compatibility with main.rs fixes.
// LOC count: 144 (excluding empty lines and this count)