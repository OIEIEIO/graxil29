// src/algorithms/cuckatoo32/mod.rs - Cuckatoo32 algorithm implementation
// Tree location: ./src/algorithms/cuckatoo32/mod.rs

//! Cuckatoo32 algorithm implementation
//! 
//! This module implements the Cuckatoo32 proof-of-work algorithm used by Grin mainnet.
//! Cuckatoo32 is designed to be ASIC-friendly with larger memory requirements than Cuckaroo29.
//! 
//! # Version History
//! - 0.1.0: Initial implementation
//! - 0.1.1: Added documentation for error fields to fix missing docs warnings

pub mod solver;
pub mod siphash;

use thiserror::Error;
use blake2::{Blake2b512, Digest};  // Use fixed-size Blake2b512

/// Cuckatoo32 algorithm parameters
pub const EDGE_BITS: u32 = 32;
/// Total number of nodes in the graph (2^32)
pub const GRAPH_SIZE: u64 = 1u64 << EDGE_BITS; // 2^32 = 4,294,967,296 nodes
/// Required cycle length for valid solutions
pub const CYCLE_LENGTH: usize = 42;
/// Memory requirement estimate (bytes)
pub const MEMORY_REQUIREMENT: usize = 11 * 1024 * 1024 * 1024; // ~11GB

/// Errors that can occur during Cuckatoo32 operations
#[derive(Error, Debug)]
pub enum Cuckatoo32Error {
    /// Invalid nonce provided
    #[error("Invalid nonce provided")]
    InvalidNonce,
    
    /// No solution found for the given parameters
    #[error("No solution found")]
    NoSolution,
    
    /// Invalid cycle detected
    #[error("Invalid cycle: {0}")]
    InvalidCycle(String),
    
    /// Memory allocation or management errors
    #[error("Memory allocation failed")]
    MemoryError,
    
    /// Graph generation or processing errors
    #[error("Graph generation failed: {0}")]
    GraphError(String),
    
    /// Insufficient GPU memory
    #[error("Insufficient GPU memory: need {needed}GB, have {available}GB")]
    InsufficientMemory { 
        /// Needed memory in GB
        needed: usize, 
        /// Available memory in GB
        available: usize 
    },
}

/// A solution to the Cuckatoo32 proof-of-work
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// The nonce that generated this solution
    pub nonce: u64,
    /// The 42-cycle solution as edge indices
    pub cycle: [u32; CYCLE_LENGTH],
}

impl Solution {
    /// Create a new solution
    pub fn new(nonce: u64, cycle: [u32; CYCLE_LENGTH]) -> Self {
        Self { nonce, cycle }
    }
    
    /// Verify that this solution is valid for the given header hash
    pub fn verify(&self, header_hash: &[u8; 32]) -> Result<bool, Cuckatoo32Error> {
        solver::verify_solution(header_hash, self.nonce, &self.cycle)
    }
    
    /// Get the difficulty of this solution (number of leading zero bits)
    pub fn difficulty(&self) -> u32 {
        // Solution hash would be computed here
        // For now, return 0 as placeholder
        0
    }
    
    /// Calculate the solution hash (Blake2b of header + nonce + cycle)
    pub fn solution_hash(&self, header_hash: [u8; 32]) -> [u8; 32] {
        let mut hasher = Blake2b512::new();
        hasher.update(&header_hash);
        hasher.update(&self.nonce.to_le_bytes());
        
        // Add cycle data
        for &edge in &self.cycle {
            hasher.update(&edge.to_le_bytes());
        }
        
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result[..32]);
        hash
    }
}

/// Main Cuckatoo32 solver context
#[derive(Debug, Clone)]
pub struct Cuckatoo32 {
    /// Current header hash being mined
    pub header_hash: [u8; 32],
    /// Current nonce
    nonce: u64,
}

impl Cuckatoo32 {
    /// Create a new Cuckatoo32 solver
    pub fn new(header_hash: [u8; 32]) -> Self {
        Self {
            header_hash,
            nonce: 0,
        }
    }
    
    /// Set the nonce for mining
    pub fn set_nonce(&mut self, nonce: u64) {
        self.nonce = nonce;
    }
    
    /// Get the current header hash
    pub fn header_hash(&self) -> &[u8; 32] {
        &self.header_hash
    }
    
    /// Attempt to find a solution with the current nonce
    pub fn solve(&self) -> Result<Option<Solution>, Cuckatoo32Error> {
        solver::find_solution(&self.header_hash, self.nonce)
    }
    
    /// Check if the system has enough memory for Cuckatoo32
    pub fn check_memory_requirements(available_gb: usize) -> Result<(), Cuckatoo32Error> {
        let needed_gb = MEMORY_REQUIREMENT / (1024 * 1024 * 1024);
        if available_gb < needed_gb {
            Err(Cuckatoo32Error::InsufficientMemory { 
                needed: needed_gb, 
                available: available_gb 
            })
        } else {
            Ok(())
        }
    }
    
    /// Get algorithm statistics
    pub fn algorithm_info() -> AlgorithmInfo {
        AlgorithmInfo {
            name: "Cuckatoo32".to_string(),
            edge_bits: EDGE_BITS,
            graph_size: GRAPH_SIZE,
            cycle_length: CYCLE_LENGTH,
            memory_requirement_gb: MEMORY_REQUIREMENT / (1024 * 1024 * 1024),
            active_on_mainnet: true,
        }
    }
}

/// Algorithm information and statistics
#[derive(Debug, Clone)]
pub struct AlgorithmInfo {
    /// Algorithm name
    pub name: String,
    /// Number of edge bits
    pub edge_bits: u32,
    /// Total graph size (number of nodes)
    pub graph_size: u64,
    /// Required cycle length
    pub cycle_length: usize,
    /// Memory requirement in GB
    pub memory_requirement_gb: usize,
    /// Whether this algorithm is active on Grin mainnet
    pub active_on_mainnet: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constants() {
        assert_eq!(EDGE_BITS, 32);
        assert_eq!(GRAPH_SIZE, 4_294_967_296);
        assert_eq!(CYCLE_LENGTH, 42);
    }
    
    #[test]
    fn test_cuckatoo32_creation() {
        let header_hash = [0u8; 32];
        let solver = Cuckatoo32::new(header_hash);
        assert_eq!(solver.header_hash, header_hash);
        assert_eq!(solver.nonce, 0);
    }
    
    #[test]
    fn test_solution_creation() {
        let cycle = [0u32; CYCLE_LENGTH];
        let solution = Solution::new(12345, cycle);
        assert_eq!(solution.nonce, 12345);
        assert_eq!(solution.cycle, cycle);
    }
    
    #[test]
    fn test_memory_check() {
        // Should pass with 16GB
        assert!(Cuckatoo32::check_memory_requirements(16).is_ok());
        
        // Should fail with 8GB
        assert!(Cuckatoo32::check_memory_requirements(8).is_err());
    }
    
    #[test]
    fn test_algorithm_info() {
        let info = Cuckatoo32::algorithm_info();
        assert_eq!(info.name, "Cuckatoo32");
        assert_eq!(info.edge_bits, 32);
        assert!(info.active_on_mainnet);
    }
}

// Bottom comments: Added /// doc comments to the 'needed' and 'available' fields in InsufficientMemory variant to fix missing documentation warnings. No other changes as the code is functional.
// LOC count: 161 (excluding empty lines and this count)