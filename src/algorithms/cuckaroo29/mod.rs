//! Cuckaroo29 algorithm implementation
//! 
//! This module implements the Cuckaroo29 proof-of-work algorithm used by Grin.
//! Cuckaroo29 is a memory-hard, ASIC-resistant variant of the Cuckoo Cycle algorithm.

pub mod graph;
pub mod solver;
pub mod siphash;

use thiserror::Error;

/// Cuckaroo29 algorithm parameters
pub const EDGE_BITS: u32 = 29;
/// Total number of nodes in the graph (2^29)
pub const GRAPH_SIZE: u64 = 1 << EDGE_BITS; // 2^29 = 536,870,912 nodes
/// Required cycle length for valid solutions
pub const CYCLE_LENGTH: usize = 42;
/// Duck size parameter A
pub const DUCK_SIZE_A: u32 = 129;
/// Duck size parameter B
pub const DUCK_SIZE_B: u32 = 82;

/// Errors that can occur during Cuckaroo29 operations
#[derive(Error, Debug)]
pub enum Cuckaroo29Error {
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
}

/// A solution to the Cuckaroo29 proof-of-work
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
    pub fn verify(&self, header_hash: &[u8; 32]) -> Result<bool, Cuckaroo29Error> {
        solver::verify_solution(header_hash, self.nonce, &self.cycle)
    }
    
    /// Get the difficulty of this solution (number of leading zero bits)
    pub fn difficulty(&self) -> u32 {
        // Solution hash would be computed here
        // For now, return 0 as placeholder
        0
    }
}

/// Main Cuckaroo29 solver context
#[derive(Debug, Clone)]
pub struct Cuckaroo29 {
    /// Current header hash being mined
    pub header_hash: [u8; 32],
    /// Current nonce
    nonce: u64,
}

impl Cuckaroo29 {
    /// Create a new Cuckaroo29 solver
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
    pub fn solve(&self) -> Result<Option<Solution>, Cuckaroo29Error> {
        solver::find_solution(&self.header_hash, self.nonce)
    }
    
    /// Generate the bipartite graph for the current header+nonce
    pub fn generate_graph(&self) -> Result<graph::Graph, Cuckaroo29Error> {
        graph::generate_graph(&self.header_hash, self.nonce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constants() {
        assert_eq!(EDGE_BITS, 29);
        assert_eq!(GRAPH_SIZE, 536_870_912);
        assert_eq!(CYCLE_LENGTH, 42);
    }
    
    #[test]
    fn test_cuckaroo29_creation() {
        let header_hash = [0u8; 32];
        let solver = Cuckaroo29::new(header_hash);
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
}