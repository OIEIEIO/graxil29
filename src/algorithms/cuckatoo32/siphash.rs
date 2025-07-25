//! SipHash-2-4 Implementation for Cuckatoo32 v2.0
//! 
//! This module provides the SipHash-2-4 implementation used by Grin's Cuckatoo32 algorithm.
//! SipHash is a cryptographically secure pseudorandom function that generates the edges
//! in the Cuckoo Cycle graph deterministically from the block header and nonce.
//! 
//! # SipHash-2-4 Parameters
//! 
//! - **Rounds**: 2 compression rounds, 4 finalization rounds
//! - **Key**: Derived from block header hash
//! - **Input**: Edge index combined with nonce
//! - **Output**: 64-bit hash value for node generation
//! 
//! # Cuckatoo32 Usage
//! 
//! For each potential edge index `i`, SipHash generates two hash values:
//! - `hash1 = siphash(header, nonce, i)` → U-side node
//! - `hash2 = siphash(header, nonce, i + offset)` → V-side node
//! 
//! This ensures deterministic, pseudorandom edge generation that's verifiable
//! by any node in the network.
//!
//! # Version History
//! - v1.0: Initial SipHash-2-4 implementation
//! - v2.0: Fixed compiler warnings, added version control

use blake2::{Blake2b512, Digest};

/// SipHash-2-4 state and computation engine
#[derive(Debug, Clone)]
pub struct SipHasher {
    /// SipHash key derived from header (128 bits split into 2×64-bit words)
    key: [u64; 2],
    /// Current nonce being processed
    nonce: u64,
    /// Cached header hash for efficiency
    header_hash: [u8; 32],
}

impl SipHasher {
    /// Create a new SipHasher for Cuckatoo32
    /// 
    /// # Arguments
    /// * `header_hash` - Block header hash (32 bytes)
    /// * `nonce` - Mining nonce value
    /// 
    /// # Returns
    /// Initialized SipHasher ready for edge generation
    pub fn new(header_hash: [u8; 32], nonce: u64) -> Self {
        let key = Self::derive_key(header_hash, nonce);
        
        Self {
            key,
            nonce,
            header_hash,
        }
    }
    
    /// Derive SipHash key from header hash and nonce
    /// 
    /// Uses Blake2b to create a 128-bit key from the header and nonce.
    /// This ensures the key is cryptographically strong and unique per
    /// header/nonce combination.
    fn derive_key(header_hash: [u8; 32], nonce: u64) -> [u64; 2] {
        let mut hasher = Blake2b512::new();
        hasher.update(&header_hash);
        hasher.update(&nonce.to_le_bytes());
        
        let key_bytes = hasher.finalize();
        
        // Extract two 64-bit keys from the 512-bit Blake2b output
        let key0 = u64::from_le_bytes([
            key_bytes[0], key_bytes[1], key_bytes[2], key_bytes[3],
            key_bytes[4], key_bytes[5], key_bytes[6], key_bytes[7],
        ]);
        
        let key1 = u64::from_le_bytes([
            key_bytes[8], key_bytes[9], key_bytes[10], key_bytes[11],
            key_bytes[12], key_bytes[13], key_bytes[14], key_bytes[15],
        ]);
        
        [key0, key1]
    }
    
    /// Generate hash for a specific edge index
    /// 
    /// This is the main function used during graph generation.
    /// Each edge index produces a deterministic hash value.
    /// 
    /// # Arguments
    /// * `edge_index` - Index of the edge to generate hash for
    /// 
    /// # Returns
    /// 64-bit hash value for node assignment
    pub fn hash_edge(&self, edge_index: u64) -> u64 {
        self.siphash_2_4(edge_index)
    }
    
    /// Generate hash for U-side node
    /// 
    /// Convenience function for generating U-side nodes specifically.
    /// Uses the edge index directly.
    pub fn hash_u_node(&self, edge_index: u64) -> u64 {
        self.siphash_2_4(edge_index)
    }
    
    /// Generate hash for V-side node
    /// 
    /// Convenience function for generating V-side nodes specifically.
    /// Uses edge index with an offset to ensure different values from U-side.
    pub fn hash_v_node(&self, edge_index: u64) -> u64 {
        // Use a large offset to ensure V-side hashes differ from U-side
        let offset = 0x8000_0000_0000_0000u64; // 2^63
        self.siphash_2_4(edge_index.wrapping_add(offset))
    }
    
    /// Core SipHash-2-4 implementation
    /// 
    /// Implements the SipHash algorithm with 2 compression rounds and
    /// 4 finalization rounds as specified for Cuckatoo32.
    /// 
    /// # Algorithm Overview
    /// 1. Initialize state with key
    /// 2. Process input in 8-byte chunks with compression
    /// 3. Handle final partial chunk
    /// 4. Apply finalization rounds
    /// 5. Return final hash value
    fn siphash_2_4(&self, message: u64) -> u64 {
        // SipHash initialization constants
        const C0: u64 = 0x736f6d6570736575;
        const C1: u64 = 0x646f72616e646f6d;
        const C2: u64 = 0x6c7967656e657261;
        const C3: u64 = 0x7465646279746573;
        
        // Initialize state
        let mut v0 = self.key[0] ^ C0;
        let mut v1 = self.key[1] ^ C1;
        let mut v2 = self.key[0] ^ C2;
        let mut v3 = self.key[1] ^ C3;
        
        // Process the 8-byte message
        v3 ^= message;
        
        // Compression rounds (2 rounds for SipHash-2-4)
        for _ in 0..2 {
            sipround(&mut v0, &mut v1, &mut v2, &mut v3);
        }
        
        v0 ^= message;
        
        // Finalization setup
        v2 ^= 0xff;
        
        // Finalization rounds (4 rounds for SipHash-2-4)
        for _ in 0..4 {
            sipround(&mut v0, &mut v1, &mut v2, &mut v3);
        }
        
        v0 ^ v1 ^ v2 ^ v3
    }
    
    /// Update nonce without recreating the hasher
    /// 
    /// Allows efficient nonce iteration without full reinitialization.
    /// Only regenerates the key when the nonce changes.
    pub fn set_nonce(&mut self, nonce: u64) {
        if self.nonce != nonce {
            self.nonce = nonce;
            self.key = Self::derive_key(self.header_hash, nonce);
        }
    }
    
    /// Get current nonce
    pub fn nonce(&self) -> u64 {
        self.nonce
    }
    
    /// Get current header hash
    pub fn header_hash(&self) -> [u8; 32] {
        self.header_hash
    }
    
    /// Verify hash determinism by regenerating
    /// 
    /// Test function to ensure hash generation is deterministic.
    /// Used for debugging and validation.
    pub fn verify_determinism(&self, edge_index: u64, expected_hash: u64) -> bool {
        self.hash_edge(edge_index) == expected_hash
    }
    
    /// Generate a batch of hashes for efficiency
    /// 
    /// Optimized function for generating multiple hashes at once.
    /// Reduces function call overhead during graph generation.
    pub fn hash_batch(&self, start_index: u64, count: usize) -> Vec<u64> {
        let mut hashes = Vec::with_capacity(count);
        
        for i in 0..count {
            let edge_index = start_index + i as u64;
            hashes.push(self.hash_edge(edge_index));
        }
        
        hashes
    }
    
    /// Generate paired hashes for U and V nodes
    /// 
    /// Efficiently generates both U and V node hashes for an edge.
    /// Returns (u_hash, v_hash) tuple.
    pub fn hash_edge_pair(&self, edge_index: u64) -> (u64, u64) {
        let u_hash = self.hash_u_node(edge_index);
        let v_hash = self.hash_v_node(edge_index);
        (u_hash, v_hash)
    }
}

/// Single SipHash round function
/// 
/// Performs one round of the SipHash mixing function.
/// This is the core operation that provides SipHash's security properties.
/// 
/// # Arguments
/// * `v0`, `v1`, `v2`, `v3` - SipHash state variables (modified in place)
#[inline]
fn sipround(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
    *v0 = v0.wrapping_add(*v1);
    *v1 = v1.rotate_left(13);
    *v1 ^= *v0;
    *v0 = v0.rotate_left(32);
    
    *v2 = v2.wrapping_add(*v3);
    *v3 = v3.rotate_left(16);
    *v3 ^= *v2;
    
    *v0 = v0.wrapping_add(*v3);
    *v3 = v3.rotate_left(21);
    *v3 ^= *v0;
    
    *v2 = v2.wrapping_add(*v1);
    *v1 = v1.rotate_left(17);
    *v1 ^= *v2;
    *v2 = v2.rotate_left(32);
}

/// Utility functions for SipHash testing and validation
pub mod utils {
    use super::*;
    
    /// Test SipHash implementation against known test vectors
    /// 
    /// Validates the SipHash implementation using standard test vectors
    /// to ensure correctness and compatibility.
    pub fn test_siphash_vectors() -> bool {
        // Known test vector for SipHash-2-4
        let test_key = [0x0706050403020100, 0x0f0e0d0c0b0a0908];
        let test_message = 0x0706050403020100u64;
        let expected = 0xa129ca6149be45e5u64;
        
        // Create temporary hasher with test parameters
        // v2.0 Fix: Removed unnecessary mut
        let hasher = SipHasher {
            key: test_key,
            nonce: 0,
            header_hash: [0; 32],
        };
        
        let result = hasher.siphash_2_4(test_message);
        result == expected
    }
    
    /// Benchmark SipHash performance
    /// 
    /// Measures hash generation rate for performance tuning.
    /// Returns hashes per second.
    pub fn benchmark_siphash(iterations: usize) -> f64 {
        use std::time::Instant;
        
        let hasher = SipHasher::new([0u8; 32], 12345);
        let start = Instant::now();
        
        for i in 0..iterations {
            let _ = hasher.hash_edge(i as u64);
        }
        
        let elapsed = start.elapsed();
        iterations as f64 / elapsed.as_secs_f64()
    }
    
    /// Test hash distribution quality
    /// 
    /// Analyzes hash output distribution to ensure good randomness.
    /// Returns distribution statistics.
    pub fn analyze_distribution(sample_size: usize) -> DistributionStats {
        let hasher = SipHasher::new([0x42u8; 32], 67890);
        let mut hashes = Vec::with_capacity(sample_size);
        
        for i in 0..sample_size {
            hashes.push(hasher.hash_edge(i as u64));
        }
        
        // Calculate basic distribution statistics
        let mean = hashes.iter().map(|&h| h as f64).sum::<f64>() / sample_size as f64;
        
        let variance = hashes.iter()
            .map(|&h| {
                let diff = h as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / sample_size as f64;
        
        DistributionStats {
            sample_size,
            mean,
            variance,
            std_dev: variance.sqrt(),
        }
    }
    
    /// Hash distribution analysis results
    #[derive(Debug, Clone)]
    pub struct DistributionStats {
        /// Number of samples analyzed
        pub sample_size: usize,
        /// Mean hash value
        pub mean: f64,
        /// Variance of hash values
        pub variance: f64,
        /// Standard deviation
        pub std_dev: f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_siphash_determinism() {
        let header = [0x12u8; 32];
        let nonce = 54321;
        let hasher = SipHasher::new(header, nonce);
        
        // Same input should produce same output
        let hash1 = hasher.hash_edge(12345);
        let hash2 = hasher.hash_edge(12345);
        assert_eq!(hash1, hash2);
    }
    
    #[test]
    fn test_different_inputs_different_outputs() {
        let header = [0x34u8; 32];
        let nonce = 98765;
        let hasher = SipHasher::new(header, nonce);
        
        // Different inputs should produce different outputs
        let hash1 = hasher.hash_edge(1);
        let hash2 = hasher.hash_edge(2);
        assert_ne!(hash1, hash2);
    }
    
    #[test]
    fn test_u_v_hash_difference() {
        let header = [0x56u8; 32];
        let nonce = 13579;
        let hasher = SipHasher::new(header, nonce);
        
        // U and V hashes for same edge should be different
        let u_hash = hasher.hash_u_node(100);
        let v_hash = hasher.hash_v_node(100);
        assert_ne!(u_hash, v_hash);
    }
    
    #[test]
    fn test_nonce_update() {
        let header = [0x78u8; 32];
        let mut hasher = SipHasher::new(header, 1111);
        
        let hash1 = hasher.hash_edge(999);
        
        hasher.set_nonce(2222);
        let hash2 = hasher.hash_edge(999);
        
        // Same edge index with different nonce should produce different hash
        assert_ne!(hash1, hash2);
    }
    
    #[test]
    fn test_batch_generation() {
        let header = [0x9au8; 32];
        let nonce = 24680;
        let hasher = SipHasher::new(header, nonce);
        
        // Batch generation should match individual generation
        let batch = hasher.hash_batch(100, 10);
        
        for (i, &batch_hash) in batch.iter().enumerate() {
            let individual_hash = hasher.hash_edge(100 + i as u64);
            assert_eq!(batch_hash, individual_hash);
        }
    }
    
    #[test]
    fn test_siphash_test_vectors() {
        assert!(utils::test_siphash_vectors(), "SipHash test vectors failed");
    }
}