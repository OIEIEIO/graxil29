//! SipHash implementation for Cuckaroo29
//!
//! This module provides the SipHash-2-4 implementation used by Grin's Cuckaroo29 algorithm.
//! Based on the reference implementation, now using BLAKE2b-256 for SipHash key derivation.

use blake2::{Blake2bVar, digest::{Update, VariableOutput}};

/// SipHash state for edge generation
#[derive(Clone, Debug)]
pub struct SipHasher {
    k0: u64,
    k1: u64,
}

impl SipHasher {
    /// Create a new SipHasher from header hash and nonce
    pub fn new(header_hash: &[u8; 32], nonce: u64) -> Self {
        // Use BLAKE2b-256 to create SipHash key
        let mut hasher = Blake2bVar::new(32).unwrap(); // 32 bytes = 256 bits
        hasher.update(header_hash);
        hasher.update(&nonce.to_le_bytes());
        let mut hash = [0u8; 32];
        hasher.finalize_variable(&mut hash).unwrap();

        let k0 = u64::from_le_bytes(hash[0..8].try_into().unwrap());
        let k1 = u64::from_le_bytes(hash[8..16].try_into().unwrap());
        Self { k0, k1 }
    }

    /// Generate edge endpoints for given edge index
    /// Returns (u_node, v_node) where u_node ∈ [0, 2^28) and v_node ∈ [2^28, 2^29)
    pub fn hash_edge(&self, edge: u64) -> (u32, u32) {
        let hash = self.siphash24(edge);

        // Split hash into two parts for bipartite graph
        let u_node = (hash & 0x0FFF_FFFF) as u32; // Lower 28 bits for U side
        let v_node = ((hash >> 32) & 0x0FFF_FFFF) as u32 | (1 << 28); // Upper 28 bits + offset for V side

        (u_node, v_node)
    }

    /// SipHash-2-4 implementation
    fn siphash24(&self, input: u64) -> u64 {
        let mut v0 = 0x736f6d6570736575u64 ^ self.k0;
        let mut v1 = 0x646f72616e646f6du64 ^ self.k1;
        let mut v2 = 0x6c7967656e657261u64 ^ self.k0;
        let mut v3 = 0x7465646279746573u64 ^ self.k1;

        // Process input
        v3 ^= input;

        // 2 rounds of SipRound
        for _ in 0..2 {
            sipround(&mut v0, &mut v1, &mut v2, &mut v3);
        }

        v0 ^= input;
        v2 ^= 0xff;

        // 4 rounds of SipRound
        for _ in 0..4 {
            sipround(&mut v0, &mut v1, &mut v2, &mut v3);
        }

        v0 ^ v1 ^ v2 ^ v3
    }
}

/// Single round of SipHash
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_siphasher_creation() {
        let header_hash = [0x42u8; 32];
        let nonce = 12345u64;
        let hasher = SipHasher::new(&header_hash, nonce);

        // Keys should be deterministic for same input
        let hasher2 = SipHasher::new(&header_hash, nonce);
        assert_eq!(hasher.k0, hasher2.k0);
        assert_eq!(hasher.k1, hasher2.k1);
    }

    #[test]
    fn test_edge_generation() {
        let header_hash = [0u8; 32];
        let hasher = SipHasher::new(&header_hash, 0);

        let (u, v) = hasher.hash_edge(0);

        // U node should be in range [0, 2^28)
        assert!(u < (1 << 28));

        // V node should be in range [2^28, 2^29)
        assert!(v >= (1 << 28));
        assert!(v < (1 << 29));
    }

    #[test]
    fn test_edge_determinism() {
        let header_hash = [0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78,
                          0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78,
                          0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78,
                          0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78];
        let hasher = SipHasher::new(&header_hash, 999);

        // Same edge should always produce same result
        let (u1, v1) = hasher.hash_edge(42);
        let (u2, v2) = hasher.hash_edge(42);

        assert_eq!(u1, u2);
        assert_eq!(v1, v2);

        // Different edges should produce different results
        let (u3, v3) = hasher.hash_edge(43);
        assert!((u1, v1) != (u3, v3));
    }

    #[test]
    fn test_siphash_properties() {
        let header_hash = [0u8; 32];
        let hasher = SipHasher::new(&header_hash, 0);

        // Test that different inputs give different outputs
        let h1 = hasher.siphash24(1);
        let h2 = hasher.siphash24(2);
        assert_ne!(h1, h2);

        // Test determinism
        let h3 = hasher.siphash24(1);
        assert_eq!(h1, h3);
    }
}
