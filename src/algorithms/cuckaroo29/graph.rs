//! Graph generation for Cuckaroo29
//! 
//! This module handles the generation of the bipartite graph used in Cuckaroo29.
//! The graph has 2^29 potential edges connecting two sets of 2^28 nodes each.

use super::{siphash::SipHasher, Cuckaroo29Error, EDGE_BITS, GRAPH_SIZE};
use std::collections::HashMap;

/// Maximum number of edges to store in memory (6GB limit for ~6GB GPUs)
const MAX_EDGES: usize = 1_000_000; // Adjust based on available memory

/// A single edge in the bipartite graph
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    /// Source node (U side: 0..2^28)
    pub u: u32,
    /// Destination node (V side: 2^28..2^29)  
    pub v: u32,
    /// Original edge index
    pub index: u32,
}

impl Edge {
    /// Create a new edge
    pub fn new(u: u32, v: u32, index: u32) -> Self {
        Self { u, v, index }
    }
}

/// Bipartite graph representation
pub struct Graph {
    /// All edges in the graph after trimming
    pub edges: Vec<Edge>,
    /// Adjacency lists for U nodes (node_id -> list of edge indices)
    pub u_adj: HashMap<u32, Vec<usize>>,
    /// Adjacency lists for V nodes (node_id -> list of edge indices)  
    pub v_adj: HashMap<u32, Vec<usize>>,
    /// Number of nodes on each side
    pub node_count: u32,
}

impl Graph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            u_adj: HashMap::new(),
            v_adj: HashMap::new(),
            node_count: 1 << (EDGE_BITS - 1), // 2^28 nodes per side
        }
    }
    
    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: Edge) {
        let edge_idx = self.edges.len();
        self.edges.push(edge);
        
        // Add to adjacency lists
        self.u_adj.entry(edge.u).or_default().push(edge_idx);
        self.v_adj.entry(edge.v).or_default().push(edge_idx);
    }
    
    /// Get the degree of a U node
    pub fn u_degree(&self, node: u32) -> usize {
        self.u_adj.get(&node).map_or(0, |adj| adj.len())
    }
    
    /// Get the degree of a V node  
    pub fn v_degree(&self, node: u32) -> usize {
        self.v_adj.get(&node).map_or(0, |adj| adj.len())
    }
    
    /// Remove edges connected to low-degree nodes (trimming)
    pub fn trim(&mut self, min_degree: usize) -> usize {
        let mut removed_count = 0;
        let mut changed = true;
        
        while changed {
            changed = false;
            let mut edges_to_remove = Vec::new();
            
            // Find edges connected to low-degree nodes
            for (i, edge) in self.edges.iter().enumerate() {
                if self.u_degree(edge.u) < min_degree || self.v_degree(edge.v) < min_degree {
                    edges_to_remove.push(i);
                }
            }
            
            if !edges_to_remove.is_empty() {
                // Remove edges (in reverse order to maintain indices)
                for &idx in edges_to_remove.iter().rev() {
                    self.remove_edge(idx);
                    removed_count += 1;
                    changed = true;
                }
            }
        }
        
        removed_count
    }
    
    /// Remove an edge by index
    fn remove_edge(&mut self, idx: usize) {
        if idx >= self.edges.len() {
            return;
        }
        
        let edge = self.edges[idx];
        
        // Remove from adjacency lists
        if let Some(adj) = self.u_adj.get_mut(&edge.u) {
            adj.retain(|&e_idx| e_idx != idx);
            if adj.is_empty() {
                self.u_adj.remove(&edge.u);
            }
        }
        
        if let Some(adj) = self.v_adj.get_mut(&edge.v) {
            adj.retain(|&e_idx| e_idx != idx);
            if adj.is_empty() {
                self.v_adj.remove(&edge.v);
            }
        }
        
        // Remove edge and update indices
        self.edges.remove(idx);
        
        // Update all edge indices in adjacency lists
        for adj_list in self.u_adj.values_mut() {
            for edge_idx in adj_list.iter_mut() {
                if *edge_idx > idx {
                    *edge_idx -= 1;
                }
            }
        }
        
        for adj_list in self.v_adj.values_mut() {
            for edge_idx in adj_list.iter_mut() {
                if *edge_idx > idx {
                    *edge_idx -= 1;
                }
            }
        }
    }
    
    /// Get statistics about the graph
    pub fn stats(&self) -> GraphStats {
        let mut u_degrees = Vec::new();
        let mut v_degrees = Vec::new();
        
        for degrees in self.u_adj.values() {
            u_degrees.push(degrees.len());
        }
        
        for degrees in self.v_adj.values() {
            v_degrees.push(degrees.len());
        }
        
        GraphStats {
            edge_count: self.edges.len(),
            u_node_count: self.u_adj.len(),
            v_node_count: self.v_adj.len(),
            avg_u_degree: if u_degrees.is_empty() { 0.0 } else { 
                u_degrees.iter().sum::<usize>() as f64 / u_degrees.len() as f64 
            },
            avg_v_degree: if v_degrees.is_empty() { 0.0 } else {
                v_degrees.iter().sum::<usize>() as f64 / v_degrees.len() as f64
            },
        }
    }
}

/// Graph statistics
#[derive(Debug)]
    /// Number of edges in the graph
pub struct GraphStats {
    /// Number of U nodes
    pub edge_count: usize,
    /// Number of V nodes
    pub u_node_count: usize,
    /// Average degree of U nodes
    pub v_node_count: usize,
    /// Average degree of V nodes
    pub avg_u_degree: f64,
    /// Average degree of V nodes in the graph
    pub avg_v_degree: f64,
}

/// Generate the initial bipartite graph from header hash and nonce
pub fn generate_graph(header_hash: &[u8; 32], nonce: u64) -> Result<Graph, Cuckaroo29Error> {
    let hasher = SipHasher::new(header_hash, nonce);
    let mut graph = Graph::new();
    
    // Generate edges up to memory limit
    let edge_limit = std::cmp::min(GRAPH_SIZE as usize, MAX_EDGES);
    
    for edge_idx in 0..edge_limit {
        let (u, v) = hasher.hash_edge(edge_idx as u64);
        let edge = Edge::new(u, v, edge_idx as u32);
        graph.add_edge(edge);
    }
    
    Ok(graph)
}

/// Perform iterative trimming to reduce graph size
pub fn trim_graph(graph: &mut Graph, target_edges: usize) -> Result<(), Cuckaroo29Error> {
    let mut trim_rounds = 0;
    const MAX_TRIM_ROUNDS: usize = 100;
    
    while graph.edges.len() > target_edges && trim_rounds < MAX_TRIM_ROUNDS {
        let initial_count = graph.edges.len();
        
        // Adaptive trimming: higher min_degree when we need more aggressive trimming
        let min_degree = if graph.edges.len() > target_edges * 4 {
            3
        } else if graph.edges.len() > target_edges * 2 {
            2
        } else {
            1
        };
        
        let removed = graph.trim(min_degree);
        trim_rounds += 1;
        
        // Break if no progress
        if removed == 0 || graph.edges.len() == initial_count {
            break;
        }
    }
    
    if graph.edges.len() > target_edges * 2 {
        return Err(Cuckaroo29Error::GraphError(
            format!("Failed to trim graph to reasonable size: {} edges remaining", graph.edges.len())
        ));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_edge_creation() {
        let edge = Edge::new(100, 268435556, 42); // V node = 100 + 2^28
        assert_eq!(edge.u, 100);
        assert_eq!(edge.v, 268435556);
        assert_eq!(edge.index, 42);
    }
    
    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::new();
        assert_eq!(graph.edges.len(), 0);
        assert_eq!(graph.node_count, 1 << 28);
        
        let edge = Edge::new(0, 1 << 28, 0);
        graph.add_edge(edge);
        
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.u_degree(0), 1);
        assert_eq!(graph.v_degree(1 << 28), 1);
    }
    
    #[test]
    fn test_graph_generation() {
        let header_hash = [0u8; 32];
        let nonce = 12345;
        
        let result = generate_graph(&header_hash, nonce);
        assert!(result.is_ok());
        
        let graph = result.unwrap();
        assert!(graph.edges.len() > 0);
        
        // Check that all edges have valid node ranges
        for edge in &graph.edges {
            assert!(edge.u < (1 << 28));
            assert!(edge.v >= (1 << 28));
            assert!(edge.v < (1 << 29));
        }
    }
    
    #[test]
    fn test_graph_trimming() {
        let mut graph = Graph::new();
        
        // Add some edges forming a small structure
        graph.add_edge(Edge::new(0, 1 << 28, 0));
        graph.add_edge(Edge::new(0, (1 << 28) + 1, 1));
        graph.add_edge(Edge::new(1, 1 << 28, 2));
        
        let initial_count = graph.edges.len();
        let removed = graph.trim(2); // Remove nodes with degree < 2
        
        assert!(removed > 0 || graph.edges.len() < initial_count);
    }
    
    #[test]
    fn test_graph_stats() {
        let mut graph = Graph::new();
        graph.add_edge(Edge::new(0, 1 << 28, 0));
        graph.add_edge(Edge::new(0, (1 << 28) + 1, 1));
        
        let stats = graph.stats();
        assert_eq!(stats.edge_count, 2);
        assert_eq!(stats.u_node_count, 1);
        assert_eq!(stats.v_node_count, 2);
        assert_eq!(stats.avg_u_degree, 2.0);
        assert_eq!(stats.avg_v_degree, 1.0);
    }
}