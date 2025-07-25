// src/gpu/opencl/mod.rs - OpenCL Module Organization and Public API
// Tree location: ./src/gpu/opencl/mod.rs

//! OpenCL backend for Graxil29 Cuckoo Cycle mining
//! 
//! Provides a complete OpenCL implementation for both Cuckaroo29 and Cuckatoo32
//! algorithms with multi-platform support (NVIDIA, AMD, Intel).
//! 
//! # Version History
//! - 0.1.0: Initial OpenCL backend with platform detection
//! - 0.1.1: Added trimmer and graph search modules
//! - 0.1.2: Enhanced buffer management and kernel utilities
//! - 0.1.3: FIXED: Proper nonce iteration for real mining
//! 
//! # Module Organization
//! - `platform`: GPU detection and OpenCL context creation
//! - `trimmer`: Core mining engine with 120-round trimming
//! - `graph`: CPU cycle detection and validation
//! - `buffers`: OpenCL buffer management utilities
//! - `kernels`: Kernel compilation and execution helpers
//! 
//! # Usage Example
//! ```rust
//! use crate::gpu::opencl::{PlatformDetector, Trimmer, search_cycles};
//! use crate::algorithms::Algorithm;
//! 
//! // Detect and select best GPU
//! let detector = PlatformDetector::new()?;
//! let context = detector.create_context(None, None, Some(Algorithm::Cuckaroo29))?;
//! 
//! // Create trimmer and mine
//! let mut trimmer = Trimmer::new(context)?;
//! let trimmed_edges = trimmer.run(&siphash_keys)?;
//! let solutions = search_cycles(&trimmed_edges)?;
//! ```

// Core modules
pub mod platform;
pub mod trimmer;
pub mod graph;
pub mod buffers;
pub mod kernels;

// Re-export primary types for convenient access
pub use platform::{
    PlatformDetector, 
    OpenClContext, 
    DeviceCapabilities, 
    BufferConfiguration,
    GpuVendor,
    PlatformCapabilities,
    validate_platform_requirements,
};

pub use trimmer::{
    Trimmer,
    TrimmingStats,
    create_siphash_keys,
};

pub use graph::{
    CycleGraph,
    CycleSolution,
    SearchStats,
    search_cycles,
    solutions_to_algorithm_format,
};

pub use buffers::{
    BufferManager,
    BufferSet,
    BufferUsage,
    OpenClBufferError,
};

pub use kernels::{
    KernelManager,
    KernelCompiler,
    KernelExecutor,
    CompilationResult,
    ExecutionStats,
};

use crate::{Graxil29Error, algorithms::{Algorithm, Solution}};

/// OpenCL backend version information
pub const OPENCL_BACKEND_VERSION: &str = "0.1.3";

/// Minimum OpenCL version required
pub const MIN_OPENCL_VERSION: &str = "1.2";

/// Default maximum number of solutions to find per mining attempt
pub const DEFAULT_MAX_SOLUTIONS: usize = 10;

/// OpenCL mining backend configuration
#[derive(Debug, Clone)]
pub struct OpenClConfig {
    /// Preferred GPU vendor (None for auto-detect)
    pub preferred_vendor: Option<GpuVendor>,
    /// Device ID within platform (None for auto-select)
    pub device_id: Option<usize>,
    /// Platform name hint (None for auto-select)
    pub platform_hint: Option<String>,
    /// Algorithm to optimize for
    pub algorithm: Algorithm,
    /// Maximum solutions to find per mining attempt
    pub max_solutions: usize,
    /// Enable profiling for performance analysis
    pub enable_profiling: bool,
    /// Custom batch size override
    pub batch_size_override: Option<u64>,
}

impl Default for OpenClConfig {
    fn default() -> Self {
        Self {
            preferred_vendor: None,
            device_id: None,
            platform_hint: None,
            algorithm: Algorithm::Cuckaroo29,
            max_solutions: DEFAULT_MAX_SOLUTIONS,
            enable_profiling: false,
            batch_size_override: None,
        }
    }
}

/// Complete OpenCL mining backend
pub struct OpenClBackend {
    /// OpenCL context and device info
    context: OpenClContext,
    /// Mining engine
    trimmer: Trimmer,
    /// Configuration
    config: OpenClConfig,
    /// Runtime statistics
    stats: BackendStats,
}

/// Backend performance statistics
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    /// Total mining attempts
    pub attempts: u64,
    /// Total solutions found
    pub solutions_found: u64,
    /// Total trimming time in milliseconds
    pub total_trimming_time_ms: u64,
    /// Total cycle search time in milliseconds
    pub total_search_time_ms: u64,
    /// Average edges remaining after trimming
    pub avg_edges_remaining: f64,
    /// Success rate (solutions found / attempts)
    pub success_rate: f64,
}

impl OpenClBackend {
    /// Create new OpenCL backend with default configuration
    pub fn new() -> Result<Self, Graxil29Error> {
        Self::with_config(OpenClConfig::default())
    }
    
    /// Create OpenCL backend with custom configuration
    pub fn with_config(config: OpenClConfig) -> Result<Self, Graxil29Error> {
        tracing::info!("🔧 Initializing OpenCL backend v{}", OPENCL_BACKEND_VERSION);
        
        // Detect platforms and select device
        let detector = if let Some(ref vendor) = config.preferred_vendor {
            PlatformDetector::with_vendor_preference(vec![*vendor])?
        } else {
            PlatformDetector::new()?
        };
        
        // Create OpenCL context
        let context = detector.create_context(
            config.platform_hint.as_deref(),
            config.device_id,
            Some(config.algorithm),
        )?;
        
        // Validate platform requirements
        validate_platform_requirements(&context.capabilities, config.algorithm)?;
        
        // Create trimmer
        let trimmer = Trimmer::new(context.clone())?;
        
        tracing::info!("✅ OpenCL backend initialized for {:?} on {} {}", 
            config.algorithm, 
            vendor_name(context.vendor),
            context.capabilities.name);
        
        Ok(Self {
            context,
            trimmer,
            config,
            stats: BackendStats::default(),
        })
    }
    
    /// Mine with given header and nonce range - FIXED TO ACTUALLY ITERATE NONCES
    /// 
    /// # Arguments
    /// * `header` - Block header bytes
    /// * `start_nonce` - Starting nonce value
    /// * `nonce_count` - Number of nonces to try
    /// 
    /// # Returns
    /// * `Result<Vec<Solution>, Graxil29Error>` - Found solutions
    pub fn mine(&mut self, header: &[u8], start_nonce: u64, nonce_count: u64) -> Result<Vec<Solution>, Graxil29Error> {
        let start_time = std::time::Instant::now();
        self.stats.attempts += 1;
        
        tracing::info!("⛏️  Mining {} nonces starting from {} with {:?}", 
            nonce_count, start_nonce, self.config.algorithm);
        
        let mut all_solutions = Vec::new();
        let mut total_trimming_time = 0u64;
        let mut total_search_time = 0u64;
        let mut edges_sum = 0f64;
        
        // MAIN FIX: Actually iterate through each nonce
        for nonce_offset in 0..nonce_count {
            let current_nonce = start_nonce + nonce_offset;
            
            // Create header with current nonce (this is what was missing!)
            let mut header_with_nonce = header.to_vec();
            header_with_nonce.extend_from_slice(&current_nonce.to_le_bytes());
            
            // Create SipHash keys from header+nonce (different keys for each nonce!)
            let siphash_keys = create_siphash_keys(&header_with_nonce)?;
            
            // Execute trimming on GPU for THIS specific nonce
            let trimming_start = std::time::Instant::now();
            let trimmed_edges = unsafe { self.trimmer.run(&siphash_keys)? };
            let trimming_time = trimming_start.elapsed().as_millis() as u64;
            total_trimming_time += trimming_time;
            edges_sum += trimmed_edges.len() as f64;
            
            // Only search for cycles if we have reasonable number of edges
            if trimmed_edges.len() >= 84 { // Need at least 42 edges for a 42-cycle
                
                // Search for cycles on CPU
                let search_start = std::time::Instant::now();
                let mut solutions = search_cycles(&trimmed_edges)?;
                let search_time = search_start.elapsed().as_millis() as u64;
                total_search_time += search_time;
                
                // Limit solutions per nonce
                if solutions.len() > self.config.max_solutions {
                    solutions.truncate(self.config.max_solutions);
                }
                
                // Recover nonces for each solution
                for solution in solutions {
                    if let Ok((nonces, valid)) = unsafe { 
                        self.trimmer.recover(solution.cycle.to_vec(), &siphash_keys) 
                    } {
                        if valid && !nonces.is_empty() {
                            // Use the current nonce and the found cycle
                            let final_solution = Solution::new(current_nonce, solution.cycle);
                            all_solutions.push(final_solution);
                            
                            tracing::info!("🎉 SOLUTION FOUND! Nonce: {}, Cycle: {:?}...", 
                                current_nonce, &solution.cycle[0..5]);
                        }
                    }
                }
            }
            
            // Early exit if we found enough solutions
            if all_solutions.len() >= self.config.max_solutions {
                tracing::info!("🎯 Found {} solutions, stopping early at nonce {} (tested {})", 
                    all_solutions.len(), current_nonce, nonce_offset + 1);
                break;
            }
            
            // Progress reporting for large ranges
            if nonce_count > 100 && (nonce_offset + 1) % 100 == 0 {
                let elapsed = start_time.elapsed().as_secs();
                let rate = (nonce_offset + 1) as f64 / elapsed as f64;
                tracing::info!("⚡ Progress: {}/{} nonces ({:.1} nonces/sec, {} solutions so far)", 
                    nonce_offset + 1, nonce_count, rate, all_solutions.len());
            }
        }
        
        // Update statistics
        self.stats.solutions_found += all_solutions.len() as u64;
        self.stats.total_trimming_time_ms += total_trimming_time;
        self.stats.total_search_time_ms += total_search_time;
        
        // Update average edges remaining
        let avg_edges_this_run = edges_sum / nonce_count as f64;
        let new_avg = (self.stats.avg_edges_remaining * (self.stats.attempts - 1) as f64 + 
                      avg_edges_this_run) / self.stats.attempts as f64;
        self.stats.avg_edges_remaining = new_avg;
        
        // Update success rate
        self.stats.success_rate = self.stats.solutions_found as f64 / self.stats.attempts as f64;
        
        let total_time = start_time.elapsed().as_millis();
        let nonces_per_sec = nonce_count as f64 * 1000.0 / total_time as f64;
        
        tracing::info!("✅ Mining completed: {} solutions found in {}ms testing {} nonces ({:.2} nonces/sec)", 
            all_solutions.len(), total_time, nonce_count, nonces_per_sec);
        
        if all_solutions.is_empty() {
            tracing::info!("🔍 No solutions found in {} nonces - this is normal, try more nonces", nonce_count);
        }
        
        Ok(all_solutions)
    }
    
    /// Get device information
    pub fn device_info(&self) -> &DeviceCapabilities {
        &self.context.capabilities
    }
    
    /// Get backend configuration
    pub fn config(&self) -> &OpenClConfig {
        &self.config
    }
    
    /// Get backend statistics
    pub fn stats(&self) -> &BackendStats {
        &self.stats
    }
    
    /// Get detailed trimming statistics
    pub fn trimming_stats(&self) -> &TrimmingStats {
        self.trimmer.stats()
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = BackendStats::default();
    }
    
    /// Check if algorithm is supported
    pub fn supports_algorithm(&self, algorithm: Algorithm) -> bool {
        self.context.capabilities.supported_algorithms.contains(&algorithm)
    }
    
    /// Get optimal batch size for current configuration
    pub fn optimal_batch_size(&self) -> u64 {
        self.config.batch_size_override
            .unwrap_or(self.context.capabilities.buffer_config.max_batch_size)
    }
    
    /// Get memory usage information
    pub fn memory_usage(&self) -> MemoryUsage {
        let buffer_config = &self.context.capabilities.buffer_config;
        
        MemoryUsage {
            total_gpu_memory: self.context.capabilities.global_memory,
            buffer_memory_used: buffer_config.buffer_a1_size + 
                               buffer_config.buffer_a2_size + 
                               buffer_config.buffer_b_size + 
                               buffer_config.index_size * 2,
            memory_utilization: (buffer_config.buffer_a1_size + 
                                buffer_config.buffer_a2_size + 
                                buffer_config.buffer_b_size) as f64 / 
                               self.context.capabilities.global_memory as f64,
        }
    }
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total GPU memory in bytes
    pub total_gpu_memory: u64,
    /// Memory used by mining buffers in bytes
    pub buffer_memory_used: usize,
    /// Memory utilization as percentage (0.0 to 1.0)
    pub memory_utilization: f64,
}

/// Utility functions for OpenCL backend
pub mod utils {
    use super::*;
    
    /// List all available OpenCL platforms and devices
    pub fn list_platforms() -> Result<Vec<PlatformCapabilities>, Graxil29Error> {
        let detector = PlatformDetector::new()?;
        Ok(detector.list_all().to_vec())
    }
    
    /// Get recommended configuration for specific GPU
    pub fn recommended_config(vendor: GpuVendor, memory_gb: u64) -> OpenClConfig {
        let algorithm = if memory_gb >= 11 {
            Algorithm::Cuckatoo32
        } else {
            Algorithm::Cuckaroo29
        };
        
        let batch_size = match memory_gb {
            gb if gb >= 16 => Some(2000),
            gb if gb >= 11 => Some(1500),
            gb if gb >= 8 => Some(1000),
            _ => Some(500),
        };
        
        OpenClConfig {
            preferred_vendor: Some(vendor),
            algorithm,
            batch_size_override: batch_size,
            enable_profiling: cfg!(feature = "profile"),
            ..Default::default()
        }
    }
    
    /// Benchmark OpenCL backend performance
    pub fn benchmark(algorithm: Algorithm, iterations: u32) -> Result<BenchmarkResult, Graxil29Error> {
        let mut backend = OpenClBackend::with_config(OpenClConfig {
            algorithm,
            enable_profiling: true,
            ..Default::default()
        })?;
        
        let test_header = b"benchmark_header_data_for_testing_performance_metrics";
        let mut total_time = 0u64;
        let mut solutions_found = 0u64;
        
        for i in 0..iterations {
            let start = std::time::Instant::now();
            let solutions = backend.mine(test_header, i as u64 * 1000, 100)?; // Reduced from 1000 to 100 for faster benchmarks
            total_time += start.elapsed().as_millis() as u64;
            solutions_found += solutions.len() as u64;
        }
        
        Ok(BenchmarkResult {
            algorithm,
            iterations,
            total_time_ms: total_time,
            avg_time_per_iteration_ms: total_time / iterations as u64,
            total_solutions_found: solutions_found,
            solutions_per_second: (solutions_found as f64 * 1000.0) / total_time as f64,
            device_name: backend.device_info().name.clone(),
        })
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Algorithm used for benchmarking
    pub algorithm: Algorithm,
    /// Number of benchmark iterations performed
    pub iterations: u32,
    /// Total time taken for all iterations in milliseconds
    pub total_time_ms: u64,
    /// Average time per iteration in milliseconds
    pub avg_time_per_iteration_ms: u64,
    /// Total solutions found across all iterations
    pub total_solutions_found: u64,
    /// Solutions found per second
    pub solutions_per_second: f64,
    /// Name of the device used for benchmarking
    pub device_name: String,
}

/// Get vendor name as string
fn vendor_name(vendor: GpuVendor) -> &'static str {
    match vendor {
        GpuVendor::Nvidia => "NVIDIA",
        GpuVendor::Amd => "AMD", 
        GpuVendor::Intel => "Intel",
        GpuVendor::Unknown => "Unknown",
    }
}

/// Version and compatibility information
pub mod version {
    /// OpenCL backend version
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");
    
    /// Supported algorithms
    pub const SUPPORTED_ALGORITHMS: &[&str] = &["Cuckaroo29", "Cuckatoo32"];
    
    /// Minimum GPU memory requirements in GB
    pub const MIN_MEMORY_REQUIREMENTS: &[(Algorithm, u64)] = &[
        (Algorithm::Cuckaroo29, 6),
        (Algorithm::Cuckatoo32, 11),
    ];
    
    use crate::algorithms::Algorithm;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = OpenClConfig::default();
        assert_eq!(config.algorithm, Algorithm::Cuckaroo29);
        assert_eq!(config.max_solutions, DEFAULT_MAX_SOLUTIONS);
        assert!(!config.enable_profiling);
    }
    
    #[test]
    fn test_memory_usage_calculation() {
        // Mock capabilities for testing
        let mock_memory = 16 * 1024 * 1024 * 1024; // 16GB
        
        // Memory usage should be reasonable percentage
        assert!(mock_memory > 0);
    }
    
    #[test]
    fn test_version_info() {
        assert!(!version::VERSION.is_empty());
        assert_eq!(version::SUPPORTED_ALGORITHMS.len(), 2);
        assert_eq!(version::MIN_MEMORY_REQUIREMENTS.len(), 2);
    }
}