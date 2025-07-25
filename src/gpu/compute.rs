// src/gpu/compute.rs - OpenCL compute pipeline for multi-algorithm Graxil29 mining
// Tree location: ./src/gpu/compute.rs

//! OpenCL-based GPU compute pipeline for multi-algorithm Graxil29 mining
//! 
//! Provides real Cuckoo Cycle implementation using proven OpenCL algorithms.
//! Supports both Cuckaroo29 and Cuckatoo32 with automatic GPU detection,
//! memory management, and algorithm-specific optimizations.
//! 
//! # Version History
//! - 0.1.0: Initial GPU compute implementation (WGSL mock solutions)
//! - 0.2.0: Added accurate VRAM detection using nvml-wrapper for NVIDIA  
//! - 0.3.0: Added proper GPU solution readback, removed mock solutions
//! - 0.4.0: Fixed Cuckatoo32 submission format and edge index calculation
//! - 0.5.0: **MAJOR**: Complete rewrite using OpenCL backend for real mining
//! - 0.5.1: Enhanced multi-platform support and performance monitoring
//! - 0.5.2: Fixed compilation errors and benchmark implementation
//! - 0.5.3: **CRITICAL FIX**: Pass difficulty parameter to OpenCL backend
//! - 0.5.4: **MAJOR FIX**: Fixed difficulty calculation and added share difficulty display
//! - 0.5.5: **MAJOR FIX**: Replaced placeholder hash with BLAKE2b-512 for pool-compatible difficulty
//! - 0.5.6: Added debug logging for raw and scaled difficulties to diagnose pool difficulty mismatch
//! - 0.5.7: Adjusted scaling factor and added test case with real pool job header
//! - 0.5.8: Updated difficulty calculation to hash sorted cycle edges only, per Grin’s method
//! - 0.5.9: Removed scaling factor, using raw difficulty directly, and tested sorted/unsorted edge encoding
//! - 0.5.10: Switched to unsorted cycle edges for Aeternity, removed scaling, per GMiner’s raw difficulty
//! - 0.5.11: Made nonce count configurable via CLI for pool mining

use nvml_wrapper::Nvml;
use crate::{Graxil29Error, algorithms::{Algorithm, Solution}};
use crate::gpu::opencl::{
    OpenClBackend, OpenClConfig, GpuVendor, DeviceCapabilities, MemoryUsage
};
use blake2::{Blake2b512, Digest};
use hex::decode;

/// GPU memory information for algorithm selection
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    /// Total GPU memory in bytes
    pub total_memory: u64,
    /// Available memory for mining in bytes  
    pub available_memory: u64,
    /// Memory limit per algorithm in bytes
    pub memory_limit: u64,
}

/// Job template information from pool
#[derive(Debug, Clone)]
pub struct JobTemplate {
    /// Block height
    pub height: u64,
    /// Job identifier
    pub job_id: String, // Changed to String to match Stratum job_id
    /// Pre-proof-of-work header
    pub pre_pow: String,
    /// Difficulty target
    pub difficulty: f64, // Changed to f64 for pool difficulty
}

/// Pool type for solution formatting
#[derive(Debug, Clone, Copy)]
pub enum PoolType {
    /// Standard Stratum protocol with mining.subscribe
    Stratum,
    /// Custom JSON-RPC format
    JsonRpc,
}

impl Default for PoolType {
    fn default() -> Self {
        Self::Stratum
    }
}

/// Solution with calculated difficulty
#[derive(Debug, Clone)]
pub struct SolutionWithDifficulty {
    /// Original solution
    pub solution: Solution,
    /// Calculated difficulty of this solution
    pub difficulty: f64,
}

/// GPU compute pipeline supporting multiple algorithms with OpenCL backend
pub struct GpuCompute {
    /// OpenCL mining backend (real implementation)
    opencl_backend: OpenClBackend,
    /// GPU memory information
    memory_info: GpuMemoryInfo,
    /// Currently loaded algorithm
    current_algorithm: Option<Algorithm>,
    /// Current job template from pool
    current_job: Option<JobTemplate>,
    /// Mining statistics
    stats: ComputeStats,
}

/// Mining performance statistics
#[derive(Debug, Clone, Default)]
pub struct ComputeStats {
    /// Total mining attempts
    pub attempts: u64,
    /// Total solutions found
    pub solutions_found: u64,
    /// Total nonces processed
    pub nonces_processed: u64,
    /// Average mining time per attempt (ms)
    pub avg_mining_time_ms: f64,
    /// Success rate (solutions found / attempts)
    pub success_rate: f64,
    /// Current mining algorithm
    pub current_algorithm: Option<Algorithm>,
}

impl GpuCompute {
    /// Create a new GPU compute instance with auto-detection
    /// 
    /// Automatically detects the best GPU, validates capabilities,
    /// and initializes the OpenCL backend for real mining.
    pub async fn new() -> Result<Self, Graxil29Error> {
        tracing::info!("Initializing OpenCL-based GPU compute pipeline");
        
        // Initialize OpenCL backend with default configuration
        let opencl_backend = OpenClBackend::new()
            .map_err(|e| Graxil29Error::Gpu(format!("OpenCL initialization failed: {}", e)))?;
        
        // Get device capabilities for memory info
        let device_caps = opencl_backend.device_info();
        let memory_usage = opencl_backend.memory_usage();
        
        let memory_info = GpuMemoryInfo {
            total_memory: device_caps.global_memory,
            available_memory: device_caps.global_memory - memory_usage.buffer_memory_used as u64,
            memory_limit: (device_caps.global_memory * 8) / 10, // Use 80% of total memory
        };
        
        // Log GPU capabilities
        tracing::info!("Selected GPU: {} ({:.1}GB VRAM)", 
            device_caps.name, 
            device_caps.global_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        
        // Determine and log supported algorithms
        let supported_algorithms = Self::detect_supported_algorithms(&memory_info);
        tracing::info!("Supported algorithms: {:?}", supported_algorithms);
        
        if supported_algorithms.is_empty() {
            return Err(Graxil29Error::Gpu(
                "GPU does not meet minimum requirements for any algorithm".to_string()
            ));
        }
        
        Ok(Self {
            opencl_backend,
            memory_info,
            current_algorithm: None,
            current_job: None,
            stats: ComputeStats::default(),
        })
    }
    
    /// Create GPU compute with custom OpenCL configuration
    pub async fn with_config(config: OpenClConfig) -> Result<Self, Graxil29Error> {
        tracing::info!("Initializing OpenCL compute with custom config");
        
        let opencl_backend = OpenClBackend::with_config(config)
            .map_err(|e| Graxil29Error::Gpu(format!("OpenCL initialization failed: {}", e)))?;
        
        let device_caps = opencl_backend.device_info();
        let memory_usage = opencl_backend.memory_usage();
        
        let memory_info = GpuMemoryInfo {
            total_memory: device_caps.global_memory,
            available_memory: device_caps.global_memory - memory_usage.buffer_memory_used as u64,
            memory_limit: (device_caps.global_memory * 8) / 10,
        };
        
        tracing::info!("Custom GPU: {} ({:.1}GB VRAM)", 
            device_caps.name, 
            device_caps.global_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        
        Ok(Self {
            opencl_backend,
            memory_info,
            current_algorithm: None,
            current_job: None,
            stats: ComputeStats::default(),
        })
    }
    
    /// Set current job template from pool
    pub fn set_job_template(&mut self, job: JobTemplate) {
        tracing::debug!("Setting job template: height={}, job_id={}", job.height, job.job_id);
        self.current_job = Some(job);
    }
    
    /// Universal mining function supporting both algorithms with REAL implementation
    /// 
    /// Now returns solutions with calculated difficulty for proper pool submission
    /// 
    /// # Arguments
    /// * `algorithm` - Mining algorithm (Cuckaroo29 or Cuckatoo32)
    /// * `header_hash` - Block header hash (32 bytes)
    /// * `start_nonce` - Starting nonce value
    /// * `nonce_count` - Number of nonces to process (user-configurable)
    /// * `pool_difficulty` - Pool target difficulty (e.g., 16.0)
    /// 
    /// # Returns
    /// * `Vec<SolutionWithDifficulty>` - Solutions that meet pool difficulty
    pub async fn mine_algorithm(
        &mut self,
        algorithm: Algorithm,
        header_hash: [u8; 32],
        start_nonce: u64,
        nonce_count: u64,
        pool_difficulty: f64,
    ) -> Result<Vec<SolutionWithDifficulty>, Graxil29Error> {
        let start_time = std::time::Instant::now();
        self.stats.attempts += 1;
        self.stats.nonces_processed += nonce_count;
        
        // Validate algorithm support
        if !self.supports_algorithm(algorithm) {
            return Err(Graxil29Error::Gpu(format!(
                "Algorithm {:?} not supported by current GPU (need {}GB+ VRAM)",
                algorithm,
                match algorithm {
                    Algorithm::Cuckaroo29 => 6,
                    Algorithm::Cuckatoo32 => 11,
                }
            )));
        }
        
        // Update current algorithm
        self.current_algorithm = Some(algorithm);
        self.stats.current_algorithm = Some(algorithm);
        
        tracing::debug!("OpenCL mining: {} nonces starting from {} with {:?} (pool difficulty: {})", 
            nonce_count, start_nonce, algorithm, pool_difficulty);
        
        // Execute REAL mining using OpenCL backend (gets all valid cycles)
        let all_solutions = self.opencl_backend.mine(&header_hash, start_nonce, nonce_count)
            .map_err(|e| Graxil29Error::Gpu(format!("OpenCL mining failed: {}", e)))?;
        
        // Calculate difficulty for each solution and filter by pool requirement
        let solutions = self.filter_and_calculate_difficulty(all_solutions, &header_hash, pool_difficulty)?;
        
        // Update statistics
        let mining_time = start_time.elapsed().as_millis() as f64;
        self.stats.solutions_found += solutions.len() as u64;
        
        // Update average mining time
        let new_avg = (self.stats.avg_mining_time_ms * (self.stats.attempts - 1) as f64 + mining_time) 
                     / self.stats.attempts as f64;
        self.stats.avg_mining_time_ms = new_avg;
        
        // Update success rate
        self.stats.success_rate = self.stats.solutions_found as f64 / self.stats.attempts as f64;
        
        let gps = nonce_count as f64 / (mining_time / 1000.0);
        tracing::debug!("OpenCL mining completed: {} solutions meeting difficulty {:.2} in {:.1}ms ({:.1} Gps)", 
            solutions.len(), pool_difficulty, mining_time, gps);
        
        Ok(solutions)
    }
    
    /// Filter solutions by pool difficulty and calculate actual difficulty of each
    /// 
    /// This implements the same difficulty calculation as pools use to verify shares
    fn filter_and_calculate_difficulty(
        &self,
        solutions: Vec<Solution>,
        header_hash: &[u8; 32],
        pool_difficulty: f64,
    ) -> Result<Vec<SolutionWithDifficulty>, Graxil29Error> {
        let solutions_count = solutions.len();
        if solutions.is_empty() {
            return Ok(vec![]);
        }
        
        let mut valid_solutions = Vec::new();
        
        for solution in &solutions {
            // Calculate the actual difficulty of this solution using pool algorithm
            let calculated_difficulty = self.calculate_solution_difficulty(&solution, header_hash)?;
            
            // Log difficulty with high precision
            tracing::debug!(
                "Solution (nonce: {}): Difficulty: {:.2}",
                solution.nonce, calculated_difficulty
            );
            
            // Check if this solution meets the pool difficulty requirement
            if calculated_difficulty >= pool_difficulty {
                valid_solutions.push(SolutionWithDifficulty {
                    solution: solution.clone(),
                    difficulty: calculated_difficulty,
                });
            }
        }
        
        tracing::debug!("Difficulty filter: {}/{} solutions meet pool difficulty {:.2}", 
            valid_solutions.len(), solutions_count, pool_difficulty);
        
        Ok(valid_solutions)
    }
    
    /// Calculate the actual difficulty of a solution using pool algorithm
    /// 
    /// This implements the same calculation pools use to verify shares.
    /// Uses BLAKE2b-512 on unsorted cycle edges, with raw difficulty per Aeternity/GMiner.
    fn calculate_solution_difficulty(
        &self,
        solution: &Solution,
        _header_hash: &[u8; 32], // Header used in edge generation, not difficulty hash
    ) -> Result<f64, Graxil29Error> {
        // Use unsorted cycle edges, per Aeternity/GMiner
        let mut hash_input = Vec::new();
        for &edge in &solution.cycle {
            hash_input.extend_from_slice(&edge.to_le_bytes());
        }
        
        // Log hash input in hex
        tracing::debug!("Hash input for nonce {} (hex, unsorted cycle): {:?}", solution.nonce, hex::encode(&hash_input));
        
        // Compute BLAKE2b-512 hash
        let mut hasher = Blake2b512::new();
        hasher.update(&hash_input);
        let hash = hasher.finalize();
        
        // Log hash
        tracing::debug!("BLAKE2b-512 hash for nonce {}: {}", solution.nonce, hex::encode(&hash));
        
        // Use first 8 bytes for difficulty
        let hash_value = u64::from_le_bytes(hash[0..8].try_into()
            .map_err(|e| Graxil29Error::Gpu(format!("Hash conversion failed: {}", e)))?);
        
        // Log hash value
        tracing::debug!("Hash value for nonce {}: {}", solution.nonce, hash_value);
        
        // Calculate raw difficulty (no scaling, per GMiner)
        const MAX_TARGET: f64 = u64::MAX as f64;
        let difficulty = MAX_TARGET / (hash_value as f64 + 1.0);
        
        // Log difficulty
        tracing::debug!("Difficulty for nonce {}: {:.2}", solution.nonce, difficulty);
        
        Ok(difficulty)
    }
    
    /// Format solution for pool submission with support for different pool types
    /// 
    /// Supports both standard Stratum and custom JSON-RPC pool formats.
    /// This is pool-agnostic - different pool types handled by stratum layer.
    pub fn format_solution_for_pool(&self, solution: &Solution, pool_type: PoolType) -> Result<serde_json::Value, Graxil29Error> {
        let job = self.current_job.as_ref()
            .ok_or_else(|| Graxil29Error::Gpu("No job template available".to_string()))?;
        
        let algorithm = self.current_algorithm
            .ok_or_else(|| Graxil29Error::Gpu("No algorithm selected".to_string()))?;
        
        let edge_bits = match algorithm {
            Algorithm::Cuckaroo29 => 29,
            Algorithm::Cuckatoo32 => 32,
        };
        
        // Create submission based on pool type
        let submission = match pool_type {
            PoolType::Stratum => {
                // Standard Stratum format (compatible with most pools)
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": "submit",
                    "id": 3,
                    "params": {
                        "edge_bits": edge_bits,
                        "height": job.height,
                        "job_id": job.job_id,
                        "nonce": solution.nonce,
                        "pow": solution.cycle.to_vec()
                    }
                })
            }
            PoolType::JsonRpc => {
                // Custom JSON-RPC format (for alternative pools)
                serde_json::json!({
                    "method": "solution",
                    "params": {
                        "algorithm": format!("Cuckaroo{}", edge_bits),
                        "height": job.height,
                        "job": job.job_id,
                        "nonce": solution.nonce,
                        "proof": solution.cycle.to_vec(),
                        "difficulty": job.difficulty
                    },
                    "id": null
                })
            }
        };
        
        tracing::debug!("Formatted solution for {:?} pool: nonce={}", pool_type, solution.nonce);
        Ok(submission)
    }
    
    /// Format solution with default pool type (backward compatibility)
    pub fn format_solution(&self, solution: &Solution) -> Result<serde_json::Value, Graxil29Error> {
        self.format_solution_for_pool(solution, PoolType::default())
    }
    
    /// Get GPU information
    pub fn gpu_info(&self) -> &DeviceCapabilities {
        self.opencl_backend.device_info()
    }
    
    /// Get memory information
    pub fn memory_info(&self) -> &GpuMemoryInfo {
        &self.memory_info
    }
    
    /// Get memory usage statistics
    pub fn memory_usage(&self) -> MemoryUsage {
        self.opencl_backend.memory_usage()
    }
    
    /// Get supported algorithms based on GPU memory
    pub fn supported_algorithms(&self) -> Vec<Algorithm> {
        Self::detect_supported_algorithms(&self.memory_info)
    }
    
    /// Check if specific algorithm is supported
    pub fn supports_algorithm(&self, algorithm: Algorithm) -> bool {
        self.opencl_backend.supports_algorithm(algorithm)
    }
    
    /// Get mining performance statistics
    pub fn stats(&self) -> &ComputeStats {
        &self.stats
    }
    
    /// Reset mining statistics
    pub fn reset_stats(&mut self) {
        self.stats = ComputeStats::default();
        self.opencl_backend.reset_stats();
        tracing::info!("Mining statistics reset");
    }
    
    /// Get optimal batch size for current configuration
    pub fn optimal_batch_size(&self) -> u64 {
        self.opencl_backend.optimal_batch_size()
    }
    
    /// Detect supported algorithms based on memory requirements
    fn detect_supported_algorithms(memory_info: &GpuMemoryInfo) -> Vec<Algorithm> {
        let mut supported = Vec::new();
        let memory_gb = memory_info.available_memory / (1024 * 1024 * 1024);
        
        // Cuckaroo29: requires 6GB minimum
        if memory_gb >= 6 {
            supported.push(Algorithm::Cuckaroo29);
            tracing::info!("Cuckaroo29 supported ({} GB >= 6 GB required)", memory_gb);
        } else {
            tracing::warn!("Cuckaroo29 not supported ({} GB < 6 GB required)", memory_gb);
        }
        
        // Cuckatoo32: requires 11GB minimum
        if memory_gb >= 11 {
            supported.push(Algorithm::Cuckatoo32);
            tracing::info!("Cuckatoo32 supported ({} GB >= 11 GB required)", memory_gb);
        } else {
            tracing::warn!("Cuckatoo32 not supported ({} GB < 11 GB required)", memory_gb);
        }
        
        supported
    }
    
    /// Get NVIDIA VRAM size using nvml-wrapper (for enhanced memory detection)
    /// 
    /// This provides more accurate memory information than OpenCL queries
    /// on NVIDIA GPUs like the 4060Ti 16GB.
    pub fn get_nvidia_vram() -> Result<(u64, u64), Graxil29Error> {
        let nvml = Nvml::init()
            .map_err(|e| Graxil29Error::Gpu(format!("NVML init failed: {}", e)))?;
        
        let device_count = nvml.device_count()
            .map_err(|e| Graxil29Error::Gpu(format!("NVML device count failed: {}", e)))?;
        
        if device_count == 0 {
            return Err(Graxil29Error::Gpu("No NVIDIA GPU found".to_string()));
        }
        
        // Get first device (can be extended for multi-GPU setups)
        let device = nvml.device_by_index(0)
            .map_err(|e| Graxil29Error::Gpu(format!("NVML device access failed: {}", e)))?;
        
        let memory = device.memory_info()
            .map_err(|e| Graxil29Error::Gpu(format!("NVML memory info failed: {}", e)))?;
        
        tracing::debug!("NVIDIA VRAM: {:.1}GB total, {:.1}GB free", 
            memory.total as f64 / (1024.0 * 1024.0 * 1024.0),
            memory.free as f64 / (1024.0 * 1024.0 * 1024.0));
        
        Ok((memory.total, memory.free))
    }
    
    /// Validate mining parameters before execution
    pub fn validate_mining_params(
        &self,
        algorithm: Algorithm,
        nonce_count: u64,
    ) -> Result<(), Graxil29Error> {
        // Check algorithm support
        if !self.supports_algorithm(algorithm) {
            return Err(Graxil29Error::Gpu(format!(
                "Algorithm {:?} not supported by current GPU", algorithm
            )));
        }
        
        // Check batch size limits
        let max_batch = self.optimal_batch_size();
        if nonce_count > max_batch * 10 { // Allow up to 10 batches
            tracing::warn!("Large nonce count ({}) may require multiple batches", nonce_count);
        }
        
        // Check memory availability
        let memory_usage = self.memory_usage();
        if memory_usage.memory_utilization > 0.9 {
            tracing::warn!("High memory utilization ({:.1}%), performance may be affected", 
                memory_usage.memory_utilization * 100.0);
        }
        
        Ok(())
    }
    
    /// Get detailed performance report
    pub fn performance_report(&self) -> PerformanceReport {
        let opencl_stats = self.opencl_backend.stats();
        let _trimming_stats = self.opencl_backend.trimming_stats();
        let device_info = self.gpu_info();
        
        PerformanceReport {
            device_name: device_info.name.clone(),
            total_attempts: self.stats.attempts,
            total_solutions: self.stats.solutions_found,
            total_nonces: self.stats.nonces_processed,
            success_rate: self.stats.success_rate,
            avg_mining_time_ms: self.stats.avg_mining_time_ms,
            avg_trimming_time_ms: opencl_stats.total_trimming_time_ms as f64 / opencl_stats.attempts.max(1) as f64,
            avg_edges_remaining: opencl_stats.avg_edges_remaining,
            memory_utilization: self.memory_usage().memory_utilization,
            current_algorithm: self.stats.current_algorithm,
        }
    }
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// GPU device name
    pub device_name: String,
    /// Total mining attempts
    pub total_attempts: u64,
    /// Total solutions found
    pub total_solutions: u64,
    /// Total nonces processed
    pub total_nonces: u64,
    /// Solution success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average mining time per attempt (ms)
    pub avg_mining_time_ms: f64,
    /// Average trimming time per attempt (ms)
    pub avg_trimming_time_ms: f64,
    /// Average edges remaining after trimming
    pub avg_edges_remaining: f64,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// Currently configured algorithm
    pub current_algorithm: Option<Algorithm>,
}

impl PerformanceReport {
    /// Print formatted performance report
    pub fn print(&self) {
        println!("\nMining Performance Report");
        println!("{:=<50}", "");
        println!("GPU Device: {}", self.device_name);
        println!("Algorithm: {:?}", self.current_algorithm.unwrap_or(Algorithm::Cuckaroo29));
        println!("Total Attempts: {}", self.total_attempts);
        println!("Solutions Found: {}", self.total_solutions);
        println!("Success Rate: {:.2}%", self.success_rate * 100.0);
        println!("Avg Mining Time: {:.1}ms", self.avg_mining_time_ms);
        println!("Avg Trimming Time: {:.1}ms", self.avg_trimming_time_ms);
        println!("Avg Edges Remaining: {:.0}", self.avg_edges_remaining);
        println!("Memory Utilization: {:.1}%", self.memory_utilization * 100.0);
        println!("Total Nonces: {}", self.total_nonces);
        
        if self.total_nonces > 0 {
            println!("Nonces per Solution: {:.0}", 
                self.total_nonces as f64 / self.total_solutions.max(1) as f64);
        }
    }
}

/// Utility functions for GPU compute
pub mod utils {
    use super::*;
    
    /// Create GPU compute with recommended settings for specific hardware
    pub async fn create_for_hardware(memory_gb: u64, vendor: GpuVendor) -> Result<GpuCompute, Graxil29Error> {
        let algorithm = if memory_gb >= 11 {
            Algorithm::Cuckatoo32
        } else {
            Algorithm::Cuckaroo29
        };
        
        let config = OpenClConfig {
            preferred_vendor: Some(vendor),
            algorithm,
            enable_profiling: cfg!(feature = "profile"),
            ..Default::default()
        };
        
        GpuCompute::with_config(config).await
    }
    
    /// Benchmark GPU performance for algorithm selection
    pub async fn benchmark_algorithms(nonce_count: u64) -> Result<(), Graxil29Error> {
        // Get supported algorithms first
        let supported = {
            let compute = GpuCompute::new().await?;
            compute.supported_algorithms()
        };
        
        println!("Benchmarking supported algorithms...");
        
        for algorithm in supported {
            // Create fresh compute instance for each test to avoid move issues
            let mut compute = GpuCompute::new().await?;
            
            // Use real pool job header (job ID 1e7671 from ae.2miners.com)
            let test_header_hex = "a8db1910d85662f0167138c160c866683410c11f1ccfecb8ed8145716feb73e1";
            let test_header = decode(test_header_hex)
                .map_err(|e| Graxil29Error::Gpu(format!("Failed to decode header: {}", e)))?
                .try_into()
                .map_err(|_| Graxil29Error::Gpu("Invalid header length".to_string()))?;
            
            let start = std::time::Instant::now();
            
            // Use user-specified nonce count
            let solutions = compute.mine_algorithm(algorithm, test_header, 0, nonce_count, 16.0).await?;
            
            let time = start.elapsed().as_millis();
            println!("  {:?}: {}ms for {} nonces, {} solutions", 
                algorithm, time, nonce_count, solutions.len());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_opencl_compute_initialization() {
        // This test requires an OpenCL-capable GPU
        if let Ok(compute) = GpuCompute::new().await {
            assert!(!compute.supported_algorithms().is_empty());
            assert!(compute.memory_info().total_memory > 0);
        }
    }
    
    #[test]
    fn test_algorithm_detection() {
        let memory_info_6gb = GpuMemoryInfo {
            total_memory: 6 * 1024 * 1024 * 1024,
            available_memory: 6 * 1024 * 1024 * 1024,
            memory_limit: 5 * 1024 * 1024 * 1024,
        };
        
        let algorithms = GpuCompute::detect_supported_algorithms(&memory_info_6gb);
        assert_eq!(algorithms, vec![Algorithm::Cuckaroo29]);
        
        let memory_info_16gb = GpuMemoryInfo {
            total_memory: 16 * 1024 * 1024 * 1024,
            available_memory: 16 * 1024 * 1024 * 1024,
            memory_limit: 14 * 1024 * 1024 * 1024,
        };
        
        let algorithms = GpuCompute::detect_supported_algorithms(&memory_info_16gb);
        assert_eq!(algorithms, vec![Algorithm::Cuckaroo29, Algorithm::Cuckatoo32]);
    }
    
    #[test]
    fn test_pool_type_solution_formatting() {
        // Test would require a full compute instance with job template
        // This is a placeholder for integration testing
        assert_eq!(PoolType::default(), PoolType::Stratum);
    }
    
    #[test]
    fn test_performance_report_display() {
        let report = PerformanceReport {
            device_name: "Test GPU".to_string(),
            total_attempts: 10,
            total_solutions: 2,
            total_nonces: 1000,
            success_rate: 0.2,
            avg_mining_time_ms: 150.0,
            avg_trimming_time_ms: 120.0,
            avg_edges_remaining: 500.0,
            memory_utilization: 0.75,
            current_algorithm: Some(Algorithm::Cuckaroo29),
        };
        
        // Test that print doesn't panic
        report.print();
    }
}