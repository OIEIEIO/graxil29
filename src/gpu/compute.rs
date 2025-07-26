// src/gpu/compute.rs - OpenCL compute pipeline for multi-algorithm Graxil29 mining

//! # Graxil29 GPU Compute Pipeline
//!
//! OpenCL-based GPU compute pipeline for Cuckaroo29 and Cuckatoo32 mining.
//! Provides real Cuckoo Cycle implementation, automatic device/memory management, job scheduling, and pool-difficulty filtering.
//!
//! ## Features
//! - Real OpenCL backend (via OpenClBackend)
//! - Algorithm auto-detect based on VRAM
//! - Pool job tracking
//! - Share difficulty calculation (BLAKE2b-512)
//! - Full stats reporting & memory info
//! - Pool-agnostic share formatting

use nvml_wrapper::Nvml;
use crate::{
    Graxil29Error,
    algorithms::{Algorithm, Solution},
    gpu::opencl::{OpenClBackend, OpenClConfig, GpuVendor, DeviceCapabilities, MemoryUsage}
};
use blake2::{Blake2b512, Digest};

/// Information about GPU VRAM and memory limits for algorithm selection.
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    /// Total GPU memory in bytes.
    pub total_memory: u64,
    /// Available memory for mining in bytes.
    pub available_memory: u64,
    /// Memory limit per algorithm in bytes.
    pub memory_limit: u64,
}

/// Pool job template, typically derived from mining.notify.
#[derive(Debug, Clone)]
pub struct JobTemplate {
    /// Block height (changes with new blocks).
    pub height: u64,
    /// Pool-side job identifier (e.g. hex string).
    pub job_id: String,
    /// Pre-proof-of-work header.
    pub pre_pow: String,
    /// Pool difficulty (e.g. 16.0).
    pub difficulty: f64,
}

/// Pool type, used for formatting share submissions.
#[derive(Debug, Clone, Copy)]
pub enum PoolType {
    /// Standard Stratum protocol with mining.subscribe
    Stratum,
    /// Custom JSON-RPC format
    JsonRpc,
}
impl Default for PoolType {
    fn default() -> Self { PoolType::Stratum }
}

/// Wraps a mined solution and its calculated difficulty.
#[derive(Debug, Clone)]
pub struct SolutionWithDifficulty {
    /// Original solution.
    pub solution: Solution,
    /// Calculated pool difficulty.
    pub difficulty: f64,
}

/// Compute pipeline for managing GPU jobs, stats, and pool interface.
pub struct GpuCompute {
    opencl_backend: OpenClBackend,
    memory_info: GpuMemoryInfo,
    current_algorithm: Option<Algorithm>,
    current_job: Option<JobTemplate>,
    stats: ComputeStats,
}

/// Performance and share-finding statistics for mining jobs.
#[derive(Debug, Clone, Default)]
pub struct ComputeStats {
    /// Total mining attempts.
    pub attempts: u64,
    /// Total solutions found.
    pub solutions_found: u64,
    /// Total nonces processed.
    pub nonces_processed: u64,
    /// Average mining time per attempt (ms).
    pub avg_mining_time_ms: f64,
    /// Success rate (solutions found / attempts).
    pub success_rate: f64,
    /// Current mining algorithm.
    pub current_algorithm: Option<Algorithm>,
}

impl GpuCompute {
    /// Initialize a new GPU mining pipeline with best available hardware.
    ///
    /// This probes the best GPU, selects the best algorithm, and initializes the OpenCL backend.
    pub async fn new() -> Result<Self, Graxil29Error> {
        tracing::info!("Initializing OpenCL-based GPU compute pipeline");

        let opencl_backend = OpenClBackend::new()
            .map_err(|e| Graxil29Error::Gpu(format!("OpenCL initialization failed: {}", e)))?;

        let device_caps = opencl_backend.device_info();
        let memory_usage = opencl_backend.memory_usage();

        let memory_info = GpuMemoryInfo {
            total_memory: device_caps.global_memory,
            available_memory: device_caps.global_memory - memory_usage.buffer_memory_used as u64,
            memory_limit: (device_caps.global_memory * 8) / 10,
        };

        tracing::info!(
            "Selected GPU: {} ({:.1}GB VRAM)",
            device_caps.name,
            device_caps.global_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        let supported_algorithms = Self::detect_supported_algorithms(&memory_info);
        tracing::info!("Supported algorithms: {:?}", supported_algorithms);

        if supported_algorithms.is_empty() {
            return Err(Graxil29Error::Gpu(
                "GPU does not meet minimum requirements for any algorithm".to_string(),
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

    /// Create GPU compute with explicit OpenCL configuration.
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

        tracing::info!(
            "Custom GPU: {} ({:.1}GB VRAM)",
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

    /// Set the current pool job. Only the latest job is tracked and mined.
    pub fn set_job_template(&mut self, job: JobTemplate) {
        tracing::debug!("Setting job template: height={}, job_id={}", job.height, job.job_id);
        self.current_job = Some(job);
    }

    /// Core mining function: mines the current (latest) pool job only.
    ///
    /// # Arguments
    /// * `algorithm` - Algorithm to mine.
    /// * `header_hash` - 32-byte header for edge/key generation.
    /// * `start_nonce` - First nonce to try.
    /// * `nonce_count` - How many nonces to process in this batch.
    /// * `pool_difficulty` - Pool share difficulty (e.g., 16.0).
    ///
    /// # Returns
    /// Vector of solutions (nonce, cycle) meeting pool difficulty.
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

        self.current_algorithm = Some(algorithm);
        self.stats.current_algorithm = Some(algorithm);

        tracing::debug!(
            "OpenCL mining: {} nonces from {} with {:?} (pool diff: {})",
            nonce_count,
            start_nonce,
            algorithm,
            pool_difficulty
        );

        let all_solutions = self
            .opencl_backend
            .mine(&header_hash, start_nonce, nonce_count)
            .map_err(|e| Graxil29Error::Gpu(format!("OpenCL mining failed: {}", e)))?;

        let solutions =
            self.filter_and_calculate_difficulty(all_solutions, &header_hash, pool_difficulty)?;

        let mining_time = start_time.elapsed().as_millis() as f64;
        self.stats.solutions_found += solutions.len() as u64;

        // Rolling average mining time.
        self.stats.avg_mining_time_ms =
            (self.stats.avg_mining_time_ms * (self.stats.attempts - 1) as f64 + mining_time)
                / self.stats.attempts as f64;

        self.stats.success_rate =
            self.stats.solutions_found as f64 / self.stats.attempts as f64;

        let gps = nonce_count as f64 / (mining_time / 1000.0);
        tracing::debug!(
            "OpenCL mining completed: {} solutions >= diff {:.2} in {:.1}ms ({:.2} Gps)",
            solutions.len(),
            pool_difficulty,
            mining_time,
            gps
        );

        Ok(solutions)
    }

    /// Filters solutions by pool difficulty, returning only valid shares with calculated difficulties.
    fn filter_and_calculate_difficulty(
        &self,
        solutions: Vec<Solution>,
        header_hash: &[u8; 32],
        pool_difficulty: f64,
    ) -> Result<Vec<SolutionWithDifficulty>, Graxil29Error> {
        let mut valid = Vec::new();

        for solution in &solutions {
            let calculated = self.calculate_solution_difficulty(solution, header_hash)?;
            tracing::debug!(
                "Solution (nonce: {}): Difficulty: {:.2}",
                solution.nonce,
                calculated
            );
            if calculated >= pool_difficulty {
                valid.push(SolutionWithDifficulty {
                    solution: solution.clone(),
                    difficulty: calculated,
                });
            }
        }

        tracing::debug!(
            "Difficulty filter: {}/{} >= diff {:.2}",
            valid.len(),
            solutions.len(),
            pool_difficulty
        );

        Ok(valid)
    }

    /// Calculates share difficulty, per pool convention, on the cycle (BLAKE2b-512, little-endian).
    fn calculate_solution_difficulty(
        &self,
        solution: &Solution,
        _header_hash: &[u8; 32],
    ) -> Result<f64, Graxil29Error> {
        let mut hash_input = Vec::new();
        for &edge in &solution.cycle {
            hash_input.extend_from_slice(&edge.to_le_bytes());
        }

        let mut hasher = Blake2b512::new();
        hasher.update(&hash_input);
        let hash = hasher.finalize();

        let hash_value = u64::from_le_bytes(hash[0..8].try_into().map_err(|e| {
            Graxil29Error::Gpu(format!("Hash conversion failed: {}", e))
        })?);

        const MAX_TARGET: f64 = u64::MAX as f64;
        let difficulty = MAX_TARGET / (hash_value as f64 + 1.0);

        Ok(difficulty)
    }

    /// Formats solution for pool submission. Handles both Stratum and JSON-RPC pools.
    pub fn format_solution_for_pool(
        &self,
        solution: &Solution,
        pool_type: PoolType,
    ) -> Result<serde_json::Value, Graxil29Error> {
        let job = self
            .current_job
            .as_ref()
            .ok_or_else(|| Graxil29Error::Gpu("No job template available".to_string()))?;

        let algorithm = self
            .current_algorithm
            .ok_or_else(|| Graxil29Error::Gpu("No algorithm selected".to_string()))?;

        let edge_bits = match algorithm {
            Algorithm::Cuckaroo29 => 29,
            Algorithm::Cuckatoo32 => 32,
        };

        let submission = match pool_type {
            PoolType::Stratum => serde_json::json!({
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
            }),
            PoolType::JsonRpc => serde_json::json!({
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
            }),
        };

        tracing::debug!(
            "Formatted solution for {:?} pool: nonce={}",
            pool_type,
            solution.nonce
        );
        Ok(submission)
    }

    /// Formats solution for submission using the default pool type (Stratum).
    pub fn format_solution(
        &self,
        solution: &Solution,
    ) -> Result<serde_json::Value, Graxil29Error> {
        self.format_solution_for_pool(solution, PoolType::default())
    }

    /// Returns info about the OpenCL device in use.
    pub fn gpu_info(&self) -> &DeviceCapabilities {
        self.opencl_backend.device_info()
    }

    /// Returns memory info struct for the GPU.
    pub fn memory_info(&self) -> &GpuMemoryInfo {
        &self.memory_info
    }

    /// Returns current buffer usage statistics.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.opencl_backend.memory_usage()
    }

    /// Returns a list of all algorithms supported given available VRAM.
    pub fn supported_algorithms(&self) -> Vec<Algorithm> {
        Self::detect_supported_algorithms(&self.memory_info)
    }

    /// Returns true if a specific algorithm is supported by this device.
    pub fn supports_algorithm(&self, algorithm: Algorithm) -> bool {
        self.opencl_backend.supports_algorithm(algorithm)
    }

    /// Returns all currently tracked performance statistics.
    pub fn stats(&self) -> &ComputeStats {
        &self.stats
    }

    /// Resets mining stats and internal backend statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ComputeStats::default();
        self.opencl_backend.reset_stats();
        tracing::info!("Mining statistics reset");
    }

    /// Returns the optimal batch size for current GPU/algorithm configuration.
    pub fn optimal_batch_size(&self) -> u64 {
        self.opencl_backend.optimal_batch_size()
    }

    /// Determines supported mining algorithms for the device's available memory.
    fn detect_supported_algorithms(memory_info: &GpuMemoryInfo) -> Vec<Algorithm> {
        let mut supported = Vec::new();
        let memory_gb = memory_info.available_memory / (1024 * 1024 * 1024);

        if memory_gb >= 6 {
            supported.push(Algorithm::Cuckaroo29);
            tracing::info!("Cuckaroo29 supported ({} GB >= 6 GB)", memory_gb);
        } else {
            tracing::warn!("Cuckaroo29 not supported ({} GB < 6 GB)", memory_gb);
        }

        if memory_gb >= 11 {
            supported.push(Algorithm::Cuckatoo32);
            tracing::info!("Cuckatoo32 supported ({} GB >= 11 GB)", memory_gb);
        } else {
            tracing::warn!("Cuckatoo32 not supported ({} GB < 11 GB)", memory_gb);
        }

        supported
    }

    /// Uses NVML to obtain the most accurate VRAM statistics on NVIDIA GPUs.
    pub fn get_nvidia_vram() -> Result<(u64, u64), Graxil29Error> {
        let nvml = Nvml::init()
            .map_err(|e| Graxil29Error::Gpu(format!("NVML init failed: {}", e)))?;

        let device_count = nvml.device_count()
            .map_err(|e| Graxil29Error::Gpu(format!("NVML device count failed: {}", e)))?;

        if device_count == 0 {
            return Err(Graxil29Error::Gpu("No NVIDIA GPU found".to_string()));
        }

        let device = nvml.device_by_index(0)
            .map_err(|e| Graxil29Error::Gpu(format!("NVML device access failed: {}", e)))?;

        let memory = device.memory_info()
            .map_err(|e| Graxil29Error::Gpu(format!("NVML memory info failed: {}", e)))?;

        tracing::debug!(
            "NVIDIA VRAM: {:.1}GB total, {:.1}GB free",
            memory.total as f64 / (1024.0 * 1024.0 * 1024.0),
            memory.free as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        Ok((memory.total, memory.free))
    }

    /// Validates that mining parameters are supported on this device.
    pub fn validate_mining_params(
        &self,
        algorithm: Algorithm,
        nonce_count: u64,
    ) -> Result<(), Graxil29Error> {
        if !self.supports_algorithm(algorithm) {
            return Err(Graxil29Error::Gpu(format!(
                "Algorithm {:?} not supported by current GPU",
                algorithm
            )));
        }

        let max_batch = self.optimal_batch_size();
        if nonce_count > max_batch * 10 {
            tracing::warn!(
                "Large nonce count ({}) may require multiple batches",
                nonce_count
            );
        }

        let memory_usage = self.memory_usage();
        if memory_usage.memory_utilization > 0.9 {
            tracing::warn!(
                "High memory utilization ({:.1}%), performance may be affected",
                memory_usage.memory_utilization * 100.0
            );
        }

        Ok(())
    }

    /// Returns a full performance report struct.
    pub fn performance_report(&self) -> PerformanceReport {
        let opencl_stats = self.opencl_backend.stats();
        let device_info = self.gpu_info();

        PerformanceReport {
            device_name: device_info.name.clone(),
            total_attempts: self.stats.attempts,
            total_solutions: self.stats.solutions_found,
            total_nonces: self.stats.nonces_processed,
            success_rate: self.stats.success_rate,
            avg_mining_time_ms: self.stats.avg_mining_time_ms,
            avg_trimming_time_ms: opencl_stats.total_trimming_time_ms as f64
                / opencl_stats.attempts.max(1) as f64,
            avg_edges_remaining: opencl_stats.avg_edges_remaining,
            memory_utilization: self.memory_usage().memory_utilization,
            current_algorithm: self.stats.current_algorithm,
        }
    }
}

/// Printable mining performance and efficiency statistics.
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Name of the GPU device.
    pub device_name: String,
    /// Total mining attempts made.
    pub total_attempts: u64,
    /// Total solutions found.
    pub total_solutions: u64,
    /// Total nonces processed.
    pub total_nonces: u64,
    /// Success rate (solutions/attempts).
    pub success_rate: f64,
    /// Average mining time per attempt (ms).
    pub avg_mining_time_ms: f64,
    /// Average trimming time per attempt (ms).
    pub avg_trimming_time_ms: f64,
    /// Average edges remaining after trimming.
    pub avg_edges_remaining: f64,
    /// Memory utilization (0.0 to 1.0).
    pub memory_utilization: f64,
    /// Algorithm being mined, if set.
    pub current_algorithm: Option<Algorithm>,
}

impl PerformanceReport {
    /// Prints this performance report in a formatted, human-readable manner.
    pub fn print(&self) {
        println!("\nMining Performance Report");
        println!("{:=<50}", "");
        println!("GPU Device: {}", self.device_name);
        println!(
            "Algorithm: {:?}",
            self.current_algorithm.unwrap_or(Algorithm::Cuckaroo29)
        );
        println!("Total Attempts: {}", self.total_attempts);
        println!("Solutions Found: {}", self.total_solutions);
        println!("Success Rate: {:.2}%", self.success_rate * 100.0);
        println!("Avg Mining Time: {:.1}ms", self.avg_mining_time_ms);
        println!("Avg Trimming Time: {:.1}ms", self.avg_trimming_time_ms);
        println!("Avg Edges Remaining: {:.0}", self.avg_edges_remaining);
        println!("Memory Utilization: {:.1}%", self.memory_utilization * 100.0);
        println!("Total Nonces: {}", self.total_nonces);

        if self.total_nonces > 0 {
            println!(
                "Nonces per Solution: {:.0}",
                self.total_nonces as f64 / self.total_solutions.max(1) as f64
            );
        }
    }
}

/// Utility helpers for setup and benchmarking.
pub mod utils {
    use super::*;

    /// Creates a GPU compute instance with the recommended algorithm for the hardware.
    ///
    /// # Arguments
    /// * `memory_gb` - Amount of GPU VRAM in GB.
    /// * `vendor` - Preferred GPU vendor.
    pub async fn create_for_hardware(
        memory_gb: u64,
        vendor: GpuVendor,
    ) -> Result<GpuCompute, Graxil29Error> {
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

    /// Benchmarks all supported algorithms for this device.
    ///
    /// # Arguments
    /// * `nonce_count` - How many nonces to process in each benchmark.
    pub async fn benchmark_algorithms(nonce_count: u64) -> Result<(), Graxil29Error> {
        let supported = {
            let compute = GpuCompute::new().await?;
            compute.supported_algorithms()
        };

        println!("Benchmarking supported algorithms...");

        for algorithm in supported {
            let mut compute = GpuCompute::new().await?;

            // Example header hash (replace with real job header as needed)
            let test_header_hex = "a8db1910d85662f0167138c160c866683410c11f1ccfecb8ed8145716feb73e1";
            let test_header = hex::decode(test_header_hex)
                .map_err(|e| Graxil29Error::Gpu(format!("Failed to decode header: {}", e)))?
                .try_into()
                .map_err(|_| Graxil29Error::Gpu("Invalid header length".to_string()))?;

            let start = std::time::Instant::now();

            let solutions = compute
                .mine_algorithm(algorithm, test_header, 0, nonce_count, 16.0)
                .await?;

            let time = start.elapsed().as_millis();
            println!(
                "  {:?}: {}ms for {} nonces, {} solutions",
                algorithm,
                time,
                nonce_count,
                solutions.len()
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_opencl_compute_initialization() {
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

        report.print();
    }
}
