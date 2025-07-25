// src/gpu/opencl/kernels.rs - OpenCL Kernel Management and Execution
// Tree location: ./src/gpu/opencl/kernels.rs

//! OpenCL kernel compilation, execution, and performance monitoring
//! 
//! Provides high-level utilities for managing OpenCL kernels with automatic
//! compilation, caching, platform-specific optimizations, and performance profiling.
//! 
//! # Version History
//! - 0.1.0: Initial kernel management with compilation and execution
//! - 0.1.1: Added performance profiling and caching
//! - 0.1.2: Enhanced error handling and platform optimizations
//! - 0.1.3: Fixed compilation errors, format strings, and API compatibility
//! 
//! # Kernel Management
//! - **Compilation**: Automatic kernel compilation with error reporting
//! - **Caching**: Compiled kernel caching for faster startup
//! - **Execution**: High-level kernel execution with parameter binding
//! - **Profiling**: Detailed performance monitoring and timing
//! - **Optimization**: Platform-specific work group size tuning
//! 
//! # Supported Kernels
//! - FluffySeed2A/2B: Edge generation using SipHash
//! - FluffyRound1/N: Graph trimming rounds (120+ iterations)
//! - FluffyTail: Final edge collection
//! - FluffyRecovery: Nonce recovery from solution cycles

use ocl::{Context, Device, Program, Kernel, Queue, SpatialDims, Event, EventList};
use ocl::enums::ProfilingInfo;
use std::collections::HashMap;
use std::time::Instant;
use std::sync::Arc;
use crate::algorithms::Algorithm;
use super::platform::{DeviceCapabilities, GpuVendor};

/// Kernel compilation and execution errors
#[derive(Debug, Clone)]
pub enum KernelError {
    /// Kernel compilation failed
    CompilationFailed { 
        /// Name of the kernel that failed compilation
        kernel_name: String, 
        /// Compilation error message
        error: String 
    },
    /// Kernel execution failed
    ExecutionFailed { 
        /// Name of the kernel that failed execution
        kernel_name: String, 
        /// Execution error message
        error: String 
    },
    /// Kernel not found
    KernelNotFound(String),
    /// Invalid kernel parameters
    InvalidParameters { 
        /// Name of the kernel with invalid parameters
        kernel_name: String, 
        /// Parameter error message
        error: String 
    },
    /// Unsupported work group size
    UnsupportedWorkGroupSize { 
        /// Requested work group size
        requested: usize, 
        /// Maximum supported work group size
        max: usize 
    },
    /// Device capability insufficient
    InsufficientCapability(String),
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CompilationFailed { kernel_name, error } => {
                write!(f, "Kernel '{}' compilation failed: {}", kernel_name, error)
            }
            Self::ExecutionFailed { kernel_name, error } => {
                write!(f, "Kernel '{}' execution failed: {}", kernel_name, error)
            }
            Self::KernelNotFound(name) => write!(f, "Kernel '{}' not found", name),
            Self::InvalidParameters { kernel_name, error } => {
                write!(f, "Invalid parameters for kernel '{}': {}", kernel_name, error)
            }
            Self::UnsupportedWorkGroupSize { requested, max } => {
                write!(f, "Work group size {} exceeds maximum {}", requested, max)
            }
            Self::InsufficientCapability(msg) => write!(f, "Insufficient device capability: {}", msg),
        }
    }
}

impl std::error::Error for KernelError {}

/// Kernel compilation result
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Compilation success status
    pub success: bool,
    /// Compilation time in milliseconds
    pub compilation_time_ms: u64,
    /// Compilation log/errors
    pub log: String,
    /// Number of kernels compiled
    pub kernel_count: usize,
    /// Compiler optimizations applied
    pub optimizations: Vec<String>,
}

/// Kernel execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Kernel name
    pub kernel_name: String,
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
    /// Queue time (waiting) in nanoseconds
    pub queue_time_ns: u64,
    /// Global work size used
    pub global_work_size: Vec<usize>,
    /// Local work size used
    pub local_work_size: Vec<usize>,
    /// Number of executions
    pub execution_count: u64,
    /// Total compute units utilized
    pub compute_units_used: u32,
}

/// Work group size configuration
#[derive(Debug, Clone)]
pub struct WorkGroupConfig {
    /// Preferred local work size
    pub local_work_size: Option<SpatialDims>,
    /// Global work size calculation
    pub global_work_size: SpatialDims,
    /// Whether to use platform-specific optimizations
    pub platform_optimized: bool,
}

/// Kernel parameter binding helper
#[derive(Debug)]
pub enum KernelArg {
    /// 64-bit unsigned integer
    U64(u64),
    /// 32-bit unsigned integer
    U32(u32),
    /// 32-bit signed integer
    I32(i32),
    /// Buffer reference (generic)
    Buffer(String), // Buffer name for deferred binding
    /// Null buffer placeholder
    NullBuffer,
}

/// High-level kernel manager
pub struct KernelManager {
    /// Compiled OpenCL program
    program: Arc<Program>,
    /// Available kernels
    kernels: HashMap<String, Kernel>,
    /// Device capabilities
    device_capabilities: DeviceCapabilities,
    /// Execution statistics
    execution_stats: HashMap<String, ExecutionStats>,
    /// GPU vendor for optimizations
    vendor: GpuVendor,
    /// Profiling enabled
    profiling_enabled: bool,
}

/// Kernel compiler with advanced optimizations
pub struct KernelCompiler {
    /// OpenCL context
    context: Arc<Context>,
    /// Target device
    device: Arc<Device>,
    /// Device capabilities
    device_capabilities: DeviceCapabilities,
    /// GPU vendor
    vendor: GpuVendor,
    /// Compilation cache
    compilation_cache: HashMap<String, Arc<Program>>,
}

/// Kernel executor with performance monitoring
pub struct KernelExecutor {
    /// Command queue
    _queue: Arc<Queue>,
    /// Kernel manager reference
    _kernel_manager: Arc<KernelManager>,
    /// Active events for profiling
    _active_events: EventList,
    /// Execution history
    _execution_history: Vec<ExecutionStats>,
    /// Performance monitoring enabled
    _monitoring_enabled: bool,
}

impl KernelCompiler {
    /// Create new kernel compiler
    pub fn new(
        context: Arc<Context>,
        device: Arc<Device>, 
        device_capabilities: DeviceCapabilities,
        vendor: GpuVendor,
    ) -> Self {
        Self {
            context,
            device,
            device_capabilities,
            vendor,
            compilation_cache: HashMap::new(),
        }
    }
    
    /// Compile kernels for specific algorithm
    pub fn compile_algorithm_kernels(&mut self, _algorithm: Algorithm) -> Result<CompilationResult, KernelError> {
        let start_time = Instant::now();
        tracing::info!("🔨 Compiling {:?} kernels for {} {}", 
            _algorithm, vendor_name(self.vendor), self.device_capabilities.name);
        
        // Select appropriate kernel source
        let kernel_source = match _algorithm {
            Algorithm::Cuckaroo29 => include_str!("../kernels/opencl/cuckaroo29.cl"),
            Algorithm::Cuckatoo32 => include_str!("../kernels/opencl/cuckatoo32.cl"),
        };
        
        // Generate cache key
        let cache_key = format!("{:?}_{}", _algorithm, self.device_capabilities.name);
        
        // Check cache first
        if let Some(_cached_program) = self.compilation_cache.get(&cache_key) {
            tracing::info!("✅ Using cached kernels for {:?}", _algorithm);
            return Ok(CompilationResult {
                success: true,
                compilation_time_ms: 0,
                log: "Used cached compilation".to_string(),
                kernel_count: self.count_kernels_in_source(kernel_source),
                optimizations: vec!["Cache hit".to_string()],
            });
        }
        
        // Build compilation options
        let build_options = self.build_compilation_options(_algorithm);
        
        // Compile program - Fixed API usage for OCL 0.19
        let program_result = Program::builder()
            .devices(&*self.device)
            .src(kernel_source)
            .cmplr_opt(&build_options)  // Fixed: use cmplr_opt instead of build_options
            .build(&*self.context);
        
        let compilation_time = start_time.elapsed().as_millis() as u64;
        
        match program_result {
            Ok(program) => {
                // Cache successful compilation
                self.compilation_cache.insert(cache_key, Arc::new(program));
                
                tracing::info!("✅ Kernel compilation successful in {}ms", compilation_time);
                
                Ok(CompilationResult {
                    success: true,
                    compilation_time_ms: compilation_time,
                    log: "Compilation successful".to_string(),
                    kernel_count: self.count_kernels_in_source(kernel_source),
                    optimizations: self.get_applied_optimizations(),
                })
            }
            Err(e) => {
                let error_msg = format!("OpenCL compilation error: {}", e);
                tracing::error!("❌ Kernel compilation failed: {}", error_msg);
                
                Err(KernelError::CompilationFailed {
                    kernel_name: format!("{:?}_kernels", _algorithm),
                    error: error_msg,
                })
            }
        }
    }
    
    /// Build compilation options for platform
    fn build_compilation_options(&self, algorithm: Algorithm) -> String {
        let mut options = Vec::new();
        
        // Basic optimization flags
        options.push("-cl-fast-relaxed-math".to_string());
        options.push("-cl-mad-enable".to_string());
        
        // Platform-specific optimizations
        match self.vendor {
            GpuVendor::Nvidia => {
                options.push("-cl-nv-verbose".to_string());
                options.push("-cl-nv-maxrregcount=64".to_string());
            }
            GpuVendor::Amd => {
                options.push("-cl-amd-media-ops".to_string());
                if self.device_capabilities.compute_units >= 36 {
                    options.push("-cl-amd-media-ops2".to_string());
                }
            }
            GpuVendor::Intel => {
                options.push("-cl-intel-gtpin-rera".to_string());
            }
            GpuVendor::Unknown => {
                // Conservative options for unknown vendors
            }
        }
        
        // Algorithm-specific definitions
        match algorithm {
            Algorithm::Cuckaroo29 => {
                options.push("-DEDGEBITS=29".to_string());
                options.push("-DCUCKAROO=1".to_string());
            }
            Algorithm::Cuckatoo32 => {
                options.push("-DEDGEBITS=32".to_string());
                options.push("-DCUCKATOO=1".to_string());
            }
        }
        
        // Memory optimizations based on available VRAM
        let memory_gb = self.device_capabilities.global_memory / (1024 * 1024 * 1024);
        if memory_gb >= 16 {
            options.push("-DLARGE_MEMORY=1".to_string());
        }
        
        options.join(" ")
    }
    
    /// Count kernels in source code
    fn count_kernels_in_source(&self, source: &str) -> usize {
        source.lines()
            .filter(|line| line.trim_start().starts_with("__kernel"))
            .count()
    }
    
    /// Get list of applied optimizations
    fn get_applied_optimizations(&self) -> Vec<String> {
        let mut optimizations = vec![
            "Fast relaxed math".to_string(),
            "MAD instructions enabled".to_string(),
        ];
        
        match self.vendor {
            GpuVendor::Nvidia => {
                optimizations.push("NVIDIA-specific optimizations".to_string());
            }
            GpuVendor::Amd => {
                optimizations.push("AMD media operations".to_string());
            }
            GpuVendor::Intel => {
                optimizations.push("Intel-specific optimizations".to_string());
            }
            GpuVendor::Unknown => {}
        }
        
        optimizations
    }
    
    /// Get compiled program for algorithm
    pub fn get_program(&self, algorithm: Algorithm) -> Option<Arc<Program>> {
        let cache_key = format!("{:?}_{}", algorithm, self.device_capabilities.name);
        self.compilation_cache.get(&cache_key).cloned()
    }
}

impl KernelManager {
    /// Create kernel manager from compiled program
    pub fn new(
        program: Arc<Program>,
        device_capabilities: DeviceCapabilities,
        vendor: GpuVendor,
        profiling_enabled: bool,
    ) -> Result<Self, KernelError> {
        let mut manager = Self {
            program,
            kernels: HashMap::new(),
            device_capabilities,
            execution_stats: HashMap::new(),
            vendor,
            profiling_enabled,
        };
        
        // Pre-build common kernels
        manager.initialize_kernels()?;
        
        Ok(manager)
    }
    
    /// Initialize all required kernels
    fn initialize_kernels(&mut self) -> Result<(), KernelError> {
        let kernel_names = vec![
            "FluffySeed2A",
            "FluffySeed2B", 
            "FluffyRound1",
            "FluffyRoundN",
            "FluffyRoundNO1",
            "FluffyRoundNON",
            "FluffyTail",
            "FluffyTailO",
            "FluffyRecovery",
        ];
        
        for kernel_name in kernel_names {
            match Kernel::builder()
                .name(kernel_name)
                .program(&*self.program)
                .build()
            {
                Ok(kernel) => {
                    tracing::debug!("✅ Kernel '{}' initialized", kernel_name);
                    self.kernels.insert(kernel_name.to_string(), kernel);
                    self.execution_stats.insert(kernel_name.to_string(), ExecutionStats {
                        kernel_name: kernel_name.to_string(),
                        ..Default::default()
                    });
                }
                Err(e) => {
                    tracing::warn!("⚠️  Failed to initialize kernel '{}': {}", kernel_name, e);
                    // Some kernels might not exist in all versions - continue
                }
            }
        }
        
        tracing::info!("🔧 Initialized {} kernels", self.kernels.len());
        Ok(())
    }
    
    /// Get kernel by name
    pub fn get_kernel(&self, name: &str) -> Result<&Kernel, KernelError> {
        self.kernels.get(name)
            .ok_or_else(|| KernelError::KernelNotFound(name.to_string()))
    }
    
    /// Get mutable kernel by name
    pub fn get_kernel_mut(&mut self, name: &str) -> Result<&mut Kernel, KernelError> {
        self.kernels.get_mut(name)
            .ok_or_else(|| KernelError::KernelNotFound(name.to_string()))
    }
    
    /// Create optimized work group configuration
    pub fn create_work_group_config(&self, kernel_name: &str, global_work_size: usize) -> WorkGroupConfig {
        let local_work_size = self.calculate_optimal_local_work_size(kernel_name, global_work_size);
        
        WorkGroupConfig {
            local_work_size,
            global_work_size: SpatialDims::One(global_work_size),
            platform_optimized: true,
        }
    }
    
    /// Calculate optimal local work size for kernel
    fn calculate_optimal_local_work_size(&self, kernel_name: &str, global_work_size: usize) -> Option<SpatialDims> {
        // Platform-specific work group sizes
        let preferred_size = match self.vendor {
            GpuVendor::Nvidia => {
                match kernel_name {
                    "FluffySeed2A" | "FluffySeed2B" => 128,
                    "FluffyRound1" | "FluffyRoundN" | "FluffyRoundNO1" | "FluffyRoundNON" => 1024,
                    "FluffyTail" | "FluffyTailO" => 1024,
                    "FluffyRecovery" => 256,
                    _ => 256, // Default
                }
            }
            GpuVendor::Amd => {
                match kernel_name {
                    "FluffySeed2A" | "FluffySeed2B" => 64,
                    "FluffyRound1" | "FluffyRoundN" | "FluffyRoundNO1" | "FluffyRoundNON" => 256,
                    "FluffyTail" | "FluffyTailO" => 256,
                    "FluffyRecovery" => 128,
                    _ => 128, // Default
                }
            }
            GpuVendor::Intel => {
                // Intel GPUs prefer smaller work groups
                match kernel_name {
                    "FluffySeed2A" | "FluffySeed2B" => 32,
                    _ => 64,
                }
            }
            GpuVendor::Unknown => 64, // Conservative default
        };
        
        // Ensure work group size doesn't exceed device limits
        let max_work_group_size = self.device_capabilities.max_work_group_size;
        let actual_size = preferred_size.min(max_work_group_size);
        
        // Ensure global work size is divisible by local work size
        if global_work_size % actual_size == 0 {
            Some(SpatialDims::One(actual_size))
        } else {
            None // Let OpenCL choose
        }
    }
    
    /// Get execution statistics for kernel
    pub fn get_execution_stats(&self, kernel_name: &str) -> Option<&ExecutionStats> {
        self.execution_stats.get(kernel_name)
    }
    
    /// Update execution statistics
    pub fn update_execution_stats(&mut self, kernel_name: &str, event: &Event, work_sizes: (Vec<usize>, Vec<usize>)) {
        if !self.profiling_enabled {
            return;
        }
        
        if let Some(stats) = self.execution_stats.get_mut(kernel_name) {
            stats.execution_count += 1;
            stats.global_work_size = work_sizes.0;
            stats.local_work_size = work_sizes.1;
            
            // Get timing information
            if let (Ok(start), Ok(end)) = (
                event.profiling_info(ProfilingInfo::Start),
                event.profiling_info(ProfilingInfo::End),
            ) {
                let execution_time = end.time().unwrap() - start.time().unwrap();
                stats.execution_time_ns = execution_time;
                
                // Update average if multiple executions
                if stats.execution_count > 1 {
                    stats.execution_time_ns = (stats.execution_time_ns + execution_time) / 2;
                }
            }
            
            if let (Ok(queued), Ok(start)) = (
                event.profiling_info(ProfilingInfo::Queued),
                event.profiling_info(ProfilingInfo::Start),
            ) {
                stats.queue_time_ns = start.time().unwrap() - queued.time().unwrap();
            }
        }
    }
    
    /// Get comprehensive performance report
    pub fn performance_report(&self) -> KernelPerformanceReport {
        let mut total_execution_time = 0u64;
        let mut kernel_reports = Vec::new();
        
        for (_name, stats) in &self.execution_stats {
            if stats.execution_count > 0 {
                total_execution_time += stats.execution_time_ns;
                kernel_reports.push(stats.clone());
            }
        }
        
        // Sort by execution time (longest first)
        kernel_reports.sort_by(|a, b| b.execution_time_ns.cmp(&a.execution_time_ns));
        
        KernelPerformanceReport {
            total_execution_time_ns: total_execution_time,
            kernel_stats: kernel_reports,
            device_name: self.device_capabilities.name.clone(),
            vendor: self.vendor,
        }
    }
}

/// Comprehensive kernel performance report
#[derive(Debug, Clone)]
pub struct KernelPerformanceReport {
    /// Total execution time across all kernels
    pub total_execution_time_ns: u64,
    /// Individual kernel statistics
    pub kernel_stats: Vec<ExecutionStats>,
    /// Device name
    pub device_name: String,
    /// GPU vendor
    pub vendor: GpuVendor,
}

impl KernelPerformanceReport {
    /// Print formatted performance report
    pub fn print_report(&self) {
        println!("\n📊 Kernel Performance Report - {} {}", 
            vendor_name(self.vendor), self.device_name);
        println!("{:=<60}", "");  // Fixed format string
        
        println!("{:<20} {:>12} {:>12} {:>10}", 
            "Kernel", "Time (ms)", "Queue (ms)", "Executions");
        println!("{:-<60}", "");  // Fixed format string
        
        for stats in &self.kernel_stats {
            println!("{:<20} {:>12.2} {:>12.2} {:>10}", 
                stats.kernel_name,
                stats.execution_time_ns as f64 / 1_000_000.0,
                stats.queue_time_ns as f64 / 1_000_000.0,
                stats.execution_count);
        }
        
        println!("{:-<60}", "");  // Fixed format string
        println!("{:<20} {:>12.2}", 
            "Total Time",
            self.total_execution_time_ns as f64 / 1_000_000.0);
    }
    
    /// Get the slowest kernel
    pub fn slowest_kernel(&self) -> Option<&ExecutionStats> {
        self.kernel_stats.first()
    }
    
    /// Get average execution time per kernel
    pub fn average_execution_time_ms(&self) -> f64 {
        if self.kernel_stats.is_empty() {
            return 0.0;
        }
        
        let total_time: u64 = self.kernel_stats.iter()
            .map(|s| s.execution_time_ns)
            .sum();
        
        (total_time as f64 / self.kernel_stats.len() as f64) / 1_000_000.0
    }
}

/// Utility functions for kernel management
pub mod utils {
    use super::*;
    
    /// Create kernel compiler for device
    pub fn create_compiler(
        context: Arc<Context>,
        device: Arc<Device>,
        device_capabilities: DeviceCapabilities,
        vendor: GpuVendor,
    ) -> KernelCompiler {
        KernelCompiler::new(context, device, device_capabilities, vendor)
    }
    
    /// Validate work group configuration
    pub fn validate_work_group_config(
        config: &WorkGroupConfig,
        device_capabilities: &DeviceCapabilities,
    ) -> Result<(), KernelError> {
        if let Some(local_size) = &config.local_work_size {
            let total_local_size = match local_size {
                SpatialDims::One(size) => *size,
                SpatialDims::Two(x, y) => x * y,
                SpatialDims::Three(x, y, z) => x * y * z,
                &SpatialDims::Unspecified => 0,  // Fixed: Added missing pattern
            };
            
            if total_local_size > device_capabilities.max_work_group_size {
                return Err(KernelError::UnsupportedWorkGroupSize {
                    requested: total_local_size,
                    max: device_capabilities.max_work_group_size,
                });
            }
        }
        
        Ok(())
    }
    
    /// Get recommended work group size for algorithm
    pub fn recommended_work_group_size(
        _algorithm: Algorithm,
        vendor: GpuVendor,
        compute_units: u32,
    ) -> usize {
        let base_size = match vendor {
            GpuVendor::Nvidia => 256,
            GpuVendor::Amd => 128,
            GpuVendor::Intel => 64,
            GpuVendor::Unknown => 64,
        };
        
        // Scale by compute units for better utilization
        let scale_factor = (compute_units as f64 / 16.0).sqrt().max(1.0).min(4.0);
        
        (base_size as f64 * scale_factor) as usize
    }
}

fn vendor_name(vendor: GpuVendor) -> &'static str {
    match vendor {
        GpuVendor::Nvidia => "NVIDIA",
        GpuVendor::Amd => "AMD",
        GpuVendor::Intel => "Intel",
        GpuVendor::Unknown => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_work_group_size_calculation() {
        // Test NVIDIA preferences
        let nvidia_size = utils::recommended_work_group_size(
            Algorithm::Cuckaroo29, 
            GpuVendor::Nvidia, 
            32
        );
        assert!(nvidia_size >= 256);
        
        // Test AMD preferences  
        let amd_size = utils::recommended_work_group_size(
            Algorithm::Cuckaroo29,
            GpuVendor::Amd,
            16
        );
        assert!(amd_size >= 128);
    }
    
    #[test]
    fn test_kernel_error_display() {
        let error = KernelError::CompilationFailed {
            kernel_name: "TestKernel".to_string(),
            error: "Syntax error".to_string(),
        };
        
        let error_str = format!("{}", error);
        assert!(error_str.contains("TestKernel"));
        assert!(error_str.contains("Syntax error"));
    }
    
    #[test]
    fn test_execution_stats_defaults() {
        let stats = ExecutionStats::default();
        assert_eq!(stats.execution_count, 0);
        assert_eq!(stats.execution_time_ns, 0);
        assert!(stats.kernel_name.is_empty());
    }
}