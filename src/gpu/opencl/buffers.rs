// src/gpu/opencl/buffers.rs - OpenCL Buffer Management for Cuckoo Cycle Mining
// Tree location: ./src/gpu/opencl/buffers.rs

//! OpenCL buffer management utilities for optimal GPU memory usage
//! 
//! Provides high-level buffer management with automatic sizing, lifecycle
//! management, and platform-specific optimizations for Cuckaroo29/Cuckatoo32.
//! 
//! # Version History
//! - 0.1.0: Initial buffer management with automatic sizing
//! - 0.1.1: Added buffer pools and memory tracking
//! - 0.1.2: Enhanced error handling and validation
//! - 0.1.3: Fixed buffer allocation and read operations for stable mining
//! 
//! # Buffer Types
//! - **A1**: Primary edge storage (largest buffer)
//! - **A2**: Secondary edge storage  
//! - **B**: Bucket storage for trimming
//! - **Index**: Edge counting and bucket management
//! - **Recovery**: Solution cycle storage
//! - **Nonces**: Nonce recovery results
//! 
//! # Memory Layout Optimization
//! - 16GB GPU: Full buffers for maximum performance
//! - 8-11GB GPU: Reduced buffers with maintained efficiency
//! - 6GB GPU: Minimal viable configuration

use ocl::{Buffer, Queue, Context};
use ocl::flags::MemFlags;
use std::collections::HashMap;
use std::sync::Arc;
use crate::algorithms::Algorithm;
use crate::Graxil29Error;
use super::platform::{BufferConfiguration, DeviceCapabilities, GpuVendor};

/// OpenCL buffer management errors
#[derive(Debug, Clone)]
pub enum OpenClBufferError {
    /// Insufficient GPU memory for requested buffers
    InsufficientMemory { 
        /// Required memory in bytes
        required: u64, 
        /// Available memory in bytes
        available: u64 
    },
    /// Buffer allocation failed
    AllocationFailed(String),
    /// Invalid buffer configuration
    InvalidConfiguration(String),
    /// Buffer size exceeds platform limits
    ExceedsLimits { 
        /// Requested size in bytes
        size: usize, 
        /// Maximum allocation size in bytes
        max_alloc: u64 
    },
    /// Memory fragmentation issues
    FragmentationError(String),
}

impl std::fmt::Display for OpenClBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientMemory { required, available } => {
                write!(f, "Insufficient GPU memory: need {}MB, have {}MB", 
                    required / (1024 * 1024), available / (1024 * 1024))
            }
            Self::AllocationFailed(msg) => write!(f, "Buffer allocation failed: {}", msg),
            Self::InvalidConfiguration(msg) => write!(f, "Invalid buffer config: {}", msg),
            Self::ExceedsLimits { size, max_alloc } => {
                write!(f, "Buffer size {}MB exceeds max allocation {}MB", 
                    size / (1024 * 1024), max_alloc / (1024 * 1024))
            }
            Self::FragmentationError(msg) => write!(f, "Memory fragmentation: {}", msg),
        }
    }
}

impl std::error::Error for OpenClBufferError {}

/// Buffer usage patterns for optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BufferUsage {
    /// Read-only buffer (input data)
    ReadOnly,
    /// Write-only buffer (output data)
    WriteOnly,
    /// Read-write buffer (intermediate processing)
    ReadWrite,
    /// Scratch buffer (temporary storage)
    Scratch,
}

impl BufferUsage {
    /// Convert to OpenCL memory flags
    pub fn to_mem_flags(self) -> MemFlags {
        match self {
            Self::ReadOnly => MemFlags::READ_ONLY,
            Self::WriteOnly => MemFlags::WRITE_ONLY,
            Self::ReadWrite => MemFlags::READ_WRITE,
            Self::Scratch => MemFlags::READ_WRITE,
        }
    }
}

/// Buffer specification for creation
#[derive(Debug, Clone)]
pub struct BufferSpec {
    /// Buffer name for identification
    pub name: String,
    /// Size in bytes
    pub size: usize,
    /// Usage pattern
    pub usage: BufferUsage,
    /// Element type (u32, u64, etc.)
    pub element_type: BufferElementType,
    /// Whether buffer needs to be initialized with zeros
    pub zero_initialize: bool,
    /// Platform-specific alignment requirements
    pub alignment: Option<usize>,
}

/// Supported buffer element types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BufferElementType {
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 32-bit signed integer
    I32,
    /// 32-bit floating point
    F32,
    /// For edge pairs
    Uint2,
    /// For vectorized operations
    Ulong4,
}

impl BufferElementType {
    /// Get size in bytes
    pub fn size_bytes(self) -> usize {
        match self {
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 => 8,
            Self::Uint2 => 8,  // 2 * u32
            Self::Ulong4 => 32, // 4 * u64
        }
    }
    
    /// Get element count for given byte size
    pub fn element_count(self, size_bytes: usize) -> usize {
        size_bytes / self.size_bytes()
    }
}

/// Collection of related buffers for mining
#[derive(Debug)]
pub struct BufferSet {
    /// Primary edge buffer A1
    pub buffer_a1: Buffer<u32>,
    /// Secondary edge buffer A2
    pub buffer_a2: Buffer<u32>,
    /// Bucket storage buffer B
    pub buffer_b: Buffer<u32>,
    /// Index buffer 1
    pub buffer_i1: Buffer<u32>,
    /// Index buffer 2
    pub buffer_i2: Buffer<u32>,
    /// Recovery buffer for solution edges
    pub buffer_recovery: Buffer<u32>,
    /// Nonce recovery buffer
    pub buffer_nonces: Buffer<u32>,
    /// Buffer configuration used
    pub config: BufferConfiguration,
    /// Total memory allocated
    pub total_memory: usize,
}

/// Buffer pool for efficient memory reuse
pub struct BufferPool {
    /// Available buffers by size
    available_buffers: HashMap<usize, Vec<Buffer<u32>>>,
    /// Pool statistics
    stats: PoolStats,
}

/// Buffer pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total allocations requested
    pub allocations_requested: u64,
    /// Allocations served from pool
    pub allocations_from_pool: u64,
    /// New allocations created
    pub allocations_created: u64,
    /// Buffers returned to pool
    pub buffers_returned: u64,
    /// Current pool size
    pub current_pool_size: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
}

/// High-level buffer manager
pub struct BufferManager {
    /// OpenCL queue
    queue: Arc<Queue>,
    /// OpenCL context  
    _context: Arc<Context>,
    /// Device capabilities
    device_capabilities: DeviceCapabilities,
    /// Buffer pool for reuse
    pool: BufferPool,
    /// Memory tracking
    memory_tracker: MemoryTracker,
    /// Vendor-specific optimizations
    vendor: GpuVendor,
}

/// Memory usage tracking
#[derive(Debug, Clone)]
struct MemoryTracker {
    /// Total allocated memory
    total_allocated: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Active allocations by name
    active_allocations: HashMap<String, usize>,
    /// Allocation history
    allocation_count: u64,
}

impl BufferManager {
    /// Create new buffer manager
    pub fn new(
        context: Arc<Context>, 
        queue: Arc<Queue>, 
        device_capabilities: DeviceCapabilities,
        vendor: GpuVendor,
    ) -> Result<Self, Graxil29Error> {
        
        tracing::info!("🔧 Initializing buffer manager for {} {}", 
            vendor_name(vendor), device_capabilities.name);
        
        Ok(Self {
            queue,
            _context: context,
            device_capabilities,
            pool: BufferPool::new(),
            memory_tracker: MemoryTracker::new(),
            vendor,
        })
    }
    
    /// Create complete buffer set for mining algorithm
    pub fn create_mining_buffers(&mut self, algorithm: Algorithm) -> Result<BufferSet, OpenClBufferError> {
        tracing::info!("📦 Creating mining buffers for {:?}", algorithm);
        
        let config = self.device_capabilities.buffer_config.clone();
        
        // Log detailed buffer information for debugging
        tracing::info!("Buffer allocation plan:");
        tracing::info!("  A1 (Primary edges): {}MB", config.buffer_a1_size / (1024 * 1024));
        tracing::info!("  A2 (Secondary edges): {}MB", config.buffer_a2_size / (1024 * 1024));
        tracing::info!("  B (Bucket storage): {}MB", config.buffer_b_size / (1024 * 1024));
        tracing::info!("  Index buffers: {}KB each", config.index_size / 1024);
        
        // Validate total memory requirements
        self.validate_memory_requirements(&config)?;
        
        // Create buffer specifications
        let specs = self.create_buffer_specs(&config, algorithm)?;
        
        // Allocate buffers with comprehensive error handling
        let buffers = self.allocate_buffer_set(&specs)?;
        
        let total_memory = specs.iter().map(|spec| spec.size).sum();
        
        tracing::info!("✅ Mining buffers created successfully: {:.1}MB total", 
            total_memory as f64 / (1024.0 * 1024.0));
        
        Ok(BufferSet {
            buffer_a1: buffers.get("A1").unwrap().clone(),
            buffer_a2: buffers.get("A2").unwrap().clone(),
            buffer_b: buffers.get("B").unwrap().clone(),
            buffer_i1: buffers.get("I1").unwrap().clone(),
            buffer_i2: buffers.get("I2").unwrap().clone(),
            buffer_recovery: buffers.get("Recovery").unwrap().clone(),
            buffer_nonces: buffers.get("Nonces").unwrap().clone(),
            config: config.clone(),
            total_memory,
        })
    }
    
    /// Create buffer specifications for algorithm
    fn create_buffer_specs(&self, config: &BufferConfiguration, algorithm: Algorithm) -> Result<Vec<BufferSpec>, OpenClBufferError> {
        let mut specs = Vec::new();
        
        // Buffer A1: Primary edge storage
        specs.push(BufferSpec {
            name: "A1".to_string(),
            size: config.buffer_a1_size,
            usage: BufferUsage::ReadWrite,
            element_type: BufferElementType::U32,
            zero_initialize: true,
            alignment: self.get_optimal_alignment(),
        });
        
        // Buffer A2: Secondary edge storage
        specs.push(BufferSpec {
            name: "A2".to_string(),
            size: config.buffer_a2_size,
            usage: BufferUsage::ReadWrite,
            element_type: BufferElementType::U32,
            zero_initialize: true,
            alignment: self.get_optimal_alignment(),
        });
        
        // Buffer B: Bucket storage
        specs.push(BufferSpec {
            name: "B".to_string(),
            size: config.buffer_b_size,
            usage: BufferUsage::ReadWrite,
            element_type: BufferElementType::U32,
            zero_initialize: true,
            alignment: self.get_optimal_alignment(),
        });
        
        // Index buffers
        specs.push(BufferSpec {
            name: "I1".to_string(),
            size: config.index_size,
            usage: BufferUsage::ReadWrite,
            element_type: BufferElementType::U32,
            zero_initialize: true,
            alignment: None,
        });
        
        specs.push(BufferSpec {
            name: "I2".to_string(),
            size: config.index_size,
            usage: BufferUsage::ReadWrite,
            element_type: BufferElementType::U32,
            zero_initialize: true,
            alignment: None,
        });
        
        // Recovery buffer (42 edges * 2 nodes * 4 bytes)
        specs.push(BufferSpec {
            name: "Recovery".to_string(),
            size: 42 * 2 * 4,
            usage: BufferUsage::ReadOnly,
            element_type: BufferElementType::U32,
            zero_initialize: false,
            alignment: None,
        });
        
        // Nonces buffer
        specs.push(BufferSpec {
            name: "Nonces".to_string(),
            size: config.index_size,
            usage: BufferUsage::WriteOnly,
            element_type: BufferElementType::U32,
            zero_initialize: true,
            alignment: None,
        });
        
        // Algorithm-specific adjustments
        match algorithm {
            Algorithm::Cuckatoo32 => {
                // Cuckatoo32 may need larger buffers for 32-bit edge space
                for spec in &mut specs {
                    if spec.name == "A1" || spec.name == "B" {
                        // Slight increase for 32-bit edge space
                        spec.size = (spec.size as f64 * 1.05) as usize;
                    }
                }
            }
            Algorithm::Cuckaroo29 => {
                // Default configuration is optimized for Cuckaroo29
            }
        }
        
        Ok(specs)
    }
    
    /// Allocate complete buffer set
    fn allocate_buffer_set(&mut self, specs: &[BufferSpec]) -> Result<HashMap<String, Buffer<u32>>, OpenClBufferError> {
        let mut buffers = HashMap::new();
        
        for spec in specs {
            tracing::debug!("Allocating buffer {}: {:.1}MB", 
                spec.name, spec.size as f64 / (1024.0 * 1024.0));
            
            let buffer = self.allocate_buffer(spec)?;
            buffers.insert(spec.name.clone(), buffer);
            
            tracing::debug!("✅ Buffer {} allocated successfully", spec.name);
        }
        
        Ok(buffers)
    }
    
    /// Allocate individual buffer with optimization
    pub fn allocate_buffer(&mut self, spec: &BufferSpec) -> Result<Buffer<u32>, OpenClBufferError> {
        // Validate buffer size
        self.validate_buffer_spec(spec)?;
        
        // Try to get from pool first
        if let Some(buffer) = self.pool.get_buffer(spec.size) {
            self.memory_tracker.track_allocation(&spec.name, spec.size);
            return Ok(buffer);
        }
        
        // Create new buffer with robust error handling
        let element_count = spec.element_type.element_count(spec.size);
        let flags = spec.usage.to_mem_flags();
        
        tracing::debug!("Creating buffer {}: {} elements ({} bytes), flags: {:?}", 
            spec.name, element_count, spec.size, flags);
        
        let buffer_result = if spec.zero_initialize {
            Buffer::<u32>::builder()
                .queue(self.queue.as_ref().clone())
                .len(element_count)
                .flags(flags)
                .fill_val(0u32)
                .build()
        } else {
            Buffer::<u32>::builder()
                .queue(self.queue.as_ref().clone())
                .len(element_count)
                .flags(flags)
                .build()
        };
        
        let buffer = buffer_result.map_err(|e| {
            let error_msg = format!("OpenCL buffer creation failed for {}: {}", spec.name, e);
            tracing::error!("❌ {}", error_msg);
            OpenClBufferError::AllocationFailed(error_msg)
        })?;
        
        // Update tracking
        self.memory_tracker.track_allocation(&spec.name, spec.size);
        self.pool.stats.allocations_created += 1;
        
        tracing::debug!("✅ Buffer {} allocated: {} elements", spec.name, element_count);
        
        Ok(buffer)
    }
    
    /// Validate buffer specification
    fn validate_buffer_spec(&self, spec: &BufferSpec) -> Result<(), OpenClBufferError> {
        // Check against max allocation size
        if spec.size as u64 > self.device_capabilities.max_alloc_size {
            return Err(OpenClBufferError::ExceedsLimits {
                size: spec.size,
                max_alloc: self.device_capabilities.max_alloc_size,
            });
        }
        
        // Check element alignment
        if spec.size % spec.element_type.size_bytes() != 0 {
            return Err(OpenClBufferError::InvalidConfiguration(
                format!("Buffer size {} not aligned to element size {}", 
                    spec.size, spec.element_type.size_bytes())
            ));
        }
        
        // Check minimum size
        if spec.size < 64 {
            return Err(OpenClBufferError::InvalidConfiguration(
                "Buffer size too small (minimum 64 bytes)".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Validate total memory requirements
    fn validate_memory_requirements(&self, config: &BufferConfiguration) -> Result<(), OpenClBufferError> {
        let total_required = config.buffer_a1_size + 
                           config.buffer_a2_size + 
                           config.buffer_b_size + 
                           config.index_size * 2 + 
                           1024; // Recovery and nonces buffers
        
        let available_memory = self.device_capabilities.global_memory;
        let safety_margin = available_memory / 10; // 10% safety margin
        
        tracing::info!("Memory validation: requiring {:.1}MB, available {:.1}MB (safety margin: {:.1}MB)", 
            total_required as f64 / (1024.0 * 1024.0),
            available_memory as f64 / (1024.0 * 1024.0),
            safety_margin as f64 / (1024.0 * 1024.0));
        
        if total_required as u64 > available_memory - safety_margin {
            return Err(OpenClBufferError::InsufficientMemory {
                required: total_required as u64,
                available: available_memory - safety_margin,
            });
        }
        
        Ok(())
    }
    
    /// Get optimal memory alignment for platform
    fn get_optimal_alignment(&self) -> Option<usize> {
        match self.vendor {
            GpuVendor::Nvidia => Some(128),    // NVIDIA prefers 128-byte alignment
            GpuVendor::Amd => Some(256),       // AMD prefers 256-byte alignment
            GpuVendor::Intel => Some(64),      // Intel is less picky
            GpuVendor::Unknown => Some(128),   // Safe default
        }
    }
    
    /// Clear all buffers in set (fill with zeros)
    pub fn clear_buffer_set(&self, buffer_set: &BufferSet) -> Result<(), Graxil29Error> {
        tracing::debug!("🧹 Clearing mining buffers");
        
        // Clear buffers with proper error handling
        buffer_set.buffer_a1.cmd().fill(0u32, None).enq()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to clear A1 buffer: {}", e)))?;
        buffer_set.buffer_a2.cmd().fill(0u32, None).enq()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to clear A2 buffer: {}", e)))?;
        buffer_set.buffer_b.cmd().fill(0u32, None).enq()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to clear B buffer: {}", e)))?;
        buffer_set.buffer_i1.cmd().fill(0u32, None).enq()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to clear I1 buffer: {}", e)))?;
        buffer_set.buffer_i2.cmd().fill(0u32, None).enq()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to clear I2 buffer: {}", e)))?;
        
        // Ensure all operations complete
        self.queue.finish().map_err(|e| Graxil29Error::Gpu(format!("Failed to finish buffer clear: {}", e)))?;
        
        tracing::debug!("✅ All mining buffers cleared successfully");
        Ok(())
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.memory_tracker.total_allocated,
            peak_usage: self.memory_tracker.peak_usage,
            pool_stats: self.pool.stats.clone(),
            device_total_memory: self.device_capabilities.global_memory,
            device_max_alloc: self.device_capabilities.max_alloc_size,
        }
    }
    
    /// Release buffer back to pool
    pub fn release_buffer(&mut self, name: &str, buffer: Buffer<u32>) {
        let size = buffer.len() * 4; // Assuming u32 elements
        self.pool.return_buffer(size, buffer);
        self.memory_tracker.release_allocation(name);
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total currently allocated memory
    pub total_allocated: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Buffer pool statistics
    pub pool_stats: PoolStats,
    /// Device total memory
    pub device_total_memory: u64,
    /// Device maximum allocation size
    pub device_max_alloc: u64,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            available_buffers: HashMap::new(),
            stats: PoolStats::default(),
        }
    }
    
    fn get_buffer(&mut self, size: usize) -> Option<Buffer<u32>> {
        self.stats.allocations_requested += 1;
        
        if let Some(buffers) = self.available_buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                self.stats.allocations_from_pool += 1;
                return Some(buffer);
            }
        }
        
        None
    }
    
    fn return_buffer(&mut self, size: usize, buffer: Buffer<u32>) {
        self.available_buffers.entry(size).or_insert_with(Vec::new).push(buffer);
        self.stats.buffers_returned += 1;
        self.update_pool_size();
    }
    
    fn update_pool_size(&mut self) {
        self.stats.current_pool_size = self.available_buffers.values()
            .map(|buffers| buffers.len())
            .sum();
    }
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            total_allocated: 0,
            peak_usage: 0,
            active_allocations: HashMap::new(),
            allocation_count: 0,
        }
    }
    
    fn track_allocation(&mut self, name: &str, size: usize) {
        self.active_allocations.insert(name.to_string(), size);
        self.total_allocated += size;
        self.peak_usage = self.peak_usage.max(self.total_allocated);
        self.allocation_count += 1;
    }
    
    fn release_allocation(&mut self, name: &str) {
        if let Some(size) = self.active_allocations.remove(name) {
            self.total_allocated = self.total_allocated.saturating_sub(size);
        }
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

/// Utility functions for buffer management
pub mod utils {
    use super::*;
    
    /// Calculate optimal buffer sizes for available memory
    pub fn calculate_optimal_sizes(
        available_memory: u64,
        _algorithm: Algorithm,
        vendor: GpuVendor,
    ) -> BufferConfiguration {
        let memory_gb = available_memory / (1024 * 1024 * 1024);
        
        tracing::info!("Calculating optimal buffer sizes for {}GB {} GPU", 
            memory_gb, vendor_name(vendor));
        
        // Base sizes from grin-miner (proven stable values)
        let base_duck_a = 129;
        let base_duck_b = 83;
        
        // Scale factors based on available memory
        let (scale_factor, batch_size) = match memory_gb {
            gb if gb >= 16 => (1.0, 2000),       // Full performance for 16GB
            gb if gb >= 11 => (0.9, 1500),      // Slight reduction for 11-15GB
            gb if gb >= 8 => (0.75, 1000),      // Significant reduction for 8-10GB
            gb if gb >= 6 => (0.6, 500),        // Minimal viable for 6-7GB
            _ => (0.4, 100),                     // Very conservative for <6GB
        };
        
        // Vendor-specific optimizations
        let vendor_scale = match vendor {
            GpuVendor::Nvidia => 1.0,           // NVIDIA handles large buffers well
            GpuVendor::Amd => 0.95,             // AMD slightly more conservative
            GpuVendor::Intel => 0.9,            // Intel more conservative
            GpuVendor::Unknown => 0.85,         // Unknown vendor - very safe
        };
        
        let final_scale = scale_factor * vendor_scale;
        
        // Calculate buffer sizes with conservative multipliers
        let duck_a = (base_duck_a as f64 * final_scale) as usize;
        let duck_b = (base_duck_b as f64 * final_scale) as usize;
        
        // Use proven stable multipliers from grin-miner
        let config = BufferConfiguration {
            buffer_a1_size: duck_a * 1024 * (4096 - 128) * 2,  // Slightly reduced for stability
            buffer_a2_size: duck_a * 1024 * 256 * 2,
            buffer_b_size: duck_b * 1024 * (4096 - 128) * 2,   // Slightly reduced for stability
            index_size: 256 * 256 * 4,
            max_batch_size: batch_size,
        };
        
        tracing::info!("Generated buffer configuration:");
        tracing::info!("  A1: {:.1}MB", config.buffer_a1_size as f64 / (1024.0 * 1024.0));
        tracing::info!("  A2: {:.1}MB", config.buffer_a2_size as f64 / (1024.0 * 1024.0));
        tracing::info!("  B:  {:.1}MB", config.buffer_b_size as f64 / (1024.0 * 1024.0));
        tracing::info!("  Index: {}KB", config.index_size / 1024);
        tracing::info!("  Max batch: {}", config.max_batch_size);
        
        config
    }
    
    /// Validate buffer configuration for device
    pub fn validate_configuration(
        config: &BufferConfiguration,
        device_caps: &DeviceCapabilities,
    ) -> Result<(), OpenClBufferError> {
        let total_size = config.buffer_a1_size + 
                        config.buffer_a2_size + 
                        config.buffer_b_size + 
                        config.index_size * 2;
        
        // Check total memory (use 80% limit for safety)
        if total_size as u64 > device_caps.global_memory * 8 / 10 {
            return Err(OpenClBufferError::InsufficientMemory {
                required: total_size as u64,
                available: device_caps.global_memory * 8 / 10,
            });
        }
        
        // Check individual buffer limits
        let max_buffer_size = config.buffer_a1_size.max(config.buffer_b_size);
        if max_buffer_size as u64 > device_caps.max_alloc_size {
            return Err(OpenClBufferError::ExceedsLimits {
                size: max_buffer_size,
                max_alloc: device_caps.max_alloc_size,
            });
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buffer_element_types() {
        assert_eq!(BufferElementType::U32.size_bytes(), 4);
        assert_eq!(BufferElementType::U64.size_bytes(), 8);
        assert_eq!(BufferElementType::Uint2.size_bytes(), 8);
        assert_eq!(BufferElementType::Ulong4.size_bytes(), 32);
    }
    
    #[test]
    fn test_buffer_usage_flags() {
        assert_eq!(BufferUsage::ReadOnly.to_mem_flags(), MemFlags::READ_ONLY);
        assert_eq!(BufferUsage::WriteOnly.to_mem_flags(), MemFlags::WRITE_ONLY);
        assert_eq!(BufferUsage::ReadWrite.to_mem_flags(), MemFlags::READ_WRITE);
    }
    
    #[test]
    fn test_optimal_size_calculation() {
        let memory_16gb = 16 * 1024 * 1024 * 1024;
        let config = utils::calculate_optimal_sizes(
            memory_16gb, 
            Algorithm::Cuckaroo29, 
            GpuVendor::Nvidia
        );
        
        // Should get full-size buffers for 16GB
        assert!(config.max_batch_size >= 1500);
        assert!(config.buffer_a1_size > 100_000_000); // Should be substantial
    }
}