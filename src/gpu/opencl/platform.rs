// src/gpu/opencl/platform.rs - OpenCL Platform Detection and GPU Management
// Tree location: ./src/gpu/opencl/platform.rs

//! OpenCL platform detection and GPU compatibility validation
//! 
//! Handles multi-vendor GPU detection (NVIDIA, AMD, Intel) with automatic
//! fallback and memory requirement validation for Cuckaroo29/Cuckatoo32.
//! 
//! # Version History
//! - 0.1.0: Initial platform detection with multi-vendor support
//! - 0.1.1: Added memory validation and buffer size optimization
//! - 0.1.2: Added fallback chains and compatibility detection
//! - 0.1.3: Fixed API compatibility errors and type mismatches
//! - 0.1.4: Fixed buffer allocation calculation for proper sizes
//! 
//! # Supported Platforms
//! - NVIDIA: CUDA OpenCL (primary for 4060Ti 16GB)
//! - AMD: ROCm/OpenCL (RX 5000+, 6000+, 7000+ series)
//! - Intel: Arc series and integrated graphics
//! - Generic: Any OpenCL 1.2+ compatible device
//! 
//! # Memory Requirements
//! - Cuckaroo29: 6GB minimum, 8GB recommended
//! - Cuckatoo32: 11GB minimum, 16GB recommended
//! - Buffer overhead: ~20% for indexes and temporary storage

use ocl::{Context, Device, Platform, Queue};
use ocl::enums::{DeviceInfo, DeviceInfoResult};
use ocl::flags::CommandQueueProperties;
use crate::algorithms::Algorithm;
use crate::Graxil29Error;
use ocl::DeviceType;
use std::fmt;

/// GPU vendor identification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuVendor {
    /// NVIDIA GPUs
    Nvidia,
    /// AMD GPUs
    Amd,
    /// Intel GPUs
    Intel,
    /// Unknown or other vendors
    Unknown,
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuVendor::Nvidia => write!(f, "NVIDIA"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Unknown => write!(f, "Unknown"),
        }
    }
}

/// OpenCL platform capabilities and preferences
#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    /// Platform vendor (NVIDIA, AMD, Intel, etc.)
    pub vendor: GpuVendor,
    /// Platform name string
    pub name: String,
    /// OpenCL version supported
    pub version: String,
    /// Available devices on this platform
    pub devices: Vec<DeviceCapabilities>,
    /// Platform priority (higher = preferred)
    pub priority: u32,
}

/// Individual GPU device capabilities
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Unique ID of the device within the platform
    pub device_id: usize,
    /// Name of the device
    pub name: String,
    /// Total global memory in bytes
    pub global_memory: u64,
    /// Maximum memory allocation size in bytes
    pub max_alloc_size: u64,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Number of compute units
    pub compute_units: u32,
    /// Maximum clock frequency in MHz
    pub max_clock_freq: u32,
    /// List of supported mining algorithms
    pub supported_algorithms: Vec<Algorithm>,
    /// Optimized buffer configuration for this device
    pub buffer_config: BufferConfiguration,
    /// Vendor of the GPU
    pub vendor: GpuVendor,
    /// Backend type (e.g., "OpenCL")
    pub backend: String,
    /// Type of device (e.g., GPU, CPU)
    pub device_type: DeviceType,
}

/// Memory buffer configuration for optimal performance
#[derive(Debug, Clone)]
pub struct BufferConfiguration {
    /// Buffer A1 size (primary edge storage)
    pub buffer_a1_size: usize,
    /// Buffer A2 size (secondary edge storage)  
    pub buffer_a2_size: usize,
    /// Buffer B size (bucket storage)
    pub buffer_b_size: usize,
    /// Index buffer size
    pub index_size: usize,
    /// Maximum batch size for mining
    pub max_batch_size: u64,
}

/// Selected OpenCL context for mining
#[derive(Clone)]  // Added Clone trait
pub struct OpenClContext {
    /// OpenCL platform
    pub platform: Platform,
    /// Selected device
    pub device: Device,
    /// OpenCL context
    pub context: Context,
    /// Command queue
    pub queue: Queue,
    /// Device capabilities
    pub capabilities: DeviceCapabilities,
    /// Platform vendor
    pub vendor: GpuVendor,
}

/// Platform detection and selection
pub struct PlatformDetector {
    /// Available platforms
    platforms: Vec<PlatformCapabilities>,
    /// Vendor preference order
    vendor_preference: Vec<GpuVendor>,
}

impl PlatformDetector {
    /// Create new platform detector with default preferences
    pub fn new() -> Result<Self, Graxil29Error> {
        let mut detector = Self {
            platforms: Vec::new(),
            // Preference order: NVIDIA first (best for mining), then AMD, then Intel
            vendor_preference: vec![GpuVendor::Nvidia, GpuVendor::Amd, GpuVendor::Intel, GpuVendor::Unknown],
        };
        
        detector.detect_platforms()?;
        Ok(detector)
    }
    
    /// Create detector with custom vendor preference
    pub fn with_vendor_preference(preference: Vec<GpuVendor>) -> Result<Self, Graxil29Error> {
        let mut detector = Self {
            platforms: Vec::new(),
            vendor_preference: preference,
        };
        
        detector.detect_platforms()?;
        Ok(detector)
    }
    
    /// Detect all available OpenCL platforms and devices
    fn detect_platforms(&mut self) -> Result<(), Graxil29Error> {
        let platforms = Platform::list();
        
        if platforms.is_empty() {
            return Err(Graxil29Error::Gpu("No OpenCL platforms found. Please install GPU drivers.".to_string()));
        }
        
        tracing::info!("🔍 Detecting OpenCL platforms...");
        
        for (platform_idx, platform) in platforms.iter().enumerate() {
            match self.analyze_platform(platform, platform_idx) {
                Ok(capabilities) => {
                    tracing::info!("✅ Platform {}: {} ({} devices)", 
                        platform_idx, capabilities.name, capabilities.devices.len());
                    self.platforms.push(capabilities);
                }
                Err(e) => {
                    tracing::warn!("⚠️  Platform {} detection failed: {}", platform_idx, e);
                }
            }
        }
        
        if self.platforms.is_empty() {
            return Err(Graxil29Error::Gpu("No usable OpenCL platforms found".to_string()));
        }
        
        // Sort platforms by preference and capability
        self.platforms.sort_by(|a, b| {
            // First by vendor preference
            let a_pref = self.vendor_preference.iter().position(|&v| v == a.vendor).unwrap_or(999);
            let b_pref = self.vendor_preference.iter().position(|&v| v == b.vendor).unwrap_or(999);
            
            match a_pref.cmp(&b_pref) {
                std::cmp::Ordering::Equal => {
                    // Then by priority (total VRAM of best device)
                    b.priority.cmp(&a.priority)
                }
                other => other,
            }
        });
        
        Ok(())
    }
    
    /// Analyze a single platform and its devices
    fn analyze_platform(&self, platform: &Platform, _platform_idx: usize) -> Result<PlatformCapabilities, Graxil29Error> {
        let platform_name = platform.name()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to get platform name: {}", e)))?;
        
        let platform_version = platform.version()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to get platform version: {}", e)))?;
        
        let vendor = Self::detect_vendor(&platform_name);
        
        tracing::debug!("🔧 Analyzing platform: {} ({})", platform_name, platform_version);
        
        // Get all devices for this platform
        let devices = Device::list_all(platform)
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to list devices: {}", e)))?;
        
        let mut device_capabilities = Vec::new();
        let mut max_memory = 0u64;
        
        for (device_idx, device) in devices.iter().enumerate() {
            match self.analyze_device(device, device_idx, vendor) {
                Ok(caps) => {
                    max_memory = max_memory.max(caps.global_memory);
                    device_capabilities.push(caps);
                }
                Err(e) => {
                    tracing::debug!("Device {} analysis failed: {}", device_idx, e);
                }
            }
        }
        
        if device_capabilities.is_empty() {
            return Err(Graxil29Error::Gpu("No usable devices found on platform".to_string()));
        }
        
        // Platform priority based on best device memory
        let priority = (max_memory / (1024 * 1024 * 1024)) as u32; // GB as priority
        
        Ok(PlatformCapabilities {
            vendor,
            name: platform_name,
            version: platform_version,
            devices: device_capabilities,
            priority,
        })
    }
    
    /// Analyze individual device capabilities
    fn analyze_device(&self, device: &Device, device_idx: usize, vendor: GpuVendor) -> Result<DeviceCapabilities, Graxil29Error> {
        let name = self.get_device_info_string(device, DeviceInfo::Name)?;
        let global_memory = self.get_device_info_u64(device, DeviceInfo::GlobalMemSize)?;
        let max_alloc_size = self.get_device_info_u64(device, DeviceInfo::MaxMemAllocSize)?;
        let max_work_group_size = self.get_device_info_usize(device, DeviceInfo::MaxWorkGroupSize)?;
        let compute_units = self.get_device_info_u32(device, DeviceInfo::MaxComputeUnits)?;
        let max_clock_freq = self.get_device_info_u32(device, DeviceInfo::MaxClockFrequency)?;
        
        tracing::debug!("  📱 Device {}: {} ({:.1}GB VRAM, {} CUs)", 
            device_idx, name, global_memory as f64 / (1024.0 * 1024.0 * 1024.0), compute_units);
        
        let supported_algorithms = Self::determine_supported_algorithms(global_memory);
        
        let buffer_config = Self::calculate_buffer_config(global_memory, max_alloc_size);
        
        let device_type = match device.info(DeviceInfo::Type) {
            Ok(DeviceInfoResult::Type(t)) => t,
            _ => DeviceType::GPU, // Default to GPU
        };
        
        Ok(DeviceCapabilities {
            device_id: device_idx,
            name,
            global_memory,
            max_alloc_size,
            max_work_group_size,
            compute_units,
            max_clock_freq,
            supported_algorithms,
            buffer_config,
            vendor,
            backend: "OpenCL".to_string(),
            device_type,
        })
    }
    
    /// Detect GPU vendor from platform name
    fn detect_vendor(platform_name: &str) -> GpuVendor {
        let name_lower = platform_name.to_lowercase();
        
        if name_lower.contains("nvidia") || name_lower.contains("cuda") {
            GpuVendor::Nvidia
        } else if name_lower.contains("amd") || name_lower.contains("advanced micro devices") || 
                  name_lower.contains("rocm") || name_lower.contains("ati") {
            GpuVendor::Amd
        } else if name_lower.contains("intel") {
            GpuVendor::Intel
        } else {
            GpuVendor::Unknown
        }
    }
    
    /// Determine which algorithms are supported based on available memory
    fn determine_supported_algorithms(global_memory: u64) -> Vec<Algorithm> {
        let mut algorithms = Vec::new();
        let memory_gb = global_memory / (1024 * 1024 * 1024);
        
        // Cuckaroo29: requires 6GB minimum, 8GB recommended
        if memory_gb >= 6 {
            algorithms.push(Algorithm::Cuckaroo29);
        }
        
        // Cuckatoo32: requires 11GB minimum, 16GB recommended  
        if memory_gb >= 11 {
            algorithms.push(Algorithm::Cuckatoo32);
        }
        
        algorithms
    }
    
    /// Calculate optimal buffer configuration for available memory
    fn calculate_buffer_config(global_memory: u64, max_alloc_size: u64) -> BufferConfiguration {
        let memory_gb = global_memory / (1024 * 1024 * 1024);
        
        // Base buffer sizes (from grin-miner constants)
        const BASE_DUCK_SIZE_A: usize = 129;
        const BASE_DUCK_SIZE_B: usize = 83;
        
        // Adjust sizes based on available memory
        let (duck_a, duck_b, batch_size) = match memory_gb {
            gb if gb >= 16 => {
                // 16GB+: Full buffers, large batches (optimal for 4060Ti 16GB)
                (BASE_DUCK_SIZE_A, BASE_DUCK_SIZE_B, 2000)
            }
            gb if gb >= 11 => {
                // 11-15GB: Slightly reduced buffers
                (BASE_DUCK_SIZE_A * 90 / 100, BASE_DUCK_SIZE_B * 90 / 100, 1500)
            }
            gb if gb >= 8 => {
                // 8-10GB: Reduced buffers
                (BASE_DUCK_SIZE_A * 75 / 100, BASE_DUCK_SIZE_B * 75 / 100, 1000)
            }
            gb if gb >= 6 => {
                // 6-7GB: Minimal viable buffers
                (BASE_DUCK_SIZE_A * 60 / 100, BASE_DUCK_SIZE_B * 60 / 100, 500)
            }
            _ => {
                // <6GB: Very conservative
                (BASE_DUCK_SIZE_A * 40 / 100, BASE_DUCK_SIZE_B * 40 / 100, 100)
            }
        };
        
        // Calculate actual buffer sizes
        let buffer_a1_size = duck_a * 1024 * (4096 - 128) * 2;
        let buffer_a2_size = duck_a * 1024 * 256 * 2;
        let buffer_b_size = duck_b * 1024 * 4096 * 2;
        let index_size = 256 * 256 * 4;
        
        // Add debugging to see what's happening
        tracing::debug!("Buffer calculation: duck_a={}, duck_b={}", duck_a, duck_b);
        tracing::debug!("Raw calculated sizes: A1={}MB, A2={}MB, B={}MB", 
            buffer_a1_size / (1024*1024), 
            buffer_a2_size / (1024*1024), 
            buffer_b_size / (1024*1024));
        tracing::debug!("Max alloc size: {}MB", max_alloc_size / (1024*1024));
        
        // FIXED: Use 80% of max allocation instead of 25%
        let max_buffer_size = (max_alloc_size * 4 / 5) as usize;
        
        tracing::debug!("Max buffer size limit: {}MB", max_buffer_size / (1024*1024));
        
        let final_config = BufferConfiguration {
            buffer_a1_size: buffer_a1_size.min(max_buffer_size),
            buffer_a2_size: buffer_a2_size.min(max_buffer_size),
            buffer_b_size: buffer_b_size.min(max_buffer_size),
            index_size: index_size.min(max_buffer_size),
            max_batch_size: batch_size,
        };
        
        tracing::debug!("Final buffer sizes: A1={}MB, A2={}MB, B={}MB", 
            final_config.buffer_a1_size / (1024*1024), 
            final_config.buffer_a2_size / (1024*1024), 
            final_config.buffer_b_size / (1024*1024));
        
        final_config
    }
    
    /// Select best platform and device for mining
    pub fn select_best_device(&self, algorithm: Option<Algorithm>) -> Result<(Platform, DeviceCapabilities), Graxil29Error> {
        if self.platforms.is_empty() {
            return Err(Graxil29Error::Gpu("No platforms available".to_string()));
        }
        
        for platform_caps in &self.platforms {
            for device_caps in &platform_caps.devices {
                // Check algorithm support if specified
                if let Some(algo) = algorithm {
                    if !device_caps.supported_algorithms.contains(&algo) {
                        continue;
                    }
                }
                
                // Return first suitable device (platforms are pre-sorted by preference)
                let platform = Platform::list().into_iter()
                    .find(|p| p.name().unwrap_or_default() == platform_caps.name)
                    .ok_or_else(|| Graxil29Error::Gpu("Platform not found".to_string()))?;
                
                return Ok((platform, device_caps.clone()));
            }
        }
        
        Err(Graxil29Error::Gpu(format!(
            "No suitable device found for algorithm: {:?}", algorithm
        )))
    }
    
    /// Create OpenCL context for selected platform and device
    pub fn create_context(&self, platform_hint: Option<&str>, device_hint: Option<usize>, 
                         algorithm: Option<Algorithm>) -> Result<OpenClContext, Graxil29Error> {
        
        let (platform, device_caps) = if let Some(platform_name) = platform_hint {
            // User specified platform
            self.select_platform_by_name(platform_name, device_hint, algorithm)?
        } else {
            // Auto-select best platform
            self.select_best_device(algorithm)?
        };
        
        // Get actual device from platform
        let devices = Device::list_all(&platform)
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to list devices: {}", e)))?;
        
        let device = devices.get(device_caps.device_id)
            .ok_or_else(|| Graxil29Error::Gpu("Device not found".to_string()))?
            .clone();
        
        // Create OpenCL context
        let context = Context::builder()
            .platform(platform.clone())
            .devices(&device)
            .build()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create context: {}", e)))?;
        
        // Create command queue with profiling for debugging
        let queue_properties = if cfg!(feature = "profile") {
            Some(CommandQueueProperties::PROFILING_ENABLE)
        } else {
            None
        };
        
        // Fixed: Remove & from device parameter
        let queue = Queue::new(&context, device, queue_properties)
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create queue: {}", e)))?;
        
        let vendor = Self::detect_vendor(&platform.name().unwrap_or_default());
        
        tracing::info!("🎯 Selected: {} {} ({:.1}GB, {} algorithms)", 
            vendor_name(vendor), device_caps.name, 
            device_caps.global_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            device_caps.supported_algorithms.len());
        
        Ok(OpenClContext {
            platform,
            device: devices[device_caps.device_id].clone(),
            context,
            queue,
            capabilities: device_caps,
            vendor,
        })
    }
    
    /// Select platform by name
    fn select_platform_by_name(&self, name: &str, device_hint: Option<usize>, 
                              algorithm: Option<Algorithm>) -> Result<(Platform, DeviceCapabilities), Graxil29Error> {
        
        let platform_caps = self.platforms.iter()
            .find(|p| p.name.to_lowercase().contains(&name.to_lowercase()))
            .ok_or_else(|| Graxil29Error::Gpu(format!("Platform '{}' not found", name)))?;
        
        let device_idx = device_hint.unwrap_or(0);
        let device_caps = platform_caps.devices.get(device_idx)
            .ok_or_else(|| Graxil29Error::Gpu(format!("Device {} not found", device_idx)))?;
        
        // Validate algorithm support
        if let Some(algo) = algorithm {
            if !device_caps.supported_algorithms.contains(&algo) {
                return Err(Graxil29Error::Gpu(format!(
                    "Device does not support {:?}", algo
                )));
            }
        }
        
        let platform = Platform::list().into_iter()
            .find(|p| p.name().unwrap_or_default() == platform_caps.name)
            .ok_or_else(|| Graxil29Error::Gpu("Platform not found".to_string()))?;
        
        Ok((platform, device_caps.clone()))
    }
    
    /// List all available platforms and devices
    pub fn list_all(&self) -> &[PlatformCapabilities] {
        &self.platforms
    }
    
    // Helper methods for device info extraction
    fn get_device_info_string(&self, device: &Device, info: DeviceInfo) -> Result<String, Graxil29Error> {
        match device.info(info) {
            Ok(DeviceInfoResult::Name(s)) | 
            Ok(DeviceInfoResult::Vendor(s)) => Ok(s),
            Ok(DeviceInfoResult::Version(version)) => Ok(format!("{:?}", version)),
            _ => Err(Graxil29Error::Gpu("Failed to get device string info".to_string()))
        }
    }
    
    fn get_device_info_u64(&self, device: &Device, info: DeviceInfo) -> Result<u64, Graxil29Error> {
        match device.info(info) {
            Ok(DeviceInfoResult::GlobalMemSize(v)) |
            Ok(DeviceInfoResult::MaxMemAllocSize(v)) => Ok(v),
            _ => Err(Graxil29Error::Gpu("Failed to get device u64 info".to_string()))
        }
    }
    
    fn get_device_info_usize(&self, device: &Device, info: DeviceInfo) -> Result<usize, Graxil29Error> {
        match device.info(info) {
            Ok(DeviceInfoResult::MaxWorkGroupSize(v)) => Ok(v),
            _ => Err(Graxil29Error::Gpu("Failed to get device usize info".to_string()))
        }
    }
    
    fn get_device_info_u32(&self, device: &Device, info: DeviceInfo) -> Result<u32, Graxil29Error> {
        match device.info(info) {
            Ok(DeviceInfoResult::MaxComputeUnits(v)) |
            Ok(DeviceInfoResult::MaxClockFrequency(v)) => Ok(v),
            _ => Err(Graxil29Error::Gpu("Failed to get device u32 info".to_string()))
        }
    }
}

/// Get human-readable vendor name
fn vendor_name(vendor: GpuVendor) -> &'static str {
    match vendor {
        GpuVendor::Nvidia => "NVIDIA",
        GpuVendor::Amd => "AMD",
        GpuVendor::Intel => "Intel",
        GpuVendor::Unknown => "Unknown",
    }
}

/// Validate that platform meets minimum requirements
pub fn validate_platform_requirements(capabilities: &DeviceCapabilities, algorithm: Algorithm) -> Result<(), Graxil29Error> {
    let memory_gb = capabilities.global_memory / (1024 * 1024 * 1024);
    
    let (min_memory, algo_name) = match algorithm {
        Algorithm::Cuckaroo29 => (6, "Cuckaroo29"),
        Algorithm::Cuckatoo32 => (11, "Cuckatoo32"),
    };
    
    if memory_gb < min_memory {
        return Err(Graxil29Error::Gpu(format!(
            "{} requires {}GB+ VRAM, but device has only {}GB",
            algo_name, min_memory, memory_gb
        )));
    }
    
    // Validate compute units (need reasonable parallelism)
    if capabilities.compute_units < 8 {
        tracing::warn!("⚠️  Low compute units ({}), performance may be poor", 
            capabilities.compute_units);
    }
    
    // Validate OpenCL version (need 1.2+ for atomics)
    // Note: This would require platform version check, skipping for now
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vendor_detection() {
        assert_eq!(PlatformDetector::detect_vendor("NVIDIA CUDA"), GpuVendor::Nvidia);
        assert_eq!(PlatformDetector::detect_vendor("AMD Accelerated Parallel Processing"), GpuVendor::Amd);
        assert_eq!(PlatformDetector::detect_vendor("Intel(R) OpenCL"), GpuVendor::Intel);
        assert_eq!(PlatformDetector::detect_vendor("Unknown Platform"), GpuVendor::Unknown);
    }
    
    #[test]
    fn test_algorithm_support() {
        let algorithms_6gb = PlatformDetector::determine_supported_algorithms(6 * 1024 * 1024 * 1024);
        assert_eq!(algorithms_6gb, vec![Algorithm::Cuckaroo29]);
        
        let algorithms_16gb = PlatformDetector::determine_supported_algorithms(16 * 1024 * 1024 * 1024);
        assert_eq!(algorithms_16gb, vec![Algorithm::Cuckaroo29, Algorithm::Cuckatoo32]);
    }
}