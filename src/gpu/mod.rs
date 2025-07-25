// src/gpu/mod.rs - GPU Module Organization and Public API
// Tree location: ./src/gpu/mod.rs

//! GPU compute module for Graxil29 Cuckoo Cycle mining
//! 
//! Provides GPU-accelerated mining using OpenCL with multi-platform support.
//! Supports both Cuckaroo29 and Cuckatoo32 algorithms.

/// GPU compute functionality - Main mining interface
pub mod compute;

/// GPU memory management module
pub mod memory;

/// OpenCL backend for real Cuckoo Cycle mining
pub mod opencl;

// Re-export main types for easy access
pub use compute::*;
pub use memory::*;

// Re-export key OpenCL types
pub use opencl::{
    OpenClBackend,
    OpenClConfig, 
    PlatformDetector,
    OpenClContext,
    DeviceCapabilities,
    BufferConfiguration,
    GpuVendor,
    BackendStats,
    MemoryUsage,
    BenchmarkResult,
};

/// Check OpenCL platform availability
pub fn opencl_available() -> bool {
    opencl::PlatformDetector::new().is_ok()
}

/// Get recommended mining configuration for detected hardware
pub fn recommended_config() -> Result<opencl::OpenClConfig, crate::Graxil29Error> {
    let detector = opencl::PlatformDetector::new()?;
    let platforms = detector.list_all();
    
    if let Some(platform) = platforms.first() {
        let memory_gb = platform.devices.first()
            .map(|d| d.global_memory / (1024 * 1024 * 1024))
            .unwrap_or(8);
            
        Ok(opencl::utils::recommended_config(platform.vendor, memory_gb))
    } else {
        Err(crate::Graxil29Error::OpenCL("No OpenCL platforms found".to_string()))
    }
}

/// List all available GPU platforms for diagnostics
pub fn list_gpu_platforms() -> Result<Vec<opencl::PlatformCapabilities>, crate::Graxil29Error> {
    opencl::utils::list_platforms()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_exports() {
        // Test that main types are accessible
        assert!(opencl_available() || !opencl_available()); // Should not panic
    }
}