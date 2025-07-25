// src/gpu/opencl/trimmer.rs - OpenCL Trimmer Engine for Cuckoo Cycle Mining
// Tree location: ./src/gpu/opencl/trimmer.rs

//! OpenCL trimmer implementation for Cuckaroo29/Cuckatoo32 mining
//! 
//! Based on proven grin-miner implementation with adaptations for graxil29.
//! Implements the complete 120+ round graph trimming algorithm that reduces
//! millions of edges down to a small set containing potential 42-cycles.
//! 
//! # Version History
//! - 0.1.0: Initial port from grin-miner trimmer.rs
//! - 0.1.1: Added platform integration and adaptive buffer sizing
//! - 0.1.2: Enhanced error handling and performance monitoring
//! - 0.1.3: Fixed kernel work sizes and added safety checks
//! - 0.1.4: Rewrote based on working grin-miner patterns

use ocl::{Buffer, Device, EventList, Kernel, Program, Queue, SpatialDims};
use ocl::enums::{ArgVal, DeviceInfo, DeviceInfoResult};
use ocl::flags::MemFlags;
use ocl::prm::{Uint2, Ulong4};
use std::collections::HashMap;
use std::env;
use std::time::Instant;
use blake2_rfc::blake2b::blake2b;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;
use crate::Graxil29Error;
use super::platform::{OpenClContext, DeviceCapabilities, GpuVendor};

// Proven constants from grin-miner
const DUCK_SIZE_A: usize = 129; // AMD 126 + 3
const DUCK_SIZE_B: usize = 83;
const BUFFER_SIZE_A1: usize = DUCK_SIZE_A * 1024 * (4096 - 128) * 2;
const BUFFER_SIZE_A2: usize = DUCK_SIZE_A * 1024 * 256 * 2;
const BUFFER_SIZE_B: usize = DUCK_SIZE_B * 1024 * 4096 * 2;
const INDEX_SIZE: usize = 256 * 256 * 4;

/// OpenCL buffer management parameters
#[derive(Debug, Clone)]
struct ClBufferParams {
    size: usize,
    flags: MemFlags,
}

/// Trimming statistics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct TrimmingStats {
    /// Total trimming time in milliseconds
    pub trimming_time_ms: u64,
    /// Edge generation time
    pub seed_time_ms: u64,
    /// Trimming rounds time
    pub rounds_time_ms: u64,
    /// Final collection time
    pub tail_time_ms: u64,
    /// Number of edges after trimming
    pub edges_remaining: usize,
    /// Number of trimming rounds executed
    pub rounds_executed: u32,
}

/// OpenCL Trimmer for Cuckoo Cycle mining
pub struct Trimmer {
    /// OpenCL command queue
    q: Queue,
    /// Compiled OpenCL program
    program: Program,
    /// Primary edge buffer (A1)
    buffer_a1: Buffer<u32>,
    /// Secondary edge buffer (A2)  
    buffer_a2: Buffer<u32>,
    /// Bucket storage buffer (B)
    buffer_b: Buffer<u32>,
    /// Index buffer 1
    buffer_i1: Buffer<u32>,
    /// Index buffer 2
    buffer_i2: Buffer<u32>,
    /// Recovery buffer for solution edges
    buffer_r: Buffer<u32>,
    /// Nonce recovery buffer
    buffer_nonces: Buffer<u32>,
    /// Device capabilities
    _device_capabilities: DeviceCapabilities,
    /// Performance statistics
    stats: TrimmingStats,
    /// Whether this is NVIDIA (for workgroup size optimization)
    is_nvidia: bool,
    /// Device name for reporting
    pub device_name: String,
    /// Device ID for reporting
    pub device_id: usize,
}

// Profiling macros for performance measurement
macro_rules! clear_buffer {
    ($buf:expr) => {
        $buf.cmd().fill(0, None).enq()?;
    };
}

macro_rules! kernel_enq {
    ($kernel:expr, $event_list:expr, $names:expr, $msg:expr) => {
        #[cfg(feature = "profile")]
        {
            unsafe { $kernel.cmd().enew(&mut $event_list).enq()?; }
            $names.push($msg);
        }
        #[cfg(not(feature = "profile"))]
        {
            unsafe { $kernel.cmd().enq()?; }
        }
    };
}

macro_rules! kernel_builder {
    ($obj: expr, $kernel: expr, $global_work_size: expr) => {
        Kernel::builder()
            .name($kernel)
            .program(&$obj.program)
            .queue($obj.q.clone())
            .global_work_size($global_work_size)
    };
}

impl Trimmer {
    /// Create new trimmer from OpenCL context
    pub fn new(context: OpenClContext) -> Result<Self, Graxil29Error> {
        unsafe { Self::setup_environment(); }
        
        let device_capabilities = context.capabilities.clone();
        let vendor = context.vendor;
        let is_nvidia = vendor == GpuVendor::Nvidia;
        
        tracing::info!("🔧 Initializing OpenCL trimmer for {} {}", 
            vendor_name(vendor), device_capabilities.name);
        
        // Use proven grin-miner buffer sizes
        let mut buffers = HashMap::new();
        buffers.insert("A1".to_string(), ClBufferParams {
            size: BUFFER_SIZE_A1,
            flags: MemFlags::empty(),
        });
        buffers.insert("A2".to_string(), ClBufferParams {
            size: BUFFER_SIZE_A2,
            flags: MemFlags::empty(),
        });
        buffers.insert("B".to_string(), ClBufferParams {
            size: BUFFER_SIZE_B,
            flags: MemFlags::empty(),
        });
        buffers.insert("I1".to_string(), ClBufferParams {
            size: INDEX_SIZE,
            flags: MemFlags::empty(),
        });
        buffers.insert("I2".to_string(), ClBufferParams {
            size: INDEX_SIZE,
            flags: MemFlags::empty(),
        });
        buffers.insert("R".to_string(), ClBufferParams {
            size: 42 * 2,  // Fixed: was 42 * 2 * 4
            flags: MemFlags::empty().read_only(),
        });
        buffers.insert("NONCES".to_string(), ClBufferParams {
            size: INDEX_SIZE,
            flags: MemFlags::empty(),
        });
        
        Self::check_device_compatibility(&context.device, &buffers)?;
        
        let program = Program::builder()
            .devices(&vec![context.device.clone()])
            .src(SRC)  // Use embedded source
            .build(&context.context)
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to compile OpenCL program: {}", e)))?;
        
        let buffer_a1 = Self::build_buffer(buffers.get("A1"), &context.queue)?;
        let buffer_a2 = Self::build_buffer(buffers.get("A2"), &context.queue)?;
        let buffer_b = Self::build_buffer(buffers.get("B"), &context.queue)?;
        let buffer_i1 = Self::build_buffer(buffers.get("I1"), &context.queue)?;
        let buffer_i2 = Self::build_buffer(buffers.get("I2"), &context.queue)?;
        let buffer_r = Self::build_buffer(buffers.get("R"), &context.queue)?;
        let buffer_nonces = Self::build_buffer(buffers.get("NONCES"), &context.queue)?;
        
        let total_mb = (BUFFER_SIZE_A1 + BUFFER_SIZE_A2 + BUFFER_SIZE_B + INDEX_SIZE * 3 + 42 * 2) as f64 / (1024.0 * 1024.0);
        tracing::info!("✅ Trimmer initialized with {:.1}MB total buffers", total_mb);
        
        Ok(Trimmer {
            q: context.queue,
            program,
            buffer_a1,
            buffer_a2,
            buffer_b,
            buffer_i1,
            buffer_i2,
            buffer_r,
            buffer_nonces,
            _device_capabilities: device_capabilities.clone(),
            stats: TrimmingStats::default(),
            is_nvidia,
            device_name: device_capabilities.name.clone(),
            device_id: device_capabilities.device_id,
        })
    }
    
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn setup_environment() {
        env::set_var("GPU_MAX_HEAP_SIZE", "100");
        env::set_var("GPU_USE_SYNC_OBJECTS", "1");
        env::set_var("GPU_MAX_ALLOC_PERCENT", "100");
        env::set_var("GPU_SINGLE_ALLOC_PERCENT", "100");
        env::set_var("GPU_64BIT_ATOMICS", "1");
        env::set_var("GPU_MAX_WORKGROUP_SIZE", "1024");
    }
    
    /// Executes the trimming process on the GPU (based on proven grin-miner implementation)
    pub unsafe fn run(&mut self, siphash_keys: &[u64; 4]) -> Result<Vec<u32>, Graxil29Error> {
        let start_time = Instant::now();
        tracing::info!("🚀 Starting trimming with keys: {:016x?}", siphash_keys);
        
        let [k0, k1, k2, k3] = *siphash_keys;
        
        // Create kernels with proven grin-miner work sizes
        let mut kernel_seed_a = kernel_builder!(self, "FluffySeed2A", 2048 * 128)
            .arg(k0).arg(k1).arg(k2).arg(k3)
            .arg(None::<&Buffer<Ulong4>>)
            .arg(None::<&Buffer<Ulong4>>)
            .arg(None::<&Buffer<u32>>)
            .build()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create seed_a kernel: {}", e)))?;
        if self.is_nvidia {
            kernel_seed_a.set_default_local_work_size(SpatialDims::One(128));
        }
        unsafe { kernel_seed_a.set_arg_unchecked(4, ArgVal::mem(&self.buffer_b))?; }
        unsafe { kernel_seed_a.set_arg_unchecked(5, ArgVal::mem(&self.buffer_a1))?; }
        unsafe { kernel_seed_a.set_arg_unchecked(6, ArgVal::mem(&self.buffer_i1))?; }

        let mut kernel_seed_b1 = kernel_builder!(self, "FluffySeed2B", 1024 * 128)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<Ulong4>>)
            .arg(None::<&Buffer<Ulong4>>)
            .arg(None::<&Buffer<i32>>)
            .arg(None::<&Buffer<i32>>)
            .arg(32)
            .build()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create seed_b1 kernel: {}", e)))?;
        if self.is_nvidia {
            kernel_seed_b1.set_default_local_work_size(SpatialDims::One(128));
        }
        unsafe { kernel_seed_b1.set_arg_unchecked(0, ArgVal::mem(&self.buffer_a1))?; }
        unsafe { kernel_seed_b1.set_arg_unchecked(1, ArgVal::mem(&self.buffer_a1))?; }
        unsafe { kernel_seed_b1.set_arg_unchecked(2, ArgVal::mem(&self.buffer_a2))?; }
        unsafe { kernel_seed_b1.set_arg_unchecked(3, ArgVal::mem(&self.buffer_i1))?; }
        unsafe { kernel_seed_b1.set_arg_unchecked(4, ArgVal::mem(&self.buffer_i2))?; }

        let mut kernel_seed_b2 = kernel_builder!(self, "FluffySeed2B", 1024 * 128)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<Ulong4>>)
            .arg(None::<&Buffer<Ulong4>>)
            .arg(None::<&Buffer<i32>>)
            .arg(None::<&Buffer<i32>>)
            .arg(0)
            .build()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create seed_b2 kernel: {}", e)))?;
        if self.is_nvidia {
            kernel_seed_b2.set_default_local_work_size(SpatialDims::One(128));
        }
        unsafe { kernel_seed_b2.set_arg_unchecked(0, ArgVal::mem(&self.buffer_b))?; }
        unsafe { kernel_seed_b2.set_arg_unchecked(1, ArgVal::mem(&self.buffer_a1))?; }
        unsafe { kernel_seed_b2.set_arg_unchecked(2, ArgVal::mem(&self.buffer_a2))?; }
        unsafe { kernel_seed_b2.set_arg_unchecked(3, ArgVal::mem(&self.buffer_i1))?; }
        unsafe { kernel_seed_b2.set_arg_unchecked(4, ArgVal::mem(&self.buffer_i2))?; }

        let mut kernel_round1 = kernel_builder!(self, "FluffyRound1", 4096 * 1024)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<i32>>)
            .arg(None::<&Buffer<i32>>)
            .arg((DUCK_SIZE_A * 1024) as i32)
            .arg((DUCK_SIZE_B * 1024) as i32)
            .build()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create round1 kernel: {}", e)))?;
        if self.is_nvidia {
            kernel_round1.set_default_local_work_size(SpatialDims::One(1024));
        }
        unsafe { kernel_round1.set_arg_unchecked(0, ArgVal::mem(&self.buffer_a1))?; }
        unsafe { kernel_round1.set_arg_unchecked(1, ArgVal::mem(&self.buffer_a2))?; }
        unsafe { kernel_round1.set_arg_unchecked(2, ArgVal::mem(&self.buffer_b))?; }
        unsafe { kernel_round1.set_arg_unchecked(3, ArgVal::mem(&self.buffer_i2))?; }
        unsafe { kernel_round1.set_arg_unchecked(4, ArgVal::mem(&self.buffer_i1))?; }

        let mut kernel_round0 = kernel_builder!(self, "FluffyRoundNO1", 4096 * 1024)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<i32>>)
            .arg(None::<&Buffer<i32>>)
            .build()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create round0 kernel: {}", e)))?;
        if self.is_nvidia {
            kernel_round0.set_default_local_work_size(SpatialDims::One(1024));
        }
        unsafe { kernel_round0.set_arg_unchecked(0, ArgVal::mem(&self.buffer_b))?; }
        unsafe { kernel_round0.set_arg_unchecked(1, ArgVal::mem(&self.buffer_a1))?; }
        unsafe { kernel_round0.set_arg_unchecked(2, ArgVal::mem(&self.buffer_i1))?; }
        unsafe { kernel_round0.set_arg_unchecked(3, ArgVal::mem(&self.buffer_i2))?; }

        let mut kernel_round_na = kernel_builder!(self, "FluffyRoundNON", 4096 * 1024)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<i32>>)
            .arg(None::<&Buffer<i32>>)
            .build()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create round_na kernel: {}", e)))?;
        if self.is_nvidia {
            kernel_round_na.set_default_local_work_size(SpatialDims::One(1024));
        }
        unsafe { kernel_round_na.set_arg_unchecked(0, ArgVal::mem(&self.buffer_b))?; }
        unsafe { kernel_round_na.set_arg_unchecked(1, ArgVal::mem(&self.buffer_a1))?; }
        unsafe { kernel_round_na.set_arg_unchecked(2, ArgVal::mem(&self.buffer_i1))?; }
        unsafe { kernel_round_na.set_arg_unchecked(3, ArgVal::mem(&self.buffer_i2))?; }

        let mut kernel_round_nb = kernel_builder!(self, "FluffyRoundNON", 4096 * 1024)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<i32>>)
            .arg(None::<&Buffer<i32>>)
            .build()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create round_nb kernel: {}", e)))?;
        if self.is_nvidia {
            kernel_round_nb.set_default_local_work_size(SpatialDims::One(1024));
        }
        unsafe { kernel_round_nb.set_arg_unchecked(0, ArgVal::mem(&self.buffer_a1))?; }
        unsafe { kernel_round_nb.set_arg_unchecked(1, ArgVal::mem(&self.buffer_b))?; }
        unsafe { kernel_round_nb.set_arg_unchecked(2, ArgVal::mem(&self.buffer_i2))?; }
        unsafe { kernel_round_nb.set_arg_unchecked(3, ArgVal::mem(&self.buffer_i1))?; }

        let mut kernel_tail = kernel_builder!(self, "FluffyTailO", 4096 * 1024)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<Uint2>>)
            .arg(None::<&Buffer<i32>>)
            .arg(None::<&Buffer<i32>>)
            .build()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create tail kernel: {}", e)))?;
        if self.is_nvidia {
            kernel_tail.set_default_local_work_size(SpatialDims::One(1024));
        }
        unsafe { kernel_tail.set_arg_unchecked(0, ArgVal::mem(&self.buffer_b))?; }
        unsafe { kernel_tail.set_arg_unchecked(1, ArgVal::mem(&self.buffer_a1))?; }
        unsafe { kernel_tail.set_arg_unchecked(2, ArgVal::mem(&self.buffer_i1))?; }
        unsafe { kernel_tail.set_arg_unchecked(3, ArgVal::mem(&self.buffer_i2))?; }

        let mut _event_list = EventList::new();
        let mut _names: Vec<&'static str> = vec![];

        // Execute trimming algorithm exactly like grin-miner
        tracing::debug!("🌱 Starting seed phase");
        clear_buffer!(self.buffer_i1);
        clear_buffer!(self.buffer_i2);
        kernel_enq!(kernel_seed_a, _event_list, _names, "seedA");
        kernel_enq!(kernel_seed_b1, _event_list, _names, "seedB1");
        kernel_enq!(kernel_seed_b2, _event_list, _names, "seedB2");
        
        tracing::debug!("🔄 Starting trimming rounds");
        clear_buffer!(self.buffer_i1);
        kernel_enq!(kernel_round1, _event_list, _names, "round1");
        clear_buffer!(self.buffer_i2);
        kernel_enq!(kernel_round0, _event_list, _names, "roundN0");
        clear_buffer!(self.buffer_i1);
        kernel_enq!(kernel_round_nb, _event_list, _names, "roundNB");
        
        // 120 trimming rounds
        for round in 0..120 {
            if round % 20 == 0 {
                tracing::debug!("Trimming progress: round {}/120", round);
            }
            clear_buffer!(self.buffer_i2);
            kernel_enq!(kernel_round_na, _event_list, _names, "roundNA");
            clear_buffer!(self.buffer_i1);
            kernel_enq!(kernel_round_nb, _event_list, _names, "roundNB");
        }
        
        tracing::debug!("🎯 Starting tail phase");
        clear_buffer!(self.buffer_i2);
        kernel_enq!(kernel_tail, _event_list, _names, "tail");

        // Read results with bounds checking
        let mut edges_count: Vec<u32> = vec![0; 1];
        self.buffer_i2.cmd().read(&mut edges_count).enq()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to read edge count: {}", e)))?;

        let edge_count = edges_count[0];
        tracing::debug!("Tail phase found {} edges", edge_count);
        
        // Sanity check
        if edge_count > 1_000_000 {
            return Err(Graxil29Error::Gpu(format!(
                "Suspicious edge count: {}. This indicates a trimming failure.", edge_count
            )));
        }
        
        if edge_count == 0 {
            tracing::info!("No edges remaining after trimming");
            self.q.finish()?;
            return Ok(Vec::new());
        }

        let mut edges_left: Vec<u32> = vec![0; (edge_count * 2) as usize];
        self.buffer_a1.cmd().read(&mut edges_left).enq()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to read edges: {}", e)))?;
        
        self.q.finish()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to finish queue: {}", e)))?;

        #[cfg(feature = "profile")]
        self.print_profiling_info(&_names, &_event_list);

        // Cleanup
        clear_buffer!(self.buffer_i1);
        clear_buffer!(self.buffer_i2);
        self.q.finish()?;

        let elapsed = start_time.elapsed().as_millis() as u64;
        tracing::info!("✅ Trimming completed: {} edges remaining in {}ms", 
            edge_count, elapsed);
        
        Ok(edges_left)
    }
    
    /// Recovers nonces from a given cycle (based on grin-miner implementation)
    pub unsafe fn recover(&mut self, mut cycle_nodes: Vec<u32>, siphash_keys: &[u64; 4]) -> Result<(Vec<u32>, bool), Graxil29Error> {
        tracing::info!("🔍 Recovering nonces for cycle: {} nodes", cycle_nodes.len());
        
        if cycle_nodes.len() != 42 {
            return Err(Graxil29Error::Gpu(format!(
                "Invalid cycle length: expected 42, got {}", cycle_nodes.len()
            )));
        }
        
        let [k0, k1, k2, k3] = *siphash_keys;
        
        let mut kernel_recovery = kernel_builder!(self, "FluffyRecovery", 2048 * 256)
            .arg(k0)
            .arg(k1) 
            .arg(k2)
            .arg(k3)
            .arg(None::<&Buffer<u64>>)
            .arg(None::<&Buffer<i32>>)
            .build()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to create recovery kernel: {}", e)))?;
        
        if self.is_nvidia {
            kernel_recovery.set_default_local_work_size(SpatialDims::One(256));
        }
        
        unsafe { kernel_recovery.set_arg_unchecked(4, ArgVal::mem(&self.buffer_r))?; }
        unsafe { kernel_recovery.set_arg_unchecked(5, ArgVal::mem(&self.buffer_nonces))?; }
        
        cycle_nodes.push(cycle_nodes[0]);
        let edges = cycle_nodes.windows(2).flatten().map(|v| *v).collect::<Vec<u32>>();
        self.buffer_r.cmd().write(edges.as_slice()).enq()?;
        self.buffer_nonces.cmd().fill(0, None).enq()?;
        
        unsafe { kernel_recovery.cmd().enq()?; }
        
        let mut nonces: Vec<u32> = vec![0; 42];
        self.buffer_nonces.cmd().read(&mut nonces).enq()?;
        self.q.finish()?;
        
        nonces.sort();
        let valid = nonces.windows(2).all(|entry| match entry {
            [p, n] => p < n && *p > 0,
            _ => true,
        });
        
        if valid {
            tracing::info!("✅ Successfully recovered {} valid nonces", 
                nonces.iter().filter(|&&n| n > 0).count());
        } else {
            tracing::warn!("⚠️  Nonce recovery validation failed");
        }
        
        Ok((nonces, valid))
    }
    
    #[cfg(feature = "profile")]
    fn print_profiling_info(&self, names: &[&'static str], events: &EventList) {
        use ocl::enums::ProfilingInfo;
        
        tracing::info!("📊 Kernel profiling:");
        for (i, name) in names.iter().enumerate() {
            if let Some(event) = events.get(i) {
                if let (Ok(start), Ok(end)) = (
                    event.profiling_info(ProfilingInfo::Start),
                    event.profiling_info(ProfilingInfo::End),
                ) {
                    let duration_ms = (end.time().unwrap() - start.time().unwrap()) / 1_000_000;
                    tracing::info!("  {}: {}ms", name, duration_ms);
                }
            }
        }
    }
    
    /// Gets trimming stats
    pub fn stats(&self) -> &TrimmingStats {
        &self.stats
    }
    
    fn check_device_compatibility(device: &Device, buffers: &HashMap<String, ClBufferParams>) -> Result<(), Graxil29Error> {
        let max_alloc_size: u64 = match device.info(DeviceInfo::MaxMemAllocSize) {
            Ok(DeviceInfoResult::MaxMemAllocSize(size)) => size,
            _ => return Err(Graxil29Error::Gpu("Failed to get max alloc size".to_string())),
        };
        
        let global_memory_size: u64 = match device.info(DeviceInfo::GlobalMemSize) {
            Ok(DeviceInfoResult::GlobalMemSize(size)) => size,
            _ => return Err(Graxil29Error::Gpu("Failed to get global memory size".to_string())),
        };
        
        let mut total_alloc: u64 = 0;
        
        for (name, params) in buffers {
            total_alloc += params.size as u64;
            if params.size as u64 > max_alloc_size {
                return Err(Graxil29Error::Gpu(format!(
                    "Buffer {} ({} bytes) exceeds max allocation size ({} bytes)",
                    name, params.size, max_alloc_size
                )));
            }
        }
        
        if total_alloc > global_memory_size {
            return Err(Graxil29Error::Gpu(format!(
                "Total buffer allocation ({} bytes) exceeds global memory ({} bytes)",
                total_alloc, global_memory_size
            )));
        }
        
        tracing::debug!("✅ Device compatibility check passed: {}MB total allocation", 
            total_alloc / (1024 * 1024));
        
        Ok(())
    }
    
    fn build_buffer(params: Option<&ClBufferParams>, queue: &Queue) -> Result<Buffer<u32>, Graxil29Error> {
        match params {
            None => Err(Graxil29Error::Gpu("Invalid buffer parameters".to_string())),
            Some(p) => {
                tracing::debug!("Building buffer: {} elements ({} bytes)", p.size, p.size * 4);
                
                Buffer::<u32>::builder()
                    .queue(queue.clone())
                    .len(p.size)  // Fixed: use size directly, not divided by 4
                    .flags(p.flags)
                    .fill_val(0)
                    .build()
                    .map_err(|e| Graxil29Error::Gpu(format!("Failed to create buffer: {}", e)))
            }
        }
    }
}

/// Creates SipHash keys from the header
pub fn create_siphash_keys(header: &[u8]) -> Result<[u64; 4], Graxil29Error> {
    let hash = blake2b(32, &[], header);
    let hash_bytes = hash.as_bytes();
    let mut cursor = Cursor::new(hash_bytes);
    
    Ok([
        cursor.read_u64::<LittleEndian>()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to read SipHash key: {}", e)))?,
        cursor.read_u64::<LittleEndian>()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to read SipHash key: {}", e)))?,
        cursor.read_u64::<LittleEndian>()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to read SipHash key: {}", e)))?,
        cursor.read_u64::<LittleEndian>()
            .map_err(|e| Graxil29Error::Gpu(format!("Failed to read SipHash key: {}", e)))?,
    ])
}

fn vendor_name(vendor: GpuVendor) -> &'static str {
    match vendor {
        GpuVendor::Nvidia => "NVIDIA",
        GpuVendor::Amd => "AMD",
        GpuVendor::Intel => "Intel",
        GpuVendor::Unknown => "Unknown",
    }
}

// Proven OpenCL source from grin-miner
const SRC: &str = include_str!("../kernels/opencl/cuckaroo29.cl");