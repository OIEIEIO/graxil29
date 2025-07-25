// src/main.rs - Multi-algorithm implementation for C29 and C32 mining
// Tree location: ./src/main.rs

//! Graxil29 Main Entry Point
//! 
//! Handles command-line arguments, initializes mining, and runs the main loop.
//! Integrates with stratum, GPU compute, and algorithm selection.
//! 
//! # Version History
//! - 0.1.0: Initial implementation with Cuckaroo29 support
//! - 0.2.0: Added Cuckatoo32, unified algorithm handling
//! - 0.3.0: Integrated new Algorithm methods, removed solution.algorithm
//! - 0.3.1: Fixed unused import, missing methods errors, improved logging
//! - 0.3.2: Added algorithm-aware stratum client integration
//! - 0.3.3: Added stratum_client.subscribe call before login for pool compatibility
//! - 0.3.4: Added stratum_client.start_listener call to handle mining.notify messages
//! - 0.3.5: Added job switching check and reconnection logic for faster job updates
//! - 0.3.6: Fixed job validation and reduced batch sizes to prevent stale job submissions
//! - 0.4.0: GMiner-style console output, removed dashboard, fixed difficulty handling
//! - 0.4.1: Added user-configurable nonce count via --nonce-count
//! - 0.4.2: Fixed JobTemplate, WorkTemplate, and moved From<FromHexError> to lib.rs
//! - 0.4.3: Rewrote mining loop with corrected Stratum integration to prevent stale shares
//! - 0.4.4: Fixed duplicate job notification logging
//! - 0.4.5: Updated submission logic to use block height, submitting valid shares
//! - 0.4.6: Added debug logging and increased default nonce count to boost share finding
//! - 0.4.7: Optimized submission to ensure shares are submitted based on height match
//! - 0.4.8: Rewritten mining loop to start new job on each pool update with lower nonce counts
//! - 0.4.9: Fixed type mismatch in job change handling and removed unused imports
//! - 0.5.0: Corrected macro syntax and cloning for job change handling
//! - 0.5.1: Fixed variable scoping issues in mining loop
//! - 0.5.2: Added missing WorkTemplate import
//! - 0.6.0: **MAJOR SIMPLIFICATION**: Removed complex job change notifications, simplified mining loop to only reset on clean_jobs=true

use clap::{Parser, Subcommand};
use graxil29::{
    init, 
    config::Settings, 
    stratum::{StratumClient, StratumSolution, client::WorkTemplate},
    gpu::compute::{GpuCompute, JobTemplate},
    algorithms::Algorithm,
    Graxil29Error, Result
};
use std::time::Instant;
use tokio::time::{sleep, Duration};
use chrono::Local;

#[derive(Parser)]
#[command(name = "graxil29")]
#[command(about = "Modern Rust Grin/Aeternity miner supporting Cuckaroo29 and Cuckatoo32")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Test mining algorithm
    Test {
        /// Algorithm to test (c29, c32, auto)
        #[arg(short, long, default_value = "auto")]
        algorithm: String,
        /// Number of nonces to test
        #[arg(short, long, default_value = "1000")]
        count: u64,
        /// Header hash (hex)
        #[arg(long, default_value = "0000000000000000000000000000000000000000000000000000000000000000")]
        header: String,
        /// Starting nonce
        #[arg(long, default_value = "0")]
        start_nonce: u64,
    },
    /// Run benchmark
    Benchmark {
        /// Algorithm to benchmark (c29, c32, auto)
        #[arg(short, long, default_value = "auto")]
        algorithm: String,
        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: u64,
        /// Number of nonces per iteration
        #[arg(long, default_value = "500000")]
        nonce_count: u64,
    },
    /// Start pool mining
    Mine {
        /// Algorithm to mine (c29, c32, auto)
        #[arg(short, long, default_value = "auto")]
        algorithm: String,
        /// Pool URL override
        #[arg(short, long)]
        pool: Option<String>,
        /// Wallet address override        
        #[arg(short = 'u', long)]
        wallet: Option<String>,
        /// Worker name override
        #[arg(short = 'w', long)]
        worker: Option<String>,
        /// Number of nonces per batch
        #[arg(long, default_value = "1000")] // Optimized for clean_jobs approach
        nonce_count: u64,
    },
    /// Show current configuration and GPU capabilities
    Config,
    /// List supported algorithms and GPU info
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    init()?;
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Test { algorithm, count, header, start_nonce } => {
            test_algorithm(&algorithm, &header, start_nonce, count).await?;
        }
        Commands::Benchmark { algorithm, iterations, nonce_count } => {
            benchmark_algorithm(&algorithm, iterations, nonce_count).await?;
        }
        Commands::Mine { algorithm, pool, wallet, worker, nonce_count } => {
            start_pool_mining(&algorithm, pool, wallet, worker, nonce_count).await?;
        }
        Commands::Config => {
            show_config().await?;
        }
        Commands::Info => {
            show_gpu_info().await?;
        }
    }
    
    Ok(())
}

/// Start pool mining with simplified mining loop - only resets on clean_jobs=true (new block height)
async fn start_pool_mining(
    algorithm_str: &str,
    pool_override: Option<String>, 
    wallet_override: Option<String>,
    worker_override: Option<String>,
    nonce_count: u64,
) -> Result<()> {
    let mut settings = Settings::default();
    
    // Apply overrides
    if let Some(pool) = pool_override {
        settings.pool_url = pool;
    }
    if let Some(wallet) = wallet_override {
        settings.wallet_address = wallet;
    }
    if let Some(worker) = worker_override {
        settings.worker_name = worker;
    }
    
    // Initialize GPU and detect/select algorithm
    let mut gpu = GpuCompute::new().await?;
    let algorithm = select_algorithm(&gpu, algorithm_str).await?;
    
    let timestamp = Local::now().format("%H:%M:%S").to_string();
    println!("{} Starting Graxil29 Miner", timestamp);
    println!("{} Algorithm: {}", timestamp, algorithm.name());
    println!("{} Pool: {}", timestamp, settings.pool_url.replace("stratum+tcp://", ""));
    println!("{} Wallet: {}", timestamp, settings.wallet_address);
    println!("{} Worker: {}", timestamp, settings.worker_name);
    println!("{} Nonce count per batch: {}", timestamp, nonce_count);
    
    // Validate algorithm is supported
    if !gpu.supported_algorithms().contains(&algorithm) {
        return Err(Graxil29Error::Config(format!(
            "{} not supported on this GPU ({}GB memory)",
            algorithm.name(),
            gpu.memory_info().available_memory / (1024 * 1024 * 1024)
        )));
    }
    
    // Algorithm-specific info
    match algorithm {
        Algorithm::Cuckaroo29 => {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} Mining Aeternity (AE) with Cuckaroo29", timestamp);
        }
        Algorithm::Cuckatoo32 => {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} Mining Grin (GRIN) with Cuckatoo32", timestamp);
        }
    }
    
    let mining_start_time = Instant::now();
    let mut status_timer = Instant::now();
    let mut total_hashes_session = 0u64;
    let mut solutions_found = 0u64;
    let mut solutions_accepted = 0u64;

    // Start Stratum client with simplified job handling
    let mut stratum_client = StratumClient::new(&settings.pool_url, algorithm).await?;
    stratum_client.subscribe().await?;
    stratum_client.login(&settings.wallet_address, &settings.worker_name).await?;
    stratum_client.extranonce_subscribe().await?;

    let timestamp = Local::now().format("%H:%M:%S").to_string();
    println!("{} Waiting for first job from pool...", timestamp);

    // Wait for initial work from pool
    let mut current_work: Option<WorkTemplate> = None;
    while current_work.is_none() {
        current_work = stratum_client.get_work().await?;
        if current_work.is_none() {
            sleep(Duration::from_millis(100)).await;
        }
    }

    let mut current_nonce = 0u64;
    let mut speed_start = Instant::now();
    let mut speed_hashes = 0u64;

    let timestamp = Local::now().format("%H:%M:%S").to_string();
    println!("{} Starting mining loop with simplified clean_jobs handling", timestamp);

    // Simplified mining loop - only resets on clean_jobs=true (new block height)
    loop {
        // Check if we need to reset mining due to clean_jobs=true (new block height)
        if stratum_client.should_reset_mining() {
            // Reset mining state for new block height
            current_nonce = 0;
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} Mining reset triggered by clean_jobs=true (new block height)", timestamp);
            
            // Get the new work template
            if let Some(work) = stratum_client.get_work().await? {
                current_work = Some(work.clone());
                
                // Set new job template in GPU
                gpu.set_job_template(JobTemplate {
                    job_id: work.job_id.clone(),
                    pre_pow: work.pre_pow.clone(),
                    height: work.height,
                    difficulty: work.difficulty,
                });
            }
        }

        // Mine with current work (solutions valid for entire block height)
        if let Some(work) = current_work.clone() {
            let solutions = match gpu.mine_algorithm(
                algorithm,
                work.header_hash,
                current_nonce,
                nonce_count,
                work.difficulty
            ).await {
                Ok(solutions) => solutions,
                Err(e) => {
                    let timestamp = Local::now().format("%H:%M:%S").to_string();
                    println!("{} Mining failed: {}", timestamp, e);
                    sleep(Duration::from_millis(1000)).await; // Brief pause on error
                    continue;
                }
            };
            
            current_nonce += nonce_count;
            total_hashes_session += nonce_count;
            speed_hashes += nonce_count;
            
            // Submit all solutions found (no stale work paranoia - valid for entire block height)
            for solution in solutions {
                let timestamp = Local::now().format("%H:%M:%S").to_string();
                solutions_found += 1;
                println!("{} GPU0: Share #{} verified, difficulty: {:.2}", 
                    timestamp, solutions_found, solution.difficulty);
                
                let stratum_solution = StratumSolution { 
                    nonce: solution.solution.nonce, 
                    job_id: work.job_id.clone(), 
                    worker_name: settings.worker_name.clone(),
                    cycle: solution.solution.cycle,
                    algorithm,
                    calculated_difficulty: solution.difficulty,
                }; 
                
                match stratum_client.submit_solution(&stratum_solution).await {
                    Ok(accepted) => {
                        let timestamp = Local::now().format("%H:%M:%S").to_string();
                        if accepted {
                            solutions_accepted += 1;
                            println!("{} GPU0: Share #{} accepted", timestamp, solutions_found);
                        } else {
                            println!("{} GPU0: Share #{} rejected by pool", timestamp, solutions_found);
                        }
                    }
                    Err(e) => {
                        let timestamp = Local::now().format("%H:%M:%S").to_string();
                        println!("{} Failed to submit solution: {}", timestamp, e);
                    }
                }
            }
        } else {
            // No work available - wait briefly
            sleep(Duration::from_millis(100)).await;
            continue;
        }

        // Show status every 30 seconds like GMiner
        if status_timer.elapsed() >= Duration::from_secs(30) {
            let speed_elapsed = speed_start.elapsed().as_secs_f64();
            let current_gps = if speed_elapsed > 0.0 {
                speed_hashes as f64 / speed_elapsed
            } else {
                0.0
            };
            
            let uptime = mining_start_time.elapsed();
            let uptime_str = format!("{}d {:02}:{:02}:{:02}", 
                uptime.as_secs() / 86400,
                (uptime.as_secs() % 86400) / 3600, 
                (uptime.as_secs() % 3600) / 60, 
                uptime.as_secs() % 60
            );
            
            let pool_url_clean = settings.pool_url.replace("stratum+tcp://", "");
            let pool_name = pool_url_clean.split(':').next().unwrap_or("unknown");
            
            let acceptance_rate = if solutions_found > 0 {
                (solutions_accepted as f64 / solutions_found as f64) * 100.0
            } else {
                100.0
            };
            
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} Pool: {} Diff: {:.2}", timestamp, pool_name, 
                current_work.as_ref().map_or(16.0, |w| w.difficulty));
            println!("{} Shares/Minute: {:.2} Accept: {:.1}% ({}/{})", timestamp, 
                if uptime.as_secs() > 0 { solutions_found as f64 * 60.0 / uptime.as_secs() as f64 } else { 0.0 },
                acceptance_rate, solutions_accepted, solutions_found);
            println!("{} Uptime: {} Speed: {:.2} Gps Total: {}", timestamp, uptime_str, current_gps, total_hashes_session);
            
            // Reset speed calculation
            status_timer = Instant::now();
            speed_start = Instant::now();
            speed_hashes = 0;
        }
        
        // Brief pause to allow for message processing
        sleep(Duration::from_millis(50)).await;
    }
}

async fn show_config() -> Result<()> {
    let settings = Settings::default();
    let gpu = GpuCompute::new().await?;
    let supported = gpu.supported_algorithms();
    let recommended = Algorithm::detect_best(
        (gpu.memory_info().available_memory / (1024 * 1024 * 1024)) as usize
    );
    
    println!("Graxil29 Configuration:");
    println!("Pool URL: {}", settings.pool_url);
    println!("Wallet Address: {}", settings.wallet_address);
    println!("Worker Name: {}", settings.worker_name);
    println!("Difficulty: {}", settings.difficulty);
    println!("GPU Threads: {}", settings.gpu_threads);
    println!("Use GPU: {}", settings.use_gpu);
    
    println!("\nGPU Information:");
    println!("GPU: {}", gpu.gpu_info().name);
    println!("Memory: {:.1}GB available", 
             gpu.memory_info().available_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("Supported algorithms: {:?}", supported.iter().map(|a| a.name()).collect::<Vec<_>>());
    println!("Recommended algorithm: {} {}", 
             recommended.name(),
             if recommended.is_mainnet_active() { "(ACTIVE)" } else { "(DEPRECATED)" }
    );
    
    println!("\nPool Information:");
    println!("• 2Miners AE: stratum+tcp://ae.2miners.com:4040 (C29 Aeternity)");
    println!("• 2Miners GRIN: stratum+tcp://grin.2miners.com:3030 (C32 Grin)");
    println!("• Grinmint C32: stratum+tcp://eu-west-stratum.grinmint.com:4416");
    println!("• F2Pool C32: stratum+tcp://grin.f2pool.com:13654");
    
    if supported.contains(&Algorithm::Cuckaroo29) {
        println!("\nCuckaroo29 Support:");
        println!("C29 is used by Aeternity blockchain");
        println!("Use ae.2miners.com:4040 for Aeternity mining");
        println!("Note: C29 and C32 use different pools and blockchains");
    }
    
    println!("\nAlgorithm-Specific Commands:");
    println!("• Test C29: cargo run -- test --algorithm c29 --count 1000");
    println!("• Test C32: cargo run -- test --algorithm c32 --count 100");
    println!("• Mine AE: cargo run -- mine --algorithm c29 --pool stratum+tcp://ae.2miners.com:4040 --nonce-count 1000");
    println!("• Mine GRIN: cargo run -- mine --algorithm c32 --pool stratum+tcp://grin.2miners.com:3030 --nonce-count 1000");
    
    Ok(())
}

async fn show_gpu_info() -> Result<()> {
    let gpu = GpuCompute::new().await?;
    let info = gpu.gpu_info();
    let memory = gpu.memory_info();
    
    println!("GPU Information:");
    println!("Name: {}", info.name);
    println!("Vendor: {}", info.vendor);
    println!("Backend: {:?}", info.backend);
    println!("Device Type: {:?}", info.device_type);
    
    println!("\nMemory Information:");
    println!("Total Memory: {:.1}GB", memory.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("Available Memory: {:.1}GB", memory.available_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("Memory Limit: {:.1}GB", memory.memory_limit as f64 / (1024.0 * 1024.0 * 1024.0));
    
    println!("\nAlgorithm Support:");
    for algorithm in [Algorithm::Cuckaroo29, Algorithm::Cuckatoo32] {
        let supported = gpu.supported_algorithms().contains(&algorithm);
        let blockchain = match algorithm {
            Algorithm::Cuckaroo29 => "Aeternity (AE)",
            Algorithm::Cuckatoo32 => "Grin (GRIN)",
        };
        let status = if supported {
            "SUPPORTED"
        } else {
            "NOT SUPPORTED"
        };
        
        println!("{}: {} - {}", algorithm.name(), status, blockchain);
        println!("  Required: {}GB, Edge bits: {}, Graph size: {}M nodes", 
                 algorithm.min_memory_gb(),
                 algorithm.edge_bits(),
                 algorithm.graph_size() / 1_000_000);
    }
    
    println!("\nPerformance Estimates:");
    let memory_gb = (gpu.memory_info().available_memory / (1024 * 1024 * 1024)) as usize;
    
    if memory_gb >= 6 {
        println!("Cuckaroo29 (AE): ~1-10K Gps (10K nonce batches)");
    }
    if memory_gb >= 11 {
        println!("Cuckatoo32 (GRIN): ~100-1K Gps (1K nonce batches)");
    }
    
    Ok(())
}

async fn test_algorithm(algorithm_str: &str, header_hex: &str, start_nonce: u64, count: u64) -> Result<()> {
    let mut gpu = GpuCompute::new().await?;
    let algorithm = select_algorithm(&gpu, algorithm_str).await?;
    
    let timestamp = Local::now().format("%H:%M:%S").to_string();
    println!("{} Testing {} algorithm on GPU", timestamp, algorithm.name());
    println!("{} Edge bits: {}, Min memory: {}GB", timestamp, algorithm.edge_bits(), algorithm.min_memory_gb());
    
    let header_hash = parse_hex_hash(header_hex)?;
    
    let start_time = Instant::now();
    let solutions = gpu.mine_algorithm(algorithm, header_hash, start_nonce, count, 16.0).await?;
    let elapsed = start_time.elapsed();
    
    let timestamp = Local::now().format("%H:%M:%S").to_string();
    println!("{} GPU test completed:", timestamp);
    println!("{} Algorithm: {}", timestamp, algorithm.name());
    println!("{} Nonces tested: {}", timestamp, count);
    println!("{} Solutions found: {}", timestamp, solutions.len());
    println!("{} Time elapsed: {:.2}s", timestamp, elapsed.as_secs_f64());
    println!("{} Rate: {:.2} Gps", timestamp, count as f64 / elapsed.as_secs_f64());
    
    // Show solution details if any found
    for (i, solution) in solutions.iter().enumerate() {
        let timestamp = Local::now().format("%H:%M:%S").to_string();
        println!("{} Solution {}: nonce={}, difficulty: {:.2}", 
                 timestamp, i + 1, solution.solution.nonce, solution.difficulty);
        
        if tracing::enabled!(tracing::Level::DEBUG) {
            println!("{} Solution {} cycle: {:?}", timestamp, i + 1, &solution.solution.cycle[..6]);
        }
    }
    
    if solutions.is_empty() {
        let timestamp = Local::now().format("%H:%M:%S").to_string();
        println!("{} No solutions found - this is normal for random testing", timestamp);
        println!("{} Try increasing --count or testing with known-good headers", timestamp);
    }
    
    Ok(())
}

async fn benchmark_algorithm(algorithm_str: &str, iterations: u64, nonce_count: u64) -> Result<()> {
    let mut gpu = GpuCompute::new().await?;
    let algorithm = select_algorithm(&gpu, algorithm_str).await?;
    
    let timestamp = Local::now().format("%H:%M:%S").to_string();
    println!("{} Benchmarking {} algorithm with {} iterations, {} nonces per iteration", 
             timestamp, algorithm.name(), iterations, nonce_count);
    println!("{} Edge bits: {}, Graph size: {}M nodes", timestamp,
             algorithm.edge_bits(), algorithm.graph_size() / 1_000_000);
    
    let header_hash = hex::decode("a8db1910d85662f0167138c160c866683410c11f1ccfecb8ed8145716feb73e1")?
        .try_into()
        .map_err(|_| Graxil29Error::Gpu("Invalid header length".to_string()))?;
    
    let start_time = Instant::now();
    let mut total_solutions = 0;
    
    for i in 0..iterations {
        let solutions = gpu.mine_algorithm(
            algorithm, 
            header_hash, 
            i * nonce_count, 
            nonce_count, 
            16.0
        ).await?;
        total_solutions += solutions.len();
        
        if (i + 1) % 10 == 0 || i == 0 {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} Completed iteration {}/{}", timestamp, i + 1, iterations);
        }
    }
    
    let elapsed = start_time.elapsed();
    let total_nonces = iterations * nonce_count;
    
    let timestamp = Local::now().format("%H:%M:%S").to_string();
    println!("{} Benchmark completed:", timestamp);
    println!("{} Algorithm: {}", timestamp, algorithm.name());
    println!("{} Total nonces: {}", timestamp, total_nonces);
    println!("{} Total solutions: {}", timestamp, total_solutions);
    println!("{} Time elapsed: {:.2}s", timestamp, elapsed.as_secs_f64());
    println!("{} Average rate: {:.2} Gps", timestamp, total_nonces as f64 / elapsed.as_secs_f64());
    
    // Calculate solution rate
    if total_solutions > 0 {
        let solution_rate = total_solutions as f64 / total_nonces as f64;
        println!("{} Solution rate: {:.6} solutions/nonce", timestamp, solution_rate);
    }
    
    // Performance analysis
    let expected_c29_rate = 5000.0;   // Realistic expectation for C29
    let expected_c32_rate = 500.0;    // Realistic expectation for C32
    
    let actual_rate = total_nonces as f64 / elapsed.as_secs_f64();
    let expected_rate = match algorithm {
        Algorithm::Cuckaroo29 => expected_c29_rate,
        Algorithm::Cuckatoo32 => expected_c32_rate,
    };
    
    let performance_ratio = actual_rate / expected_rate * 100.0;
    println!("{} Performance: {:.1}% of expected {} rate", timestamp, performance_ratio, algorithm.name());
    
    Ok(())
}

async fn select_algorithm(gpu: &GpuCompute, algorithm_str: &str) -> Result<Algorithm> {
    let algorithm = match algorithm_str.to_lowercase().as_str() {
        "auto" => {
            let memory_gb = (gpu.memory_info().available_memory / (1024 * 1024 * 1024)) as usize;
            let detected = Algorithm::detect_best(memory_gb);
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} Auto-detected algorithm: {} ({}GB GPU memory)", 
                     timestamp, detected.name(), memory_gb);
            detected
        }
        "c29" | "cuckaroo29" | "29" => {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} User selected: Cuckaroo29", timestamp);
            println!("{} C29 for Aeternity (AE) blockchain", timestamp);
            Algorithm::Cuckaroo29
        }
        "c32" | "cuckatoo32" | "32" => {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} User selected: Cuckatoo32", timestamp);
            println!("{} C32 for Grin (GRIN) blockchain", timestamp);
            Algorithm::Cuckatoo32
        }
        _ => {
            return Err(Graxil29Error::Config(format!(
                "Invalid algorithm: '{}'. Use 'auto', 'c29', or 'c32'", algorithm_str
            )));
        }
    };
    
    Ok(algorithm)
}

fn parse_hex_hash(hex: &str) -> Result<[u8; 32]> {
    let hex = hex.trim_start_matches("0x");
    if hex.len() != 64 {
        return Err(Graxil29Error::Config("Invalid header hash length".to_string()));
    }
    
    let mut hash = [0u8; 32];
    for i in 0..32 {
        hash[i] = u8::from_str_radix(&hex[i*2..i*2+2], 16)
            .map_err(|_| Graxil29Error::Config("Invalid hex in header hash".to_string()))?;
    }
    
    Ok(hash)
}