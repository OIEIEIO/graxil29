// src/stratum/client.rs - Professional mining client with clean_jobs handling and simplified mining loop integration
// Tree location: ./src/stratum/client.rs

//! Stratum Client Implementation
//! 
//! Professional mining client with request/response handling, pool communication,
//! and simplified clean_jobs-based mining loop integration.
//! 
//! # Version History
//! - 0.1.0: Initial Stratum client implementation with basic pool communication
//! - 0.2.0: Added professional request/response handling with async channels
//! - 0.3.0: Integrated job change notification system for real-time mining updates
//! - 0.3.1: Added extranonce subscription support for pool compatibility
//! - 0.3.2: Enhanced error handling and connection management
//! - 0.3.3: Added GMiner-style console output and job display formatting
//! - 0.3.4: Implemented complex job change notification with watch channels
//! - 0.3.5: Added atomic clean_job_flag for emergency mining stops
//! - 0.3.6: Enhanced job validation and stale work detection
//! - 0.3.7: Added job change timestamps and notification system
//! - 0.3.8: Improved difficulty handling and pool message routing
//! - 0.3.9: Added professional message listener with response channel routing
//! - 0.4.0: Enhanced solution submission with job ID validation
//! - 0.4.1: Added worker identification and algorithm-specific handling
//! - 0.4.2: Improved connection error handling and reconnection logic
//! - 0.4.3: Added comprehensive job template management
//! - 0.4.4: Enhanced pool difficulty tracking and updates
//! - 0.4.5: Added solution difficulty calculation and display
//! - 0.4.6: Improved job notification logging and status display
//! - 0.4.7: Added job change receiver system for mining loop integration
//! - 0.4.8: Enhanced clean_jobs flag handling for block height changes
//! - 0.4.9: Added atomic job abandonment flags and reset mechanisms
//! - 0.5.0: **MAJOR SIMPLIFICATION**: Removed complex job change notifications, watch channels, and JobChangeNotification structs - simplified to clean_jobs flag polling only
//! - 0.5.1: **CRITICAL FIX**: Removed stale job validation completely - solutions valid for entire block height, not specific job IDs (confirmed via ngrep pool analysis)
//! - 0.5.2: **PROTOCOL FIX**: Removed "jsonrpc":"2.0" from all messages - pool uses classic Stratum protocol, not JSON-RPC 2.0 (GMiner comparison analysis)

use tokio::net::TcpStream;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::sync::{Mutex, oneshot};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use chrono::Local;
use crate::{Result, Graxil29Error, algorithms::Algorithm};

// Re-export Solution type for convenience
pub use crate::algorithms::Solution;

/// Work template received from mining pool
#[derive(Debug, Clone)]
pub struct WorkTemplate {
    /// 32-byte header hash for mining
    pub header_hash: [u8; 32],
    /// Starting nonce for mining range
    pub nonce_start: u64,
    /// Pool difficulty target (e.g., 16.0)
    pub difficulty: f64,
    /// Job identifier from pool
    pub job_id: String,
    /// Block height
    pub height: u64,
    /// Network difficulty
    pub network_difficulty: f64,
    /// Full pre_pow block header hex (for C32)
    pub pre_pow: String,
    /// Whether this job requires clean restart (block found)
    pub clean_jobs: bool,
}

/// Solution to submit to mining pool
#[derive(Debug)]
pub struct StratumSolution {
    /// Nonce that found the solution
    pub nonce: u64,
    /// Job ID this solution belongs to
    pub job_id: String,
    /// Worker name for identification
    pub worker_name: String,
    /// 42-element cycle proof
    pub cycle: [u32; 42],
    /// Algorithm used to find this solution
    pub algorithm: Algorithm,
    /// Calculated difficulty of this solution
    pub calculated_difficulty: f64,
}

/// Professional mining client with simplified clean_jobs handling
pub struct StratumClient {
    // Message sending
    writer: Arc<Mutex<BufWriter<OwnedWriteHalf>>>,
    message_id: AtomicU64,
    
    // Response routing via channels - professional approach
    response_channels: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    
    // Mining state - immediately updated by listener
    current_work: Arc<Mutex<Option<WorkTemplate>>>,
    pool_difficulty: Arc<Mutex<f64>>,
    
    // Simple clean job flag for mining reset
    clean_job_flag: Arc<AtomicBool>,
    
    // Connection state
    wallet_address: String,
    worker_name: String,
    algorithm: Algorithm,
    subscription_id: Option<String>,
    extranonce: Option<String>,
    extranonce2_size: Option<u64>,
    
    // Listener lifecycle management
    listener_handle: Option<tokio::task::JoinHandle<()>>,
}

impl StratumClient {
    /// Create a new professional mining client with simplified mining loop integration
    pub async fn new(pool_url: &str, algorithm: Algorithm) -> Result<Self> {
        let url = pool_url.replace("stratum+tcp://", "").replace("tcp://", "");
        let timestamp = Local::now().format("%H:%M:%S").to_string();
        println!("{} Connecting to {}", timestamp, url);

        match TcpStream::connect(&url).await {
            Ok(stream) => {
                let timestamp = Local::now().format("%H:%M:%S").to_string();
                println!("{} Connected to {}", timestamp, url);
                let (reader_half, writer_half) = stream.into_split();
                let writer = Arc::new(Mutex::new(BufWriter::new(writer_half)));

                let mut client = Self {
                    writer,
                    message_id: AtomicU64::new(1),
                    response_channels: Arc::new(Mutex::new(HashMap::new())),
                    current_work: Arc::new(Mutex::new(None)),
                    pool_difficulty: Arc::new(Mutex::new(1.0)),
                    clean_job_flag: Arc::new(AtomicBool::new(false)),
                    wallet_address: String::new(),
                    worker_name: String::new(),
                    algorithm,
                    subscription_id: None,
                    extranonce: None,
                    extranonce2_size: None,
                    listener_handle: None,
                };

                // Start the professional message listener
                client.start_message_listener(reader_half).await;
                
                Ok(client)
            }
            Err(e) => {
                let timestamp = Local::now().format("%H:%M:%S").to_string();
                println!("{} Failed to connect to pool: {}", timestamp, e);
                Err(Graxil29Error::Config(format!("Failed to connect to pool: {}", e)))
            }
        }
    }

    /// Start the professional message listener - handles ALL incoming messages with simplified job handling
    async fn start_message_listener(&mut self, reader_half: OwnedReadHalf) {
        let reader = Arc::new(Mutex::new(BufReader::new(reader_half)));
        let response_channels = Arc::clone(&self.response_channels);
        let current_work = Arc::clone(&self.current_work);
        let pool_difficulty = Arc::clone(&self.pool_difficulty);
        let clean_job_flag = Arc::clone(&self.clean_job_flag);

        let handle = tokio::spawn(async move {
            let mut line = String::new();

            loop {
                line.clear();
                let message = {
                    let mut reader = reader.lock().await;
                    match reader.read_line(&mut line).await {
                        Ok(0) => {
                            tracing::error!("Connection closed by pool");
                            break;
                        }
                        Ok(_) => {
                            if line.trim().is_empty() {
                                continue;
                            }
                            match serde_json::from_str::<Value>(&line.trim()) {
                                Ok(msg) => msg,
                                Err(e) => {
                                    tracing::error!("Failed to parse message: {}", e);
                                    continue;
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Error reading from pool: {}", e);
                            break;
                        }
                    }
                };

                // Professional message routing
                if let Some(id) = message.get("id").and_then(|v| v.as_u64()) {
                    // This is a response to our request - route it via channel
                    let mut channels = response_channels.lock().await;
                    if let Some(sender) = channels.remove(&id) {
                        if sender.send(message).is_err() {
                            tracing::debug!("Response channel closed for ID {}", id);
                        }
                    } else {
                        tracing::debug!("No waiting channel for response ID {}", id);
                    }
                } else if let Some(method) = message.get("method").and_then(|v| v.as_str()) {
                    // This is an unsolicited message from pool - handle immediately
                    match method {
                        "mining.notify" => {
                            Self::handle_mining_notify_static(
                                &message, 
                                &current_work, 
                                &pool_difficulty, 
                                &clean_job_flag
                            ).await;
                        }
                        "mining.set_difficulty" => {
                            Self::handle_set_difficulty_static(&message, &pool_difficulty).await;
                        }
                        _ => {
                            tracing::debug!("Ignoring unsolicited method: {}", method);
                        }
                    }
                } else {
                    tracing::debug!("Ignoring message without ID or method: {:?}", message);
                }
            }
        });

        self.listener_handle = Some(handle);
    }

    /// Professional request/response handling - wait for specific response via channel
    async fn send_request_and_wait(&mut self, request: Value) -> Result<Value> {
        let request_id = request.get("id").and_then(|v| v.as_u64())
            .ok_or_else(|| Graxil29Error::Config("Request missing ID".to_string()))?;

        // Create channel for response
        let (sender, receiver) = oneshot::channel();
        {
            let mut channels = self.response_channels.lock().await;
            channels.insert(request_id, sender);
        }

        // Send request
        self.send_message(request).await?;

        // Wait for response via channel
        match receiver.await {
            Ok(response) => Ok(response),
            Err(_) => Err(Graxil29Error::Config("Response channel closed".to_string()))
        }
    }

    /// Subscribe to the mining pool using proper Stratum protocol (no JSON-RPC 2.0)
    pub async fn subscribe(&mut self) -> Result<()> {
        let _algorithm_name = match self.algorithm {
            Algorithm::Cuckaroo29 => "cuckaroo29", 
            Algorithm::Cuckatoo32 => "cuckatoo32",
        };

        let timestamp = Local::now().format("%H:%M:%S").to_string();
        println!("{} Subscribing to Stratum Server", timestamp);

        let request_id = self.next_id();
        let subscribe_msg = json!({
            "method": "mining.subscribe",
            "id": request_id,
            "params": [
                "Graxil29/0.1.0",
                null,
                "ae.2miners.com",
                "4040"
            ]
        });

        let subscribe_response = self.send_request_and_wait(subscribe_msg).await?;

        if let Some(result) = subscribe_response.get("result") {
            if let Some(array) = result.as_array() {
                if array.len() >= 3 {
                    let subscription_details = array[0]
                        .as_array()
                        .ok_or_else(|| Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))?;
                    let subscription_id = subscription_details[0]
                        .as_array()
                        .and_then(|v| v.get(1))
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))?
                        .to_string();
                    let extranonce = array[1]
                        .as_str()
                        .ok_or_else(|| Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))?
                        .to_string();
                    let extranonce2_size = array[2]
                        .as_u64()
                        .ok_or_else(|| Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))?;

                    self.subscription_id = Some(subscription_id.clone());
                    self.extranonce = Some(extranonce.clone());
                    self.extranonce2_size = Some(extranonce2_size);
                    
                    let timestamp = Local::now().format("%H:%M:%S").to_string();
                    println!("{} Subscribed to Stratum Server", timestamp);
                    println!("{} Set Extra Nonce: {}", timestamp, extranonce);
                    Ok(())
                } else {
                    Err(Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))
                }
            } else {
                Err(Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))
            }
        } else if let Some(error) = subscribe_response.get("error") {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} Subscription error: {:?}", timestamp, error);
            Err(Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))
        } else {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} Invalid subscription response: {:?}", timestamp, subscribe_response);
            Err(Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))
        }
    }

    /// Subscribe to extranonce updates using proper Stratum protocol
    pub async fn extranonce_subscribe(&mut self) -> Result<()> {
        let timestamp = Local::now().format("%H:%M:%S").to_string();
        println!("{} Subscribing to extranonce updates", timestamp);

        let request_id = self.next_id();
        let msg = json!({
            "method": "mining.extranonce.subscribe",
            "id": request_id,
            "params": []
        });

        let response = self.send_request_and_wait(msg).await?;

        if let Some(result) = response.get("result") {
            if result.as_bool() == Some(true) {
                let timestamp = Local::now().format("%H:%M:%S").to_string();
                println!("{} Extranonce subscription successful", timestamp);
                Ok(())
            } else {
                let timestamp = Local::now().format("%H:%M:%S").to_string();
                println!("{} Extranonce subscription failed: {:?}", timestamp, result);
                Err(Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))
            }
        } else if let Some(error) = response.get("error") {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} Extranonce subscription error: {:?}", timestamp, error);
            Err(Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))
        } else {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} Invalid extranonce subscription response: {:?}", timestamp, response);
            Err(Graxil29Error::Stratum(crate::stratum::types::StratumError::SubscriptionFailed))
        }
    }

    /// Login to the mining pool using proper Stratum protocol
    pub async fn login(&mut self, wallet_address: &str, worker_name: &str) -> Result<()> {
        self.wallet_address = wallet_address.to_string();
        self.worker_name = worker_name.to_string();

        let timestamp = Local::now().format("%H:%M:%S").to_string();
        println!("{} Authorizing on Stratum Server", timestamp);

        let request_id = self.next_id();
        let login_msg = json!({
            "method": "mining.authorize",
            "id": request_id,
            "params": [
                wallet_address,
                worker_name
            ]
        });

        let login_response = self.send_request_and_wait(login_msg).await?;

        if let Some(result) = login_response.get("result") {
            if result.as_bool() == Some(true) {
                let timestamp = Local::now().format("%H:%M:%S").to_string();
                println!("{} Authorized on Stratum Server", timestamp);
                Ok(())
            } else {
                Err(Graxil29Error::Config(format!(
                    "Pool login failed for {}: {:?}", self.algorithm.name(), result
                )))
            }
        } else if let Some(error) = login_response.get("error") {
            Err(Graxil29Error::Config(format!(
                "Pool login error for {}: {:?}", self.algorithm.name(), error
            )))
        } else {
            Err(Graxil29Error::Config(format!(
                "Invalid login response for {}: {:?}", self.algorithm.name(), login_response
            )))
        }
    }

    /// Get new work from the pool (always returns latest job)
    pub async fn get_work(&mut self) -> Result<Option<WorkTemplate>> {
        let current_work = self.current_work.lock().await;
        Ok(current_work.clone())
    }

    /// Check if mining should be reset due to clean_jobs=true (new block height)
    /// This is the simplified replacement for the complex job change notification system
    pub fn should_reset_mining(&self) -> bool {
        self.clean_job_flag.swap(false, Ordering::Relaxed)
    }

    /// Submit a solution using proper Stratum protocol (no JSON-RPC 2.0)
    /// 
    /// No stale job validation - solutions valid for entire block height.
    /// Uses classic Stratum format matching GMiner's successful submissions.
    pub async fn submit_solution(&mut self, solution: &StratumSolution) -> Result<bool> {
        if self.subscription_id.is_none() {
            return Err(Graxil29Error::Config("No subscription ID available".to_string()));
        }

        // Display share like GMiner
        let timestamp = Local::now().format("%H:%M:%S").to_string();
        println!("{} GPU0: Share #1 verified, difficulty: {:.2}", 
            timestamp, solution.calculated_difficulty);

        let extranonce2 = format!("{:08x}", solution.nonce % 0x100000000);
        let job_id = &solution.job_id;
        let request_id = self.next_id();

        // Use proper Stratum protocol format (no jsonrpc field) matching GMiner
        let submit_msg = json!({
            "method": "mining.submit",
            "id": request_id,
            "params": [
                self.wallet_address,
                job_id,
                extranonce2,
                solution.cycle.iter().map(|&x| format!("{:08x}", x)).collect::<Vec<String>>()
            ]
        });

        let response = self.send_request_and_wait(submit_msg).await?;

        if let Some(result) = response.get("result") {
            if result.as_bool() == Some(true) {
                let timestamp = Local::now().format("%H:%M:%S").to_string();
                println!("{} GPU0: Share #1 accepted", timestamp);
                Ok(true)
            } else {
                let timestamp = Local::now().format("%H:%M:%S").to_string();
                println!("{} GPU0: Share #1 rejected: {:?}", timestamp, result);
                Ok(false)
            }
        } else if let Some(error) = response.get("error") {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} GPU0: Share #1 error: {:?}", timestamp, error);
            Ok(false)
        } else {
            let timestamp = Local::now().format("%H:%M:%S").to_string();
            println!("{} GPU0: Invalid response: {:?}", timestamp, response);
            Ok(false)
        }
    }

    /// Get current algorithm
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Send a JSON message to the pool
    async fn send_message(&mut self, message: Value) -> Result<()> {
        let mut writer = self.writer.lock().await;
        let message_str = format!("{}\n", message.to_string());
        writer.write_all(message_str.as_bytes()).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Handle mining.notify messages - simplified job switching with clean_jobs support
    async fn handle_mining_notify_static(
        message: &Value, 
        current_work: &Arc<Mutex<Option<WorkTemplate>>>, 
        pool_difficulty: &Arc<Mutex<f64>>,
        clean_job_flag: &Arc<AtomicBool>
    ) {
        if let Some(params) = message.get("params").and_then(|v| v.as_array()) {
            if params.len() >= 5 {
                let job_id = params[0].as_str().unwrap_or("0").to_string();
                let pre_pow = params[1].as_str().unwrap_or("").to_string();
                let height = params[2].as_u64().unwrap_or(0);
                let _difficulty_hex = params[3].as_str().unwrap_or("0");
                let clean_jobs = params[4].as_bool().unwrap_or(false);

                let pre_pow_bytes = hex::decode(&pre_pow).unwrap_or_default();
                let mut header_hash = [0u8; 32];
                if pre_pow_bytes.len() >= 32 {
                    header_hash.copy_from_slice(&pre_pow_bytes[..32]);
                }

                // Get current pool difficulty
                let current_difficulty = *pool_difficulty.lock().await;

                let work = WorkTemplate {
                    header_hash,
                    nonce_start: 0,
                    difficulty: current_difficulty,
                    job_id: job_id.clone(),
                    height,
                    network_difficulty: current_difficulty,
                    pre_pow,
                    clean_jobs,
                };

                // IMMEDIATE job switching - critical for real-time mining
                {
                    let mut current_work = current_work.lock().await;
                    *current_work = Some(work);
                }

                // Set clean job flag for simplified mining loop reset
                if clean_jobs {
                    clean_job_flag.store(true, Ordering::Relaxed);
                }
                
                // Display job notification like GMiner - show clean_jobs status and block height
                let timestamp = Local::now().format("%H:%M:%S").to_string();
                if clean_jobs {
                    println!("{} New Job: {} Height: {} Diff: {:.2} [New Block Height]", 
                        timestamp, job_id, height, current_difficulty);
                } else {
                    println!("{} New Job: {} Height: {} Diff: {:.2}", 
                        timestamp, job_id, height, current_difficulty);
                }
            }
        }
    }

    /// Handle mining.set_difficulty messages - IMMEDIATE difficulty updates
    async fn handle_set_difficulty_static(message: &Value, pool_difficulty: &Arc<Mutex<f64>>) {
        if let Some(params) = message.get("params").and_then(|v| v.as_array()) {
            if let Some(difficulty) = params.get(0).and_then(|v| v.as_f64()) {
                let mut pool_diff = pool_difficulty.lock().await;
                *pool_diff = difficulty;
                tracing::debug!("Pool difficulty updated: {:.2}", difficulty);
            }
        }
    }

    /// Get next message ID
    fn next_id(&self) -> u64 {
        self.message_id.fetch_add(1, Ordering::SeqCst)
    }
}

impl Drop for StratumClient {
    /// Clean shutdown of listener
    fn drop(&mut self) {
        if let Some(handle) = self.listener_handle.take() {
            handle.abort();
        }
    }
}