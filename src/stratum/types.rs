use thiserror::Error;

#[derive(Error, Debug)]
/// Stratum protocol error types
pub enum StratumError {
    /// Connection to pool failed
    #[error("Connection failed")]
    Connection,
    /// Subscription to pool failed
    #[error("Subscription failed")]
    SubscriptionFailed,
}