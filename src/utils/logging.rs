use tracing_subscriber::{EnvFilter, FmtSubscriber};
use crate::utils::Result;

/// Initialize the tracing subscriber with default settings
pub fn init_tracing() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| crate::utils::Error::Logging(e.to_string()))
}

/// Initialize the tracing subscriber with custom settings
pub fn init_tracing_with_config(_settings: &crate::config::Settings) -> Result<()> {
    // Create a simple default settings for now since we don't have log config
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| crate::utils::Error::Logging(e.to_string()))
}
