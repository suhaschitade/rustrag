use serde::Deserialize;

/// Application settings loaded from environment variables or configuration files
#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    pub server: ServerSettings,
    pub database: DatabaseSettings,
    pub log: LogSettings,
    pub openai: OpenAISettings,
    pub auth: AuthSettings,
    pub rate_limit: RateLimitSettings,
}

/// Authentication settings
#[derive(Debug, Deserialize, Clone)]
pub struct AuthSettings {
    /// List of static API keys for basic authentication
    pub static_api_keys: Vec<String>,
    /// Enable or disable authentication
    pub enabled: bool,
}

/// Rate limiting settings
#[derive(Debug, Deserialize, Clone)]
pub struct RateLimitSettings {
    /// Enable or disable rate limiting
    pub enabled: bool,
    /// Requests per minute for unauthenticated users (by IP)
    pub unauthenticated_rpm: u32,
    /// Requests per minute for authenticated users (by API key)
    pub authenticated_rpm: u32,
}

/// Server-related settings
#[derive(Debug, Deserialize, Clone)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
    pub workers: usize,
}

/// Database-related settings
#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseSettings {
    pub url: String,
    pub max_connections: u32,
}

/// Logging settings
#[derive(Debug, Deserialize, Clone)]
pub struct LogSettings {
    pub level: String,
    pub format: String,
}

/// OpenAI-related settings
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAISettings {
    pub api_key: String,
    pub model: String,
}

impl Settings {
    /// Load settings from environment with default values
    pub fn from_env() -> Result<Self, config::ConfigError> {
        // For now, return default settings since config parsing is complex
        // In a real implementation, this would load from environment variables
        Ok(Settings {
            server: ServerSettings {
                host: std::env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: std::env::var("SERVER_PORT")
                    .unwrap_or_else(|_| "8000".to_string())
                    .parse()
                    .unwrap_or(8000),
                workers: std::env::var("SERVER_WORKERS")
                    .unwrap_or_else(|_| "4".to_string())
                    .parse()
                    .unwrap_or(4),
            },
            database: DatabaseSettings {
                url: std::env::var("DATABASE_URL")
                    .unwrap_or_else(|_| "postgresql://localhost/rustrag".to_string()),
                max_connections: std::env::var("DATABASE_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .unwrap_or(10),
            },
            log: LogSettings {
                level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
                format: std::env::var("LOG_FORMAT").unwrap_or_else(|_| "text".to_string()),
            },
            openai: OpenAISettings {
                api_key: std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "not_set".to_string()),
                model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4".to_string()),
            },
            auth: AuthSettings {
                static_api_keys: std::env::var("STATIC_API_KEYS")
                    .unwrap_or_default()
                    .split(',')
                    .filter(|key| !key.trim().is_empty())
                    .map(|key| key.trim().to_string())
                    .collect(),
                enabled: std::env::var("AUTH_ENABLED")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
            },
            rate_limit: RateLimitSettings {
                enabled: std::env::var("RATE_LIMIT_ENABLED")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                unauthenticated_rpm: std::env::var("RATE_LIMIT_UNAUTHENTICATED_RPM")
                    .unwrap_or_else(|_| "60".to_string())
                    .parse()
                    .unwrap_or(60),
                authenticated_rpm: std::env::var("RATE_LIMIT_AUTHENTICATED_RPM")
                    .unwrap_or_else(|_| "1000".to_string())
                    .parse()
                    .unwrap_or(1000),
            },
        })
    }
}
