use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the embedding service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Primary embedding provider
    pub primary_provider: String,
    /// Fallback providers in order of preference
    pub fallback_providers: Vec<String>,
    /// Provider-specific configurations
    pub providers: HashMap<String, ProviderConfig>,
    /// Default model configuration
    pub default_model: EmbeddingModel,
    /// Maximum batch size for embedding generation
    pub max_batch_size: usize,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum retries for failed requests
    pub max_retries: u32,
    /// Enable request caching
    pub enable_caching: bool,
}

/// Configuration for a specific embedding provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Whether this provider is enabled
    pub enabled: bool,
    /// API key or authentication token (if applicable)
    pub api_key: Option<String>,
    /// Base URL for API endpoints (if applicable)
    pub base_url: Option<String>,
    /// Model configuration
    pub model: EmbeddingModel,
    /// Provider-specific options
    pub options: HashMap<String, serde_json::Value>,
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    /// Model name or identifier
    pub name: String,
    /// Expected embedding dimension
    pub dimension: usize,
    /// Maximum input tokens/characters
    pub max_input_length: usize,
    /// Model-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        let mut providers = HashMap::new();
        
        // Default mock provider configuration
        providers.insert(
            "mock".to_string(),
            ProviderConfig {
                enabled: true,
                api_key: None,
                base_url: None,
                model: EmbeddingModel::default(),
                options: HashMap::new(),
            },
        );

        Self {
            primary_provider: "mock".to_string(),
            fallback_providers: vec![],
            providers,
            default_model: EmbeddingModel::default(),
            max_batch_size: 100,
            timeout_seconds: 30,
            max_retries: 3,
            enable_caching: true,
        }
    }
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self {
            name: "mock-embeddings-v1".to_string(),
            dimension: 1536, // OpenAI ada-002 compatible
            max_input_length: 8192,
            parameters: HashMap::new(),
        }
    }
}

impl EmbeddingConfig {
    /// Create a new configuration with OpenAI as primary provider
    pub fn with_openai(api_key: String) -> Self {
        let mut config = Self::default();
        
        config.primary_provider = "openai".to_string();
        config.providers.insert(
            "openai".to_string(),
            ProviderConfig {
                enabled: true,
                api_key: Some(api_key),
                base_url: Some("https://api.openai.com/v1".to_string()),
                model: EmbeddingModel {
                    name: "text-embedding-ada-002".to_string(),
                    dimension: 1536,
                    max_input_length: 8192,
                    parameters: HashMap::new(),
                },
                options: HashMap::new(),
            },
        );
        
        config
    }
    
    /// Create a new configuration with ONNX as primary provider
    pub fn with_onnx(model_path: String) -> Self {
        let mut config = Self::default();
        let mut options = HashMap::new();
        options.insert("model_path".to_string(), serde_json::Value::String(model_path));
        
        config.primary_provider = "onnx".to_string();
        config.providers.insert(
            "onnx".to_string(),
            ProviderConfig {
                enabled: true,
                api_key: None,
                base_url: None,
                model: EmbeddingModel {
                    name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                    dimension: 384,
                    max_input_length: 512,
                    parameters: HashMap::new(),
                },
                options,
            },
        );
        
        config
    }
    
    /// Add a fallback provider
    pub fn add_fallback(mut self, provider_name: String, config: ProviderConfig) -> Self {
        self.fallback_providers.push(provider_name.clone());
        self.providers.insert(provider_name, config);
        self
    }
    
    /// Get provider configuration by name
    pub fn get_provider_config(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.get(name)
    }
    
    /// Get all enabled providers
    pub fn enabled_providers(&self) -> Vec<&String> {
        self.providers
            .iter()
            .filter(|(_, config)| config.enabled)
            .map(|(name, _)| name)
            .collect()
    }
}
