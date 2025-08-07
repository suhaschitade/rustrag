use crate::utils::{Error, Result};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Key, Nonce,
};
use base64::{engine::general_purpose, Engine as _};
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Master encryption key for data at rest
    pub master_key: String,
    /// Encryption algorithm to use
    pub algorithm: EncryptionAlgorithm,
    /// Key rotation interval in days
    pub key_rotation_days: u32,
    /// Whether to enable field-level encryption
    pub enable_field_encryption: bool,
    /// Fields that should be encrypted in database
    pub encrypted_fields: Vec<String>,
    /// Whether to compress data before encryption
    pub compress_before_encrypt: bool,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            master_key: Self::generate_default_key(),
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            key_rotation_days: 90,
            enable_field_encryption: true,
            encrypted_fields: vec![
                "content".to_string(),
                "metadata".to_string(),
                "user_data".to_string(),
                "api_key".to_string(),
            ],
            compress_before_encrypt: true,
        }
    }
}

impl EncryptionConfig {
    fn generate_default_key() -> String {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        general_purpose::STANDARD.encode(key)
    }
}

/// Supported encryption algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    Aes256Gcm,
}

/// Encryption errors
#[derive(Debug, thiserror::Error)]
pub enum EncryptionError {
    #[error("Invalid key format: {0}")]
    InvalidKey(String),
    #[error("Encryption failed: {0}")]
    EncryptionFailed(String),
    #[error("Decryption failed: {0}")]
    DecryptionFailed(String),
    #[error("Key rotation failed: {0}")]
    KeyRotationFailed(String),
    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// Encrypted data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    /// Base64 encoded encrypted data
    pub data: String,
    /// Base64 encoded nonce/IV
    pub nonce: String,
    /// Algorithm used for encryption
    pub algorithm: EncryptionAlgorithm,
    /// Key version for rotation support
    pub key_version: u32,
    /// Timestamp when encrypted
    pub encrypted_at: chrono::DateTime<chrono::Utc>,
    /// Optional compression info
    pub compressed: bool,
}

/// Service for handling data encryption at rest
pub struct EncryptionService {
    config: EncryptionConfig,
    cipher: Aes256Gcm,
    current_key_version: u32,
    key_history: HashMap<u32, Vec<u8>>,
}

impl EncryptionService {
    /// Create a new encryption service
    pub fn new(config: EncryptionConfig) -> Result<Self> {
        let key_bytes = Self::decode_key(&config.master_key)?;
        let key: &Key<Aes256Gcm> = key_bytes.as_slice().into();
        let cipher = Aes256Gcm::new(key);

        let mut key_history = HashMap::new();
        key_history.insert(1, key_bytes.clone());

        Ok(Self {
            config,
            cipher,
            current_key_version: 1,
            key_history,
        })
    }

    /// Create encryption service with auto-generated key
    pub fn with_auto_key() -> Result<Self> {
        let config = EncryptionConfig::default();
        Self::new(config)
    }

    /// Decrypt a key from base64
    fn decode_key(key_str: &str) -> Result<Vec<u8>> {
        general_purpose::STANDARD
            .decode(key_str)
            .map_err(|e| Error::security(format!("Invalid key format: {}", e)))
    }

    /// Generate a random nonce
    fn generate_nonce() -> [u8; 12] {
        let mut nonce = [0u8; 12];
        OsRng.fill_bytes(&mut nonce);
        nonce
    }

    /// Encrypt data using AES-256-GCM
    pub fn encrypt(&self, data: &[u8]) -> Result<EncryptedData> {
        let nonce_bytes = Self::generate_nonce();
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Optionally compress data before encryption
        let data_to_encrypt = if self.config.compress_before_encrypt {
            self.compress_data(data)?
        } else {
            data.to_vec()
        };

        // Encrypt the data
        let encrypted_data = self.cipher
            .encrypt(nonce, data_to_encrypt.as_ref())
            .map_err(|e| Error::security(format!("Encryption failed: {}", e)))?;

        Ok(EncryptedData {
            data: general_purpose::STANDARD.encode(encrypted_data),
            nonce: general_purpose::STANDARD.encode(nonce_bytes),
            algorithm: self.config.algorithm,
            key_version: self.current_key_version,
            encrypted_at: chrono::Utc::now(),
            compressed: self.config.compress_before_encrypt,
        })
    }

    /// Decrypt data
    pub fn decrypt(&self, encrypted_data: &EncryptedData) -> Result<Vec<u8>> {
        // Get the correct key version
        let key_bytes = self.key_history
            .get(&encrypted_data.key_version)
            .ok_or_else(|| Error::security(format!(
                "Key version {} not found", 
                encrypted_data.key_version
            )))?;

        let key: &Key<Aes256Gcm> = key_bytes.as_slice().into();
        let cipher = Aes256Gcm::new(key);

        // Decode nonce and encrypted data
        let nonce_bytes = general_purpose::STANDARD
            .decode(&encrypted_data.nonce)
            .map_err(|e| Error::security(format!("Invalid nonce: {}", e)))?;
        
        let encrypted_bytes = general_purpose::STANDARD
            .decode(&encrypted_data.data)
            .map_err(|e| Error::security(format!("Invalid encrypted data: {}", e)))?;

        let nonce = Nonce::from_slice(&nonce_bytes);

        // Decrypt the data
        let decrypted_data = cipher
            .decrypt(nonce, encrypted_bytes.as_ref())
            .map_err(|e| Error::security(format!("Decryption failed: {}", e)))?;

        // Decompress if needed
        if encrypted_data.compressed {
            self.decompress_data(&decrypted_data)
        } else {
            Ok(decrypted_data)
        }
    }

    /// Encrypt a string
    pub fn encrypt_string(&self, text: &str) -> Result<EncryptedData> {
        self.encrypt(text.as_bytes())
    }

    /// Decrypt to string
    pub fn decrypt_string(&self, encrypted_data: &EncryptedData) -> Result<String> {
        let decrypted_bytes = self.decrypt(encrypted_data)?;
        String::from_utf8(decrypted_bytes)
            .map_err(|e| Error::security(format!("Invalid UTF-8: {}", e)))
    }

    /// Rotate encryption key
    pub fn rotate_key(&mut self, new_key: Option<String>) -> Result<u32> {
        let new_key = new_key.unwrap_or_else(|| {
            let mut key = [0u8; 32];
            OsRng.fill_bytes(&mut key);
            general_purpose::STANDARD.encode(key)
        });

        let key_bytes = Self::decode_key(&new_key)?;
        let key: &Key<Aes256Gcm> = key_bytes.as_slice().into();
        let new_cipher = Aes256Gcm::new(key);

        // Increment version
        let new_version = self.current_key_version + 1;
        
        // Store old key for decryption of existing data
        self.key_history.insert(new_version, key_bytes);
        
        // Update current key
        self.cipher = new_cipher;
        self.current_key_version = new_version;
        self.config.master_key = new_key;

        info!("Encryption key rotated to version {}", new_version);
        Ok(new_version)
    }

    /// Check if key rotation is needed
    pub fn needs_key_rotation(&self) -> bool {
        // Simple time-based rotation check
        let rotation_interval = chrono::Duration::days(self.config.key_rotation_days as i64);
        let next_rotation = chrono::Utc::now() - rotation_interval;
        
        // In a real implementation, we'd track when the key was last rotated
        // For now, return false to avoid automatic rotation
        false
    }

    /// Compress data using zlib
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::{write::ZlibEncoder, Compression};
        use std::io::Write;
        
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)
            .map_err(|e| Error::security(format!("Compression failed: {}", e)))?;
        encoder.finish()
            .map_err(|e| Error::security(format!("Compression failed: {}", e)))
    }

    /// Decompress data using zlib
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::ZlibDecoder;
        use std::io::Read;
        
        let mut decoder = ZlibDecoder::new(data);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)
            .map_err(|e| Error::security(format!("Decompression failed: {}", e)))?;
        Ok(result)
    }

    /// Get encryption statistics
    pub fn get_stats(&self) -> EncryptionStats {
        EncryptionStats {
            algorithm: self.config.algorithm,
            current_key_version: self.current_key_version,
            total_key_versions: self.key_history.len(),
            compression_enabled: self.config.compress_before_encrypt,
            field_encryption_enabled: self.config.enable_field_encryption,
            encrypted_fields: self.config.encrypted_fields.clone(),
        }
    }
}

/// Encryption statistics
#[derive(Debug, Clone, Serialize)]
pub struct EncryptionStats {
    pub algorithm: EncryptionAlgorithm,
    pub current_key_version: u32,
    pub total_key_versions: usize,
    pub compression_enabled: bool,
    pub field_encryption_enabled: bool,
    pub encrypted_fields: Vec<String>,
}

/// Trait for encrypting specific data types
pub trait DataEncryption {
    fn encrypt_sensitive_fields(&mut self, encryption_service: &EncryptionService) -> Result<()>;
    fn decrypt_sensitive_fields(&mut self, encryption_service: &EncryptionService) -> Result<()>;
}

/// Field-level encryption helper
pub struct FieldEncryption;

impl FieldEncryption {
    /// Encrypt specific fields in a JSON object
    pub fn encrypt_json_fields(
        json: &mut serde_json::Value,
        fields_to_encrypt: &[String],
        encryption_service: &EncryptionService,
    ) -> Result<()> {
        if let serde_json::Value::Object(obj) = json {
            for field_name in fields_to_encrypt {
                if let Some(field_value) = obj.get_mut(field_name) {
                    if !field_value.is_null() {
                        let field_str = field_value.to_string();
                        let encrypted = encryption_service.encrypt_string(&field_str)?;
                        let encrypted_json = serde_json::to_value(encrypted)
                            .map_err(|e| Error::security(format!("JSON serialization failed: {}", e)))?;
                        *field_value = encrypted_json;
                    }
                }
            }
        }
        Ok(())
    }

    /// Decrypt specific fields in a JSON object
    pub fn decrypt_json_fields(
        json: &mut serde_json::Value,
        fields_to_decrypt: &[String],
        encryption_service: &EncryptionService,
    ) -> Result<()> {
        if let serde_json::Value::Object(obj) = json {
            for field_name in fields_to_decrypt {
                if let Some(field_value) = obj.get_mut(field_name) {
                    if !field_value.is_null() {
                        let encrypted_data: EncryptedData = serde_json::from_value(field_value.clone())
                            .map_err(|e| Error::security(format!("Invalid encrypted field: {}", e)))?;
                        let decrypted = encryption_service.decrypt_string(&encrypted_data)?;
                        let decrypted_json: serde_json::Value = serde_json::from_str(&decrypted)
                            .map_err(|e| Error::security(format!("Invalid JSON in decrypted field: {}", e)))?;
                        *field_value = decrypted_json;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_decryption() {
        let service = EncryptionService::with_auto_key().unwrap();
        let test_data = "Hello, World! This is sensitive data.";
        
        let encrypted = service.encrypt_string(test_data).unwrap();
        let decrypted = service.decrypt_string(&encrypted).unwrap();
        
        assert_eq!(test_data, decrypted);
        assert_ne!(encrypted.data, general_purpose::STANDARD.encode(test_data));
    }

    #[test]
    fn test_key_rotation() {
        let mut service = EncryptionService::with_auto_key().unwrap();
        let test_data = "Test data for key rotation";
        
        // Encrypt with original key
        let encrypted_v1 = service.encrypt_string(test_data).unwrap();
        assert_eq!(encrypted_v1.key_version, 1);
        
        // Rotate key
        let new_version = service.rotate_key(None).unwrap();
        assert_eq!(new_version, 2);
        
        // Encrypt with new key
        let encrypted_v2 = service.encrypt_string(test_data).unwrap();
        assert_eq!(encrypted_v2.key_version, 2);
        
        // Should be able to decrypt both versions
        let decrypted_v1 = service.decrypt_string(&encrypted_v1).unwrap();
        let decrypted_v2 = service.decrypt_string(&encrypted_v2).unwrap();
        
        assert_eq!(decrypted_v1, test_data);
        assert_eq!(decrypted_v2, test_data);
    }

    #[test]
    fn test_field_encryption() {
        let service = EncryptionService::with_auto_key().unwrap();
        let mut json = serde_json::json!({
            "id": "123",
            "content": "This is sensitive content",
            "metadata": {"key": "value"},
            "public_field": "This is not encrypted"
        });
        
        let fields_to_encrypt = vec!["content".to_string(), "metadata".to_string()];
        
        // Encrypt specific fields
        FieldEncryption::encrypt_json_fields(&mut json, &fields_to_encrypt, &service).unwrap();
        
        // Verify that only specified fields are encrypted
        assert!(json["content"].is_object()); // Now contains EncryptedData
        assert!(json["metadata"].is_object()); // Now contains EncryptedData
        assert_eq!(json["public_field"], "This is not encrypted");
        
        // Decrypt fields back
        FieldEncryption::decrypt_json_fields(&mut json, &fields_to_encrypt, &service).unwrap();
        
        assert_eq!(json["content"], "\"This is sensitive content\"");
        assert_eq!(json["public_field"], "This is not encrypted");
    }
}
