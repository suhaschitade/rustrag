use crate::utils::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// TLS configuration for encryption in transit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Whether TLS is enabled
    pub enabled: bool,
    /// Path to certificate file (PEM format)
    pub cert_file: PathBuf,
    /// Path to private key file (PEM format)
    pub key_file: PathBuf,
    /// Path to CA certificate file for client verification
    pub ca_cert_file: Option<PathBuf>,
    /// TLS version to use
    pub tls_version: TlsVersion,
    /// Supported cipher suites
    pub cipher_suites: Vec<String>,
    /// Whether to require client certificates
    pub require_client_cert: bool,
    /// Certificate auto-renewal settings
    pub auto_renewal: AutoRenewalConfig,
    /// OCSP stapling configuration
    pub ocsp_stapling: bool,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cert_file: PathBuf::from("certs/server.crt"),
            key_file: PathBuf::from("certs/server.key"),
            ca_cert_file: None,
            tls_version: TlsVersion::V1_3,
            cipher_suites: vec![
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_AES_128_GCM_SHA256".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
            ],
            require_client_cert: false,
            auto_renewal: AutoRenewalConfig::default(),
            ocsp_stapling: false,
        }
    }
}

/// Supported TLS versions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TlsVersion {
    V1_2,
    V1_3,
}

/// Certificate auto-renewal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoRenewalConfig {
    /// Whether auto-renewal is enabled
    pub enabled: bool,
    /// Days before expiry to trigger renewal
    pub renewal_days_before_expiry: u32,
    /// ACME provider for automatic certificate renewal
    pub acme_provider: Option<AcmeProvider>,
    /// Email for certificate renewal notifications
    pub notification_email: Option<String>,
}

impl Default for AutoRenewalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            renewal_days_before_expiry: 30,
            acme_provider: None,
            notification_email: None,
        }
    }
}

/// ACME provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcmeProvider {
    /// ACME directory URL
    pub directory_url: String,
    /// Account email
    pub account_email: String,
    /// Challenge type
    pub challenge_type: AcmeChallenge,
}

/// ACME challenge types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcmeChallenge {
    Http01,
    Dns01,
}

/// Certificate information
#[derive(Debug, Clone, Serialize)]
pub struct CertificateInfo {
    /// Certificate subject
    pub subject: String,
    /// Certificate issuer
    pub issuer: String,
    /// Not valid before timestamp
    pub not_before: chrono::DateTime<chrono::Utc>,
    /// Not valid after timestamp
    pub not_after: chrono::DateTime<chrono::Utc>,
    /// Serial number
    pub serial_number: String,
    /// Subject alternative names
    pub san: Vec<String>,
    /// Days until expiry
    pub days_until_expiry: i64,
    /// Whether certificate is expired
    pub is_expired: bool,
    /// Whether certificate is self-signed
    pub is_self_signed: bool,
}

/// TLS manager for certificate operations
pub struct TlsManager {
    config: TlsConfig,
}

impl TlsManager {
    /// Create a new TLS manager
    pub fn new(config: TlsConfig) -> Self {
        Self { config }
    }

    /// Validate TLS configuration
    pub fn validate_config(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check if certificate files exist
        if !self.config.cert_file.exists() {
            return Err(Error::security(format!(
                "Certificate file not found: {}",
                self.config.cert_file.display()
            )));
        }

        if !self.config.key_file.exists() {
            return Err(Error::security(format!(
                "Private key file not found: {}",
                self.config.key_file.display()
            )));
        }

        // Validate certificate
        let cert_info = self.get_certificate_info()?;
        
        if cert_info.is_expired {
            warn!("Certificate is expired: expires on {}", cert_info.not_after);
            return Err(Error::security("Certificate is expired".to_string()));
        }

        if cert_info.days_until_expiry <= 30 {
            warn!(
                "Certificate expires soon: {} days remaining",
                cert_info.days_until_expiry
            );
        }

        info!("TLS configuration validated successfully");
        Ok(())
    }

    /// Get certificate information from the configured certificate file
    pub fn get_certificate_info(&self) -> Result<CertificateInfo> {
        use std::fs;
        use x509_parser::prelude::*;

        let cert_pem = fs::read_to_string(&self.config.cert_file)
            .map_err(|e| Error::security(format!("Failed to read certificate: {}", e)))?;

        // Parse PEM to get DER data
        let der_data = self.parse_pem_certificate(&cert_pem)?;
        
        // Parse X.509 certificate
        let (_, cert) = X509Certificate::from_der(&der_data)
            .map_err(|e| Error::security(format!("Failed to parse certificate: {}", e)))?;

        let subject = cert.subject().to_string();
        let issuer = cert.issuer().to_string();
        let serial_number = format!("{:X}", cert.serial);

        let not_before = chrono::DateTime::from_timestamp(
            cert.validity().not_before.timestamp(),
            0
        ).unwrap_or_default();

        let not_after = chrono::DateTime::from_timestamp(
            cert.validity().not_after.timestamp(),
            0
        ).unwrap_or_default();

        let now = chrono::Utc::now();
        let days_until_expiry = (not_after - now).num_days();
        let is_expired = now > not_after;
        let is_self_signed = subject == issuer;

        // Extract SAN (Subject Alternative Names)
        let mut san = Vec::new();
        if let Ok(Some(ext)) = cert.get_extension_unique(&x509_parser::oid_registry::OID_X509_EXT_SUBJECT_ALT_NAME) {
            if let ParsedExtension::SubjectAlternativeName(san_ext) = &ext.parsed_extension() {
                for name in &san_ext.general_names {
                    if let GeneralName::DNSName(dns_name) = name {
                        san.push(dns_name.to_string());
                    }
                }
            }
        }

        Ok(CertificateInfo {
            subject,
            issuer,
            not_before,
            not_after,
            serial_number,
            san,
            days_until_expiry,
            is_expired,
            is_self_signed,
        })
    }

    /// Parse PEM certificate to get DER data
    fn parse_pem_certificate(&self, pem_data: &str) -> Result<Vec<u8>> {
        let pem = pem::parse(pem_data)
            .map_err(|e| Error::security(format!("Failed to parse PEM: {}", e)))?;
        
        if pem.tag != "CERTIFICATE" {
            return Err(Error::security(format!(
                "Expected CERTIFICATE PEM, got {}",
                pem.tag
            )));
        }

        Ok(pem.contents)
    }

    /// Generate self-signed certificate for development
    pub fn generate_self_signed_certificate(&self, domain: &str) -> Result<()> {
        use std::process::Command;

        info!("Generating self-signed certificate for domain: {}", domain);

        // Create certs directory if it doesn't exist
        if let Some(parent) = self.config.cert_file.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::security(format!("Failed to create certs directory: {}", e)))?;
        }

        // Generate private key
        let key_output = Command::new("openssl")
            .args([
                "genrsa",
                "-out",
                &self.config.key_file.to_string_lossy(),
                "2048",
            ])
            .output()
            .map_err(|e| Error::security(format!("Failed to generate private key: {}", e)))?;

        if !key_output.status.success() {
            return Err(Error::security(format!(
                "OpenSSL key generation failed: {}",
                String::from_utf8_lossy(&key_output.stderr)
            )));
        }

        // Generate certificate
        let cert_output = Command::new("openssl")
            .args([
                "req",
                "-new",
                "-x509",
                "-key",
                &self.config.key_file.to_string_lossy(),
                "-out",
                &self.config.cert_file.to_string_lossy(),
                "-days",
                "365",
                "-subj",
                &format!("/CN={}", domain),
                "-extensions",
                "SAN",
                "-config",
                "/dev/stdin",
            ])
            .input(format!(
                "[req]\n\
                 distinguished_name = req_distinguished_name\n\
                 [SAN]\n\
                 subjectAltName = DNS:{}\n",
                domain
            ).as_bytes())
            .output()
            .map_err(|e| Error::security(format!("Failed to generate certificate: {}", e)))?;

        if !cert_output.status.success() {
            return Err(Error::security(format!(
                "OpenSSL certificate generation failed: {}",
                String::from_utf8_lossy(&cert_output.stderr)
            )));
        }

        info!("Self-signed certificate generated successfully");
        Ok(())
    }

    /// Check if certificate needs renewal
    pub fn needs_renewal(&self) -> Result<bool> {
        let cert_info = self.get_certificate_info()?;
        let renewal_threshold = self.config.auto_renewal.renewal_days_before_expiry as i64;
        
        Ok(cert_info.days_until_expiry <= renewal_threshold)
    }

    /// Renew certificate using ACME if configured
    pub async fn renew_certificate(&self) -> Result<()> {
        if !self.config.auto_renewal.enabled {
            return Err(Error::security("Auto-renewal is not enabled".to_string()));
        }

        let acme_config = self.config.auto_renewal.acme_provider.as_ref()
            .ok_or_else(|| Error::security("ACME provider not configured".to_string()))?;

        info!("Starting certificate renewal with ACME");
        
        // In a real implementation, you would integrate with an ACME client library
        // For now, we'll just log the attempt
        warn!("ACME certificate renewal not yet implemented");
        warn!("ACME directory: {}", acme_config.directory_url);
        warn!("Account email: {}", acme_config.account_email);
        warn!("Challenge type: {:?}", acme_config.challenge_type);

        // Placeholder for actual ACME implementation
        Err(Error::security("ACME renewal not yet implemented".to_string()))
    }

    /// Get TLS configuration for axum server
    pub fn get_axum_tls_config(&self) -> Result<axum_server::tls_rustls::RustlsConfig> {
        if !self.config.enabled {
            return Err(Error::security("TLS is not enabled".to_string()));
        }

        let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(
            &self.config.cert_file,
            &self.config.key_file,
        );

        config.map_err(|e| Error::security(format!("Failed to create TLS config: {}", e)))
    }

    /// Get security headers for HTTPS responses
    pub fn get_security_headers(&self) -> Vec<(String, String)> {
        vec![
            ("Strict-Transport-Security".to_string(), "max-age=31536000; includeSubDomains".to_string()),
            ("X-Content-Type-Options".to_string(), "nosniff".to_string()),
            ("X-Frame-Options".to_string(), "DENY".to_string()),
            ("X-XSS-Protection".to_string(), "1; mode=block".to_string()),
            ("Referrer-Policy".to_string(), "strict-origin-when-cross-origin".to_string()),
            ("Content-Security-Policy".to_string(), "default-src 'self'".to_string()),
        ]
    }
}

/// TLS certificate monitoring task
pub struct CertificateMonitor {
    tls_manager: TlsManager,
    check_interval_hours: u64,
}

impl CertificateMonitor {
    /// Create a new certificate monitor
    pub fn new(tls_manager: TlsManager, check_interval_hours: u64) -> Self {
        Self {
            tls_manager,
            check_interval_hours,
        }
    }

    /// Start monitoring certificate expiry
    pub async fn start_monitoring(self) {
        let mut interval = tokio::time::interval(
            tokio::time::Duration::from_secs(self.check_interval_hours * 3600)
        );

        loop {
            interval.tick().await;
            
            match self.tls_manager.get_certificate_info() {
                Ok(cert_info) => {
                    debug!("Certificate expires in {} days", cert_info.days_until_expiry);
                    
                    if cert_info.days_until_expiry <= 30 {
                        warn!("Certificate expires soon: {} days", cert_info.days_until_expiry);
                        
                        if self.tls_manager.config.auto_renewal.enabled {
                            if let Err(e) = self.tls_manager.renew_certificate().await {
                                warn!("Failed to renew certificate: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to check certificate: {}", e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_tls_config_validation() {
        // Test with disabled TLS
        let config = TlsConfig {
            enabled: false,
            ..Default::default()
        };
        let manager = TlsManager::new(config);
        assert!(manager.validate_config().is_ok());
    }

    #[test]
    fn test_certificate_generation() {
        let temp_dir = tempdir().unwrap();
        let cert_path = temp_dir.path().join("test.crt");
        let key_path = temp_dir.path().join("test.key");

        let config = TlsConfig {
            enabled: true,
            cert_file: cert_path.clone(),
            key_file: key_path.clone(),
            ..Default::default()
        };

        let manager = TlsManager::new(config);
        
        // Generate self-signed certificate (requires openssl)
        if let Err(e) = manager.generate_self_signed_certificate("localhost") {
            // Skip test if openssl is not available
            if e.to_string().contains("Failed to generate") {
                return;
            }
        }

        // Verify files were created
        assert!(cert_path.exists());
        assert!(key_path.exists());
    }

    #[test]
    fn test_security_headers() {
        let config = TlsConfig::default();
        let manager = TlsManager::new(config);
        let headers = manager.get_security_headers();
        
        assert!(!headers.is_empty());
        assert!(headers.iter().any(|(k, _)| k == "Strict-Transport-Security"));
        assert!(headers.iter().any(|(k, _)| k == "X-Content-Type-Options"));
        assert!(headers.iter().any(|(k, _)| k == "Content-Security-Policy"));
    }
}
