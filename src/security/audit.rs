use crate::utils::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tracing::{debug, error, info, warn};
use tokio::sync::RwLock;

/// Configuration for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Whether audit logging is enabled
    pub enabled: bool,
    /// Log level for audit events
    pub log_level: AuditLogLevel,
    /// Storage backend for audit logs
    pub storage_backend: AuditStorageBackend,
    /// Retention period for audit logs (in days)
    pub retention_days: u32,
    /// Whether to log sensitive data (be careful with this)
    pub log_sensitive_data: bool,
    /// Maximum batch size for bulk logging
    pub batch_size: usize,
    /// Batch timeout in seconds
    pub batch_timeout_seconds: u64,
    /// Storage configuration
    pub storage_config: AuditStorageConfig,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_level: AuditLogLevel::Info,
            storage_backend: AuditStorageBackend::Database,
            retention_days: 90,
            log_sensitive_data: false,
            batch_size: 100,
            batch_timeout_seconds: 30,
            storage_config: AuditStorageConfig::default(),
        }
    }
}

/// Audit log levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AuditLogLevel {
    Critical,
    High,
    Medium,
    Info,
    Debug,
}

/// Storage backends for audit logs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditStorageBackend {
    Database,
    File,
    Elasticsearch,
    CloudWatch,
    Syslog,
}

/// Storage configuration for different backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditStorageConfig {
    /// Database connection string (for Database backend)
    pub database_url: Option<String>,
    /// File path (for File backend)
    pub file_path: Option<String>,
    /// Elasticsearch configuration
    pub elasticsearch_config: Option<ElasticsearchConfig>,
    /// CloudWatch configuration
    pub cloudwatch_config: Option<CloudWatchConfig>,
    /// Syslog configuration
    pub syslog_config: Option<SyslogConfig>,
}

impl Default for AuditStorageConfig {
    fn default() -> Self {
        Self {
            database_url: None,
            file_path: Some("logs/audit.jsonl".to_string()),
            elasticsearch_config: None,
            cloudwatch_config: None,
            syslog_config: None,
        }
    }
}

/// Elasticsearch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticsearchConfig {
    pub url: String,
    pub index_prefix: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

/// AWS CloudWatch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudWatchConfig {
    pub log_group: String,
    pub log_stream: String,
    pub region: String,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
}

/// Syslog configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyslogConfig {
    pub host: String,
    pub port: u16,
    pub facility: String,
    pub protocol: SyslogProtocol,
}

/// Syslog protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyslogProtocol {
    Udp,
    Tcp,
    Tls,
}

/// Types of audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    // Authentication events
    UserLogin,
    UserLogout,
    UserLoginFailed,
    PasswordChanged,
    TokenGenerated,
    TokenRevoked,
    
    // Document events
    DocumentUploaded,
    DocumentDeleted,
    DocumentModified,
    DocumentViewed,
    DocumentDownloaded,
    
    // Query events
    QueryExecuted,
    QueryFailed,
    
    // System events
    SystemStartup,
    SystemShutdown,
    ConfigurationChanged,
    
    // Security events
    SecurityViolation,
    AccessDenied,
    PrivilegeEscalation,
    DataExfiltration,
    
    // Admin events
    UserCreated,
    UserDeleted,
    UserModified,
    RoleChanged,
    PermissionChanged,
    
    // Data events
    DataEncrypted,
    DataDecrypted,
    DataExported,
    DataImported,
    DataPurged,
    
    // Compliance events
    RetentionPolicyApplied,
    DataSubjectRequest,
    ConsentGranted,
    ConsentRevoked,
}

/// Audit event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event ID
    pub id: Uuid,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Type of event
    pub event_type: AuditEventType,
    /// User who triggered the event
    pub user_id: Option<String>,
    /// Session ID
    pub session_id: Option<String>,
    /// IP address
    pub ip_address: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Resource affected (document ID, query ID, etc.)
    pub resource_id: Option<String>,
    /// Resource type
    pub resource_type: Option<String>,
    /// Action performed
    pub action: String,
    /// Event severity
    pub severity: AuditLogLevel,
    /// Whether the action was successful
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Additional event details
    pub details: HashMap<String, serde_json::Value>,
    /// Request/response size in bytes
    pub data_size: Option<u64>,
    /// Processing duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Source component that generated the event
    pub source: String,
    /// Correlation ID for tracing related events
    pub correlation_id: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event
    pub fn new(
        event_type: AuditEventType,
        action: String,
        source: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type,
            user_id: None,
            session_id: None,
            ip_address: None,
            user_agent: None,
            resource_id: None,
            resource_type: None,
            action,
            severity: AuditLogLevel::Info,
            success: true,
            error_message: None,
            details: HashMap::new(),
            data_size: None,
            duration_ms: None,
            source,
            correlation_id: None,
        }
    }

    /// Set user context
    pub fn with_user(mut self, user_id: String, session_id: Option<String>) -> Self {
        self.user_id = Some(user_id);
        self.session_id = session_id;
        self
    }

    /// Set network context
    pub fn with_network(mut self, ip_address: String, user_agent: Option<String>) -> Self {
        self.ip_address = Some(ip_address);
        self.user_agent = user_agent;
        self
    }

    /// Set resource context
    pub fn with_resource(mut self, resource_id: String, resource_type: String) -> Self {
        self.resource_id = Some(resource_id);
        self.resource_type = Some(resource_type);
        self
    }

    /// Set severity level
    pub fn with_severity(mut self, severity: AuditLogLevel) -> Self {
        self.severity = severity;
        self
    }

    /// Mark as failed with error message
    pub fn with_error(mut self, error_message: String) -> Self {
        self.success = false;
        self.error_message = Some(error_message);
        self.severity = AuditLogLevel::High;
        self
    }

    /// Add custom detail
    pub fn with_detail(mut self, key: String, value: serde_json::Value) -> Self {
        self.details.insert(key, value);
        self
    }

    /// Set performance metrics
    pub fn with_metrics(mut self, data_size: Option<u64>, duration_ms: u64) -> Self {
        self.data_size = data_size;
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Set correlation ID for tracing
    pub fn with_correlation_id(mut self, correlation_id: String) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }
}

/// Audit query parameters for searching logs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditQuery {
    /// Filter by event types
    pub event_types: Option<Vec<AuditEventType>>,
    /// Filter by user ID
    pub user_id: Option<String>,
    /// Filter by resource ID
    pub resource_id: Option<String>,
    /// Filter by success status
    pub success: Option<bool>,
    /// Filter by date range
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    /// Filter by severity
    pub min_severity: Option<AuditLogLevel>,
    /// Search text in action or details
    pub search_text: Option<String>,
    /// Pagination
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    /// Sort order
    pub sort_by: Option<AuditSortBy>,
    pub sort_order: Option<SortOrder>,
}

/// Sort options for audit queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditSortBy {
    Timestamp,
    EventType,
    UserId,
    Severity,
    Duration,
}

/// Sort order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    Asc,
    Desc,
}

/// Audit report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    /// Report generation time
    pub generated_at: DateTime<Utc>,
    /// Total events in period
    pub total_events: u64,
    /// Events by type
    pub events_by_type: HashMap<String, u64>,
    /// Events by user
    pub events_by_user: HashMap<String, u64>,
    /// Events by severity
    pub events_by_severity: HashMap<String, u64>,
    /// Failed events count
    pub failed_events: u64,
    /// Security events count
    pub security_events: u64,
    /// Most active users
    pub top_users: Vec<(String, u64)>,
    /// Most common actions
    pub top_actions: Vec<(String, u64)>,
    /// Average response times
    pub avg_duration_ms: f64,
    /// Data volume processed
    pub total_data_bytes: u64,
}

/// Main audit logger service
pub struct AuditLogger {
    config: AuditConfig,
    event_buffer: Arc<RwLock<Vec<AuditEvent>>>,
    storage: Arc<dyn AuditStorage + Send + Sync>,
}

impl AuditLogger {
    /// Create a new audit logger
    pub async fn new(config: AuditConfig) -> Result<Self> {
        let storage = create_storage(&config).await?;
        
        Ok(Self {
            config,
            event_buffer: Arc::new(RwLock::new(Vec::new())),
            storage,
        })
    }

    /// Log an audit event
    pub async fn log_event(&self, event: AuditEvent) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Filter by log level
        if !self.should_log(&event) {
            return Ok(());
        }

        // Add to buffer
        {
            let mut buffer = self.event_buffer.write().await;
            buffer.push(event.clone());

            // Flush buffer if it's full
            if buffer.len() >= self.config.batch_size {
                let events: Vec<AuditEvent> = buffer.drain(..).collect();
                drop(buffer);
                
                self.flush_events(events).await?;
            }
        }

        // Log to tracing as well
        self.log_to_tracing(&event);

        Ok(())
    }

    /// Log authentication event
    pub async fn log_auth_event(&self, event_type: AuditEventType, user_id: Option<String>, ip: Option<String>, success: bool, error: Option<String>) -> Result<()> {
        let mut event = AuditEvent::new(
            event_type,
            format!("{:?}", event_type),
            "auth".to_string()
        );

        if let Some(user) = user_id {
            event = event.with_user(user, None);
        }

        if let Some(ip_addr) = ip {
            event = event.with_network(ip_addr, None);
        }

        if !success {
            event.success = false;
            if let Some(err) = error {
                event = event.with_error(err);
            }
        }

        self.log_event(event).await
    }

    /// Log document event
    pub async fn log_document_event(&self, event_type: AuditEventType, user_id: String, document_id: String, action: String, data_size: Option<u64>) -> Result<()> {
        let event = AuditEvent::new(event_type, action, "document".to_string())
            .with_user(user_id, None)
            .with_resource(document_id, "document".to_string())
            .with_metrics(data_size, 0);

        self.log_event(event).await
    }

    /// Log query event
    pub async fn log_query_event(&self, user_id: String, query: String, success: bool, duration_ms: u64, result_count: u32) -> Result<()> {
        let mut event = AuditEvent::new(
            AuditEventType::QueryExecuted,
            "query".to_string(),
            "query_engine".to_string()
        )
        .with_user(user_id, None)
        .with_metrics(None, duration_ms)
        .with_detail("query_text".to_string(), serde_json::Value::String(query))
        .with_detail("result_count".to_string(), serde_json::Value::Number(serde_json::Number::from(result_count)));

        if !success {
            event.success = false;
            event.event_type = AuditEventType::QueryFailed;
        }

        self.log_event(event).await
    }

    /// Log security violation
    pub async fn log_security_violation(&self, user_id: Option<String>, violation_type: String, details: String, ip: Option<String>) -> Result<()> {
        let mut event = AuditEvent::new(
            AuditEventType::SecurityViolation,
            violation_type,
            "security".to_string()
        )
        .with_severity(AuditLogLevel::Critical)
        .with_detail("violation_details".to_string(), serde_json::Value::String(details));

        if let Some(user) = user_id {
            event = event.with_user(user, None);
        }

        if let Some(ip_addr) = ip {
            event = event.with_network(ip_addr, None);
        }

        self.log_event(event).await
    }

    /// Query audit logs
    pub async fn query_events(&self, query: AuditQuery) -> Result<Vec<AuditEvent>> {
        self.storage.query_events(query).await
    }

    /// Generate audit report
    pub async fn generate_report(&self, start_date: DateTime<Utc>, end_date: DateTime<Utc>) -> Result<AuditReport> {
        let query = AuditQuery {
            start_date: Some(start_date),
            end_date: Some(end_date),
            ..Default::default()
        };

        let events = self.query_events(query).await?;
        
        let mut report = AuditReport {
            generated_at: Utc::now(),
            total_events: events.len() as u64,
            events_by_type: HashMap::new(),
            events_by_user: HashMap::new(),
            events_by_severity: HashMap::new(),
            failed_events: 0,
            security_events: 0,
            top_users: Vec::new(),
            top_actions: Vec::new(),
            avg_duration_ms: 0.0,
            total_data_bytes: 0,
        };

        // Process events for report
        let mut total_duration = 0u64;
        let mut duration_count = 0u64;

        for event in &events {
            // Count by type
            let type_key = format!("{:?}", event.event_type);
            *report.events_by_type.entry(type_key).or_insert(0) += 1;

            // Count by user
            if let Some(user) = &event.user_id {
                *report.events_by_user.entry(user.clone()).or_insert(0) += 1;
            }

            // Count by severity
            let severity_key = format!("{:?}", event.severity);
            *report.events_by_severity.entry(severity_key).or_insert(0) += 1;

            // Count failures
            if !event.success {
                report.failed_events += 1;
            }

            // Count security events
            match event.event_type {
                AuditEventType::SecurityViolation | AuditEventType::AccessDenied | 
                AuditEventType::PrivilegeEscalation | AuditEventType::DataExfiltration => {
                    report.security_events += 1;
                }
                _ => {}
            }

            // Calculate average duration
            if let Some(duration) = event.duration_ms {
                total_duration += duration;
                duration_count += 1;
            }

            // Sum data bytes
            if let Some(size) = event.data_size {
                report.total_data_bytes += size;
            }
        }

        // Calculate averages
        if duration_count > 0 {
            report.avg_duration_ms = total_duration as f64 / duration_count as f64;
        }

        // Sort top users and actions
        let mut user_counts: Vec<_> = report.events_by_user.iter().collect();
        user_counts.sort_by(|a, b| b.1.cmp(a.1));
        report.top_users = user_counts.into_iter()
            .take(10)
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        Ok(report)
    }

    /// Flush buffered events
    pub async fn flush(&self) -> Result<()> {
        let events = {
            let mut buffer = self.event_buffer.write().await;
            buffer.drain(..).collect()
        };

        if !events.is_empty() {
            self.flush_events(events).await?;
        }

        Ok(())
    }

    /// Check if event should be logged based on level
    fn should_log(&self, event: &AuditEvent) -> bool {
        let event_level = match event.severity {
            AuditLogLevel::Critical => 4,
            AuditLogLevel::High => 3,
            AuditLogLevel::Medium => 2,
            AuditLogLevel::Info => 1,
            AuditLogLevel::Debug => 0,
        };

        let config_level = match self.config.log_level {
            AuditLogLevel::Critical => 4,
            AuditLogLevel::High => 3,
            AuditLogLevel::Medium => 2,
            AuditLogLevel::Info => 1,
            AuditLogLevel::Debug => 0,
        };

        event_level >= config_level
    }

    /// Log event to tracing system
    fn log_to_tracing(&self, event: &AuditEvent) {
        let log_msg = format!(
            "AUDIT: {} - {} by user {} on resource {}",
            event.action,
            format!("{:?}", event.event_type),
            event.user_id.as_deref().unwrap_or("anonymous"),
            event.resource_id.as_deref().unwrap_or("unknown")
        );

        match event.severity {
            AuditLogLevel::Critical => error!(
                target: "audit",
                event_id = %event.id,
                event_type = ?event.event_type,
                user_id = ?event.user_id,
                success = event.success,
                "{}", log_msg
            ),
            AuditLogLevel::High => warn!(
                target: "audit",
                event_id = %event.id,
                event_type = ?event.event_type,
                user_id = ?event.user_id,
                success = event.success,
                "{}", log_msg
            ),
            AuditLogLevel::Medium => info!(
                target: "audit",
                event_id = %event.id,
                event_type = ?event.event_type,
                user_id = ?event.user_id,
                success = event.success,
                "{}", log_msg
            ),
            AuditLogLevel::Info => info!(
                target: "audit",
                event_id = %event.id,
                event_type = ?event.event_type,
                user_id = ?event.user_id,
                success = event.success,
                "{}", log_msg
            ),
            AuditLogLevel::Debug => debug!(
                target: "audit",
                event_id = %event.id,
                event_type = ?event.event_type,
                user_id = ?event.user_id,
                success = event.success,
                "{}", log_msg
            ),
        }
    }

    /// Flush events to storage
    async fn flush_events(&self, events: Vec<AuditEvent>) -> Result<()> {
        if let Err(e) = self.storage.store_events(events).await {
            error!("Failed to store audit events: {}", e);
            return Err(e);
        }
        Ok(())
    }
}

/// Trait for audit storage backends
#[async_trait::async_trait]
pub trait AuditStorage {
    async fn store_events(&self, events: Vec<AuditEvent>) -> Result<()>;
    async fn query_events(&self, query: AuditQuery) -> Result<Vec<AuditEvent>>;
    async fn delete_old_events(&self, older_than: DateTime<Utc>) -> Result<u64>;
}

/// File-based audit storage implementation
pub struct FileAuditStorage {
    file_path: String,
}

impl FileAuditStorage {
    pub fn new(file_path: String) -> Self {
        Self { file_path }
    }
}

#[async_trait::async_trait]
impl AuditStorage for FileAuditStorage {
    async fn store_events(&self, events: Vec<AuditEvent>) -> Result<()> {
        use tokio::fs::OpenOptions;
        use tokio::io::AsyncWriteExt;

        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(&self.file_path).parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| Error::storage(format!("Failed to create audit directory: {}", e)))?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .await
            .map_err(|e| Error::storage(format!("Failed to open audit file: {}", e)))?;

        for event in events {
            let json_line = serde_json::to_string(&event)
                .map_err(|e| Error::serialization(format!("Failed to serialize audit event: {}", e)))?;
            
            file.write_all(format!("{}\n", json_line).as_bytes()).await
                .map_err(|e| Error::storage(format!("Failed to write audit event: {}", e)))?;
        }

        file.flush().await
            .map_err(|e| Error::storage(format!("Failed to flush audit file: {}", e)))?;

        Ok(())
    }

    async fn query_events(&self, _query: AuditQuery) -> Result<Vec<AuditEvent>> {
        // For file storage, we'd need to implement file scanning and filtering
        // This is a simplified implementation
        warn!("File-based audit querying not fully implemented");
        Ok(Vec::new())
    }

    async fn delete_old_events(&self, _older_than: DateTime<Utc>) -> Result<u64> {
        // For file storage, this would require rewriting the file
        warn!("File-based audit cleanup not implemented");
        Ok(0)
    }
}

/// Create appropriate storage backend
async fn create_storage(config: &AuditConfig) -> Result<Arc<dyn AuditStorage + Send + Sync>> {
    match config.storage_backend {
        AuditStorageBackend::File => {
            let file_path = config.storage_config.file_path
                .as_ref()
                .unwrap_or(&"logs/audit.jsonl".to_string())
                .clone();
            Ok(Arc::new(FileAuditStorage::new(file_path)))
        }
        AuditStorageBackend::Database => {
            // TODO: Implement database storage
            warn!("Database audit storage not yet implemented, falling back to file");
            let file_path = "logs/audit.jsonl".to_string();
            Ok(Arc::new(FileAuditStorage::new(file_path)))
        }
        _ => {
            warn!("Unsupported audit storage backend, falling back to file");
            let file_path = "logs/audit.jsonl".to_string();
            Ok(Arc::new(FileAuditStorage::new(file_path)))
        }
    }
}

impl Default for AuditQuery {
    fn default() -> Self {
        Self {
            event_types: None,
            user_id: None,
            resource_id: None,
            success: None,
            start_date: None,
            end_date: None,
            min_severity: None,
            search_text: None,
            limit: Some(100),
            offset: None,
            sort_by: Some(AuditSortBy::Timestamp),
            sort_order: Some(SortOrder::Desc),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_audit_event_creation() {
        let event = AuditEvent::new(
            AuditEventType::UserLogin,
            "user login".to_string(),
            "auth".to_string(),
        )
        .with_user("user123".to_string(), Some("session456".to_string()))
        .with_network("192.168.1.1".to_string(), Some("Mozilla/5.0".to_string()));

        assert_eq!(event.event_type, AuditEventType::UserLogin);
        assert_eq!(event.user_id, Some("user123".to_string()));
        assert_eq!(event.session_id, Some("session456".to_string()));
        assert_eq!(event.ip_address, Some("192.168.1.1".to_string()));
        assert!(event.success);
    }

    #[tokio::test]
    async fn test_file_audit_storage() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_audit.jsonl").to_string_lossy().to_string();
        
        let storage = FileAuditStorage::new(file_path.clone());
        
        let event = AuditEvent::new(
            AuditEventType::UserLogin,
            "test login".to_string(),
            "test".to_string(),
        );
        
        let events = vec![event];
        storage.store_events(events).await.unwrap();
        
        // Verify file was created and contains data
        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert!(content.contains("test login"));
    }

    #[tokio::test]
    async fn test_audit_logger() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_audit.jsonl").to_string_lossy().to_string();
        
        let config = AuditConfig {
            enabled: true,
            storage_backend: AuditStorageBackend::File,
            storage_config: AuditStorageConfig {
                file_path: Some(file_path.clone()),
                ..Default::default()
            },
            batch_size: 1, // Force immediate flush
            ..Default::default()
        };
        
        let logger = AuditLogger::new(config).await.unwrap();
        
        logger.log_auth_event(
            AuditEventType::UserLogin,
            Some("test_user".to_string()),
            Some("127.0.0.1".to_string()),
            true,
            None,
        ).await.unwrap();
        
        // Wait a moment for async write
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert!(content.contains("test_user"));
        assert!(content.contains("UserLogin"));
    }
}
