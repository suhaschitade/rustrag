use crate::utils::{Error, Result};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Configuration for data retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    /// Whether retention policies are enabled
    pub enabled: bool,
    /// Default retention period in days (if not specified per data type)
    pub default_retention_days: u32,
    /// Check interval for retention cleanup in hours
    pub cleanup_interval_hours: u64,
    /// Batch size for processing deletions
    pub batch_size: usize,
    /// Whether to perform soft delete (mark as deleted) or hard delete
    pub soft_delete: bool,
    /// Grace period before hard deletion after soft delete (in days)
    pub grace_period_days: u32,
    /// Data type specific retention policies
    pub data_type_policies: HashMap<DataType, DataRetentionPolicy>,
    /// Legal hold settings
    pub legal_hold_config: LegalHoldConfig,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        let mut data_type_policies = HashMap::new();
        
        // Set default policies for different data types
        data_type_policies.insert(DataType::UserDocument, DataRetentionPolicy {
            retention_days: 2555, // ~7 years for user documents
            auto_delete: true,
            require_approval: false,
            backup_before_delete: true,
        });
        
        data_type_policies.insert(DataType::QueryLog, DataRetentionPolicy {
            retention_days: 365, // 1 year for query logs
            auto_delete: true,
            require_approval: false,
            backup_before_delete: false,
        });
        
        data_type_policies.insert(DataType::AuditLog, DataRetentionPolicy {
            retention_days: 2555, // ~7 years for audit logs (compliance)
            auto_delete: false, // Don't auto-delete audit logs
            require_approval: true,
            backup_before_delete: true,
        });
        
        data_type_policies.insert(DataType::UserAccount, DataRetentionPolicy {
            retention_days: 90, // 3 months after account deletion
            auto_delete: false, // Manual approval required
            require_approval: true,
            backup_before_delete: true,
        });
        
        data_type_policies.insert(DataType::SystemLog, DataRetentionPolicy {
            retention_days: 90, // 3 months for system logs
            auto_delete: true,
            require_approval: false,
            backup_before_delete: false,
        });

        Self {
            enabled: true,
            default_retention_days: 365,
            cleanup_interval_hours: 24,
            batch_size: 100,
            soft_delete: true,
            grace_period_days: 30,
            data_type_policies,
            legal_hold_config: LegalHoldConfig::default(),
        }
    }
}

/// Legal hold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalHoldConfig {
    /// Whether legal holds are enabled
    pub enabled: bool,
    /// Approval workflow for legal holds
    pub require_approval: bool,
    /// Notification settings
    pub notification_config: NotificationConfig,
}

impl Default for LegalHoldConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            require_approval: true,
            notification_config: NotificationConfig::default(),
        }
    }
}

/// Notification configuration for retention events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Email notifications enabled
    pub email_enabled: bool,
    /// Email recipients for notifications
    pub email_recipients: Vec<String>,
    /// Webhook URL for notifications
    pub webhook_url: Option<String>,
    /// Days before deletion to send warning
    pub warning_days_before: u32,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            email_enabled: false,
            email_recipients: Vec::new(),
            webhook_url: None,
            warning_days_before: 7,
        }
    }
}

/// Types of data that can have retention policies
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    UserDocument,
    QueryLog,
    AuditLog,
    UserAccount,
    SystemLog,
    EmbeddingVector,
    SearchIndex,
    CachedResult,
    SessionData,
    BackupData,
}

/// Retention policy for a specific data type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionPolicy {
    /// Retention period in days
    pub retention_days: u32,
    /// Whether to automatically delete when policy expires
    pub auto_delete: bool,
    /// Whether manual approval is required before deletion
    pub require_approval: bool,
    /// Whether to create backup before deletion
    pub backup_before_delete: bool,
}

/// Retention record for tracking data lifecycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRecord {
    /// Unique record ID
    pub id: Uuid,
    /// Type of data
    pub data_type: DataType,
    /// Identifier of the data item
    pub data_id: String,
    /// When the data was created
    pub created_at: DateTime<Utc>,
    /// When the data should be deleted according to policy
    pub expires_at: DateTime<Utc>,
    /// Current status of the retention record
    pub status: RetentionStatus,
    /// When the record was last updated
    pub updated_at: DateTime<Utc>,
    /// Size of data in bytes (if known)
    pub data_size: Option<u64>,
    /// Related user ID (if applicable)
    pub user_id: Option<String>,
    /// Any legal holds preventing deletion
    pub legal_holds: Vec<LegalHold>,
    /// Metadata about the data
    pub metadata: HashMap<String, serde_json::Value>,
    /// When deletion was requested (for approval workflow)
    pub deletion_requested_at: Option<DateTime<Utc>>,
    /// Who requested the deletion
    pub deletion_requested_by: Option<String>,
    /// When deletion was approved
    pub deletion_approved_at: Option<DateTime<Utc>>,
    /// Who approved the deletion
    pub deletion_approved_by: Option<String>,
    /// Backup location before deletion
    pub backup_location: Option<String>,
}

impl RetentionRecord {
    /// Create a new retention record
    pub fn new(
        data_type: DataType,
        data_id: String,
        created_at: DateTime<Utc>,
        retention_days: u32,
    ) -> Self {
        let expires_at = created_at + ChronoDuration::days(retention_days as i64);
        
        Self {
            id: Uuid::new_v4(),
            data_type,
            data_id,
            created_at,
            expires_at,
            status: RetentionStatus::Active,
            updated_at: Utc::now(),
            data_size: None,
            user_id: None,
            legal_holds: Vec::new(),
            metadata: HashMap::new(),
            deletion_requested_at: None,
            deletion_requested_by: None,
            deletion_approved_at: None,
            deletion_approved_by: None,
            backup_location: None,
        }
    }

    /// Check if this record is expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at && self.legal_holds.is_empty()
    }

    /// Check if this record has active legal holds
    pub fn has_legal_hold(&self) -> bool {
        self.legal_holds.iter().any(|hold| hold.is_active())
    }

    /// Add a legal hold
    pub fn add_legal_hold(&mut self, legal_hold: LegalHold) {
        self.legal_holds.push(legal_hold);
        self.updated_at = Utc::now();
    }

    /// Remove a legal hold by ID
    pub fn remove_legal_hold(&mut self, hold_id: &Uuid) {
        self.legal_holds.retain(|hold| hold.id != *hold_id);
        self.updated_at = Utc::now();
    }

    /// Request deletion
    pub fn request_deletion(&mut self, requested_by: String) {
        self.status = RetentionStatus::DeletionRequested;
        self.deletion_requested_at = Some(Utc::now());
        self.deletion_requested_by = Some(requested_by);
        self.updated_at = Utc::now();
    }

    /// Approve deletion
    pub fn approve_deletion(&mut self, approved_by: String) {
        self.status = RetentionStatus::DeletionApproved;
        self.deletion_approved_at = Some(Utc::now());
        self.deletion_approved_by = Some(approved_by);
        self.updated_at = Utc::now();
    }

    /// Mark as deleted
    pub fn mark_deleted(&mut self, backup_location: Option<String>) {
        self.status = RetentionStatus::Deleted;
        self.backup_location = backup_location;
        self.updated_at = Utc::now();
    }
}

/// Status of a retention record
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetentionStatus {
    /// Data is active and within retention period
    Active,
    /// Data is expired but not yet processed
    Expired,
    /// Deletion has been requested and waiting for approval
    DeletionRequested,
    /// Deletion has been approved
    DeletionApproved,
    /// Data has been soft deleted
    SoftDeleted,
    /// Data has been hard deleted
    Deleted,
    /// Data is on legal hold
    LegalHold,
}

/// Legal hold information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalHold {
    /// Unique hold ID
    pub id: Uuid,
    /// Description of the legal hold
    pub description: String,
    /// When the hold was created
    pub created_at: DateTime<Utc>,
    /// When the hold expires (if applicable)
    pub expires_at: Option<DateTime<Utc>>,
    /// Who created the hold
    pub created_by: String,
    /// Case or matter reference
    pub case_reference: Option<String>,
    /// Whether the hold is currently active
    pub active: bool,
}

impl LegalHold {
    /// Create a new legal hold
    pub fn new(description: String, created_by: String, case_reference: Option<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            description,
            created_at: Utc::now(),
            expires_at: None,
            created_by,
            case_reference,
            active: true,
        }
    }

    /// Check if the legal hold is active
    pub fn is_active(&self) -> bool {
        if !self.active {
            return false;
        }

        if let Some(expires_at) = self.expires_at {
            return Utc::now() <= expires_at;
        }

        true
    }

    /// Deactivate the legal hold
    pub fn deactivate(&mut self) {
        self.active = false;
    }
}

/// Query parameters for retention records
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetentionQuery {
    /// Filter by data type
    pub data_type: Option<DataType>,
    /// Filter by status
    pub status: Option<RetentionStatus>,
    /// Filter by user ID
    pub user_id: Option<String>,
    /// Filter expired records
    pub expired_only: bool,
    /// Filter records with legal holds
    pub legal_hold_only: bool,
    /// Date range filters
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
    pub expires_after: Option<DateTime<Utc>>,
    pub expires_before: Option<DateTime<Utc>>,
    /// Pagination
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

/// Retention statistics and reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionReport {
    /// When the report was generated
    pub generated_at: DateTime<Utc>,
    /// Total number of retention records
    pub total_records: u64,
    /// Records by status
    pub records_by_status: HashMap<RetentionStatus, u64>,
    /// Records by data type
    pub records_by_type: HashMap<DataType, u64>,
    /// Total data size under management
    pub total_data_size_bytes: u64,
    /// Data size by type
    pub data_size_by_type: HashMap<DataType, u64>,
    /// Records expiring soon (within warning period)
    pub expiring_soon: u64,
    /// Records with legal holds
    pub legal_hold_count: u64,
    /// Records pending approval
    pub pending_approval: u64,
    /// Storage space that could be freed
    pub potential_space_savings: u64,
}

/// Main retention policy service
pub struct RetentionService {
    config: RetentionConfig,
    records: Arc<RwLock<HashMap<String, RetentionRecord>>>,
    storage: Arc<dyn RetentionStorage + Send + Sync>,
}

impl RetentionService {
    /// Create a new retention service
    pub async fn new(config: RetentionConfig) -> Result<Self> {
        let storage = create_storage(&config).await?;
        
        Ok(Self {
            config,
            records: Arc::new(RwLock::new(HashMap::new())),
            storage,
        })
    }

    /// Add data to retention tracking
    pub async fn track_data(
        &self,
        data_type: DataType,
        data_id: String,
        created_at: Option<DateTime<Utc>>,
        user_id: Option<String>,
        data_size: Option<u64>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Uuid> {
        if !self.config.enabled {
            return Err(Error::configuration("Retention policies are disabled"));
        }

        let policy = self.get_policy_for_type(data_type);
        let created = created_at.unwrap_or_else(Utc::now);
        
        let mut record = RetentionRecord::new(
            data_type,
            data_id.clone(),
            created,
            policy.retention_days,
        );

        record.user_id = user_id;
        record.data_size = data_size;
        
        if let Some(meta) = metadata {
            record.metadata = meta;
        }

        let record_id = record.id;

        // Store in memory cache
        {
            let mut records = self.records.write().await;
            records.insert(data_id.clone(), record.clone());
        }

        // Persist to storage
        self.storage.store_record(record).await?;

        info!(
            "Tracking data for retention: type={:?}, id={}, expires_at={}",
            data_type, data_id, record.expires_at
        );

        Ok(record_id)
    }

    /// Update retention record
    pub async fn update_record(&self, data_id: &str, updates: RetentionRecordUpdate) -> Result<()> {
        let mut records = self.records.write().await;
        
        if let Some(record) = records.get_mut(data_id) {
            if let Some(expires_at) = updates.expires_at {
                record.expires_at = expires_at;
            }
            
            if let Some(status) = updates.status {
                record.status = status;
            }
            
            if let Some(data_size) = updates.data_size {
                record.data_size = Some(data_size);
            }
            
            record.updated_at = Utc::now();

            // Persist changes
            self.storage.store_record(record.clone()).await?;
            
            info!("Updated retention record for data_id: {}", data_id);
        } else {
            return Err(Error::not_found(format!("Retention record not found for data_id: {}", data_id)));
        }

        Ok(())
    }

    /// Add legal hold to data
    pub async fn add_legal_hold(
        &self,
        data_id: &str,
        description: String,
        created_by: String,
        case_reference: Option<String>,
    ) -> Result<Uuid> {
        let legal_hold = LegalHold::new(description, created_by, case_reference);
        let hold_id = legal_hold.id;

        let mut records = self.records.write().await;
        
        if let Some(record) = records.get_mut(data_id) {
            record.add_legal_hold(legal_hold);
            record.status = RetentionStatus::LegalHold;

            // Persist changes
            self.storage.store_record(record.clone()).await?;

            info!("Added legal hold {} to data_id: {}", hold_id, data_id);
        } else {
            return Err(Error::not_found(format!("Retention record not found for data_id: {}", data_id)));
        }

        Ok(hold_id)
    }

    /// Remove legal hold from data
    pub async fn remove_legal_hold(&self, data_id: &str, hold_id: &Uuid) -> Result<()> {
        let mut records = self.records.write().await;
        
        if let Some(record) = records.get_mut(data_id) {
            record.remove_legal_hold(hold_id);
            
            // Update status if no more legal holds
            if !record.has_legal_hold() {
                record.status = if record.is_expired() {
                    RetentionStatus::Expired
                } else {
                    RetentionStatus::Active
                };
            }

            // Persist changes
            self.storage.store_record(record.clone()).await?;

            info!("Removed legal hold {} from data_id: {}", hold_id, data_id);
        } else {
            return Err(Error::not_found(format!("Retention record not found for data_id: {}", data_id)));
        }

        Ok(())
    }

    /// Request deletion for data (for manual approval workflow)
    pub async fn request_deletion(&self, data_id: &str, requested_by: String) -> Result<()> {
        let mut records = self.records.write().await;
        
        if let Some(record) = records.get_mut(data_id) {
            if record.has_legal_hold() {
                return Err(Error::validation("Cannot request deletion for data under legal hold"));
            }

            record.request_deletion(requested_by.clone());

            // Persist changes
            self.storage.store_record(record.clone()).await?;

            info!("Deletion requested for data_id: {} by {}", data_id, requested_by);
        } else {
            return Err(Error::not_found(format!("Retention record not found for data_id: {}", data_id)));
        }

        Ok(())
    }

    /// Approve deletion for data
    pub async fn approve_deletion(&self, data_id: &str, approved_by: String) -> Result<()> {
        let mut records = self.records.write().await;
        
        if let Some(record) = records.get_mut(data_id) {
            if record.status != RetentionStatus::DeletionRequested {
                return Err(Error::validation("Deletion must be requested before approval"));
            }

            record.approve_deletion(approved_by.clone());

            // Persist changes
            self.storage.store_record(record.clone()).await?;

            info!("Deletion approved for data_id: {} by {}", data_id, approved_by);
        } else {
            return Err(Error::not_found(format!("Retention record not found for data_id: {}", data_id)));
        }

        Ok(())
    }

    /// Process expired data according to retention policies
    pub async fn process_expired_data(&self) -> Result<ProcessingReport> {
        if !self.config.enabled {
            return Ok(ProcessingReport::default());
        }

        info!("Starting retention policy processing");

        let query = RetentionQuery {
            expired_only: true,
            limit: Some(self.config.batch_size as u32),
            ..Default::default()
        };

        let expired_records = self.query_records(query).await?;
        let mut report = ProcessingReport::default();

        for record in expired_records {
            if record.has_legal_hold() {
                debug!("Skipping expired record {} due to legal hold", record.data_id);
                report.skipped_legal_hold += 1;
                continue;
            }

            let policy = self.get_policy_for_type(record.data_type);

            if policy.require_approval && record.status != RetentionStatus::DeletionApproved {
                // Request approval if not already requested
                if record.status != RetentionStatus::DeletionRequested {
                    self.request_deletion(&record.data_id, "system".to_string()).await?;
                    report.deletion_requested += 1;
                }
                continue;
            }

            // Process deletion
            match self.delete_data(&record.data_id, policy.backup_before_delete).await {
                Ok(deleted_size) => {
                    report.deleted_count += 1;
                    report.freed_space_bytes += deleted_size;
                }
                Err(e) => {
                    error!("Failed to delete data {}: {}", record.data_id, e);
                    report.failed_count += 1;
                }
            }
        }

        info!(
            "Retention processing completed: {} deleted, {} requested, {} skipped, {} failed",
            report.deleted_count, report.deletion_requested, report.skipped_legal_hold, report.failed_count
        );

        Ok(report)
    }

    /// Query retention records
    pub async fn query_records(&self, query: RetentionQuery) -> Result<Vec<RetentionRecord>> {
        self.storage.query_records(query).await
    }

    /// Generate retention report
    pub async fn generate_report(&self) -> Result<RetentionReport> {
        let all_records = self.query_records(RetentionQuery::default()).await?;
        
        let mut report = RetentionReport {
            generated_at: Utc::now(),
            total_records: all_records.len() as u64,
            records_by_status: HashMap::new(),
            records_by_type: HashMap::new(),
            total_data_size_bytes: 0,
            data_size_by_type: HashMap::new(),
            expiring_soon: 0,
            legal_hold_count: 0,
            pending_approval: 0,
            potential_space_savings: 0,
        };

        let warning_date = Utc::now() + ChronoDuration::days(self.config.legal_hold_config.notification_config.warning_days_before as i64);

        for record in &all_records {
            // Count by status
            *report.records_by_status.entry(record.status).or_insert(0) += 1;
            
            // Count by type
            *report.records_by_type.entry(record.data_type).or_insert(0) += 1;
            
            // Size calculations
            if let Some(size) = record.data_size {
                report.total_data_size_bytes += size;
                *report.data_size_by_type.entry(record.data_type).or_insert(0) += size;
                
                // Calculate potential savings from expired data
                if record.is_expired() || record.status == RetentionStatus::DeletionApproved {
                    report.potential_space_savings += size;
                }
            }
            
            // Expiring soon
            if record.expires_at <= warning_date && !record.has_legal_hold() {
                report.expiring_soon += 1;
            }
            
            // Legal holds
            if record.has_legal_hold() {
                report.legal_hold_count += 1;
            }
            
            // Pending approval
            if record.status == RetentionStatus::DeletionRequested {
                report.pending_approval += 1;
            }
        }

        Ok(report)
    }

    /// Delete data and update retention record
    async fn delete_data(&self, data_id: &str, backup_before_delete: bool) -> Result<u64> {
        let mut backup_location = None;
        let mut freed_space = 0u64;

        // Get record details
        let record = {
            let records = self.records.read().await;
            records.get(data_id).cloned()
        };

        if let Some(record) = record {
            // Create backup if required
            if backup_before_delete {
                backup_location = Some(self.create_backup(&record).await?);
            }

            // Perform deletion through storage backend
            freed_space = self.storage.delete_data(data_id, record.data_type).await?;

            // Update retention record
            {
                let mut records = self.records.write().await;
                if let Some(mut record) = records.get_mut(data_id) {
                    if self.config.soft_delete {
                        record.status = RetentionStatus::SoftDeleted;
                    } else {
                        record.mark_deleted(backup_location.clone());
                    }

                    // Persist changes
                    self.storage.store_record(record.clone()).await?;
                }
            }

            info!(
                "Deleted data: id={}, type={:?}, size={} bytes, backup={}",
                data_id,
                record.data_type,
                freed_space,
                backup_location.as_deref().unwrap_or("none")
            );
        }

        Ok(freed_space)
    }

    /// Create backup of data before deletion
    async fn create_backup(&self, record: &RetentionRecord) -> Result<String> {
        // TODO: Implement actual backup creation
        // This would depend on the storage backend and backup strategy
        let backup_path = format!("backups/{}/{}", record.data_type as u8, record.data_id);
        warn!("Backup creation not fully implemented: {}", backup_path);
        Ok(backup_path)
    }

    /// Get retention policy for a data type
    fn get_policy_for_type(&self, data_type: DataType) -> DataRetentionPolicy {
        self.config
            .data_type_policies
            .get(&data_type)
            .cloned()
            .unwrap_or_else(|| DataRetentionPolicy {
                retention_days: self.config.default_retention_days,
                auto_delete: true,
                require_approval: false,
                backup_before_delete: false,
            })
    }
}

/// Update parameters for retention records
#[derive(Debug, Clone)]
pub struct RetentionRecordUpdate {
    pub expires_at: Option<DateTime<Utc>>,
    pub status: Option<RetentionStatus>,
    pub data_size: Option<u64>,
}

/// Processing report for retention cleanup
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessingReport {
    /// Number of records successfully deleted
    pub deleted_count: u64,
    /// Number of deletion requests created
    pub deletion_requested: u64,
    /// Number of records skipped due to legal hold
    pub skipped_legal_hold: u64,
    /// Number of failed deletions
    pub failed_count: u64,
    /// Total space freed in bytes
    pub freed_space_bytes: u64,
}

/// Trait for retention storage backends
#[async_trait::async_trait]
pub trait RetentionStorage {
    async fn store_record(&self, record: RetentionRecord) -> Result<()>;
    async fn query_records(&self, query: RetentionQuery) -> Result<Vec<RetentionRecord>>;
    async fn delete_data(&self, data_id: &str, data_type: DataType) -> Result<u64>;
}

/// File-based retention storage implementation
pub struct FileRetentionStorage {
    records_path: String,
}

impl FileRetentionStorage {
    pub fn new(records_path: String) -> Self {
        Self { records_path }
    }
}

#[async_trait::async_trait]
impl RetentionStorage for FileRetentionStorage {
    async fn store_record(&self, record: RetentionRecord) -> Result<()> {
        use tokio::fs::OpenOptions;
        use tokio::io::AsyncWriteExt;

        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(&self.records_path).parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| Error::storage(format!("Failed to create retention directory: {}", e)))?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.records_path)
            .await
            .map_err(|e| Error::storage(format!("Failed to open retention file: {}", e)))?;

        let json_line = serde_json::to_string(&record)
            .map_err(|e| Error::serialization(format!("Failed to serialize retention record: {}", e)))?;
        
        file.write_all(format!("{}\n", json_line).as_bytes()).await
            .map_err(|e| Error::storage(format!("Failed to write retention record: {}", e)))?;

        file.flush().await
            .map_err(|e| Error::storage(format!("Failed to flush retention file: {}", e)))?;

        Ok(())
    }

    async fn query_records(&self, _query: RetentionQuery) -> Result<Vec<RetentionRecord>> {
        // For file storage, we'd need to implement file scanning and filtering
        // This is a simplified implementation
        warn!("File-based retention querying not fully implemented");
        Ok(Vec::new())
    }

    async fn delete_data(&self, data_id: &str, data_type: DataType) -> Result<u64> {
        // This would depend on the actual data storage backend
        warn!("Data deletion not fully implemented for data_id: {}, type: {:?}", data_id, data_type);
        Ok(0)
    }
}

/// Create appropriate storage backend
async fn create_storage(_config: &RetentionConfig) -> Result<Arc<dyn RetentionStorage + Send + Sync>> {
    // For now, use file storage
    let records_path = "logs/retention_records.jsonl".to_string();
    Ok(Arc::new(FileRetentionStorage::new(records_path)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_retention_record_creation() {
        let record = RetentionRecord::new(
            DataType::UserDocument,
            "doc123".to_string(),
            Utc::now() - ChronoDuration::days(30),
            365,
        );

        assert_eq!(record.data_type, DataType::UserDocument);
        assert_eq!(record.data_id, "doc123");
        assert_eq!(record.status, RetentionStatus::Active);
        assert!(!record.is_expired());
    }

    #[tokio::test]
    async fn test_legal_hold() {
        let mut record = RetentionRecord::new(
            DataType::UserDocument,
            "doc123".to_string(),
            Utc::now() - ChronoDuration::days(400), // Expired
            365,
        );

        // Initially expired
        assert!(record.is_expired());

        // Add legal hold
        let legal_hold = LegalHold::new(
            "Litigation hold".to_string(),
            "legal@company.com".to_string(),
            Some("CASE-2024-001".to_string()),
        );
        record.add_legal_hold(legal_hold);

        // Now should not be considered expired due to legal hold
        assert!(!record.is_expired());
        assert!(record.has_legal_hold());
    }

    #[tokio::test]
    async fn test_retention_config_defaults() {
        let config = RetentionConfig::default();
        
        assert!(config.enabled);
        assert_eq!(config.default_retention_days, 365);
        assert!(config.soft_delete);
        
        // Check that audit logs have longer retention
        let audit_policy = config.data_type_policies.get(&DataType::AuditLog).unwrap();
        assert_eq!(audit_policy.retention_days, 2555); // ~7 years
        assert!(!audit_policy.auto_delete); // Require manual approval
    }

    #[test]
    fn test_data_type_serialization() {
        let data_type = DataType::UserDocument;
        let json = serde_json::to_string(&data_type).unwrap();
        let deserialized: DataType = serde_json::from_str(&json).unwrap();
        assert_eq!(data_type, deserialized);
    }
}
