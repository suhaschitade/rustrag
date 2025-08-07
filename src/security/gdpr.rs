use crate::utils::{Error, Result};
use crate::security::audit::{AuditLogger, AuditEventType};
use crate::security::retention::{RetentionService, DataType};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Configuration for GDPR compliance features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdprConfig {
    /// Whether GDPR features are enabled
    pub enabled: bool,
    /// Data controller information
    pub data_controller: DataControllerInfo,
    /// Default data processing lawful bases
    pub default_lawful_bases: Vec<LawfulBasis>,
    /// Data retention periods for different types
    pub data_retention_days: HashMap<DataType, u32>,
    /// Whether to require explicit consent for all processing
    pub require_explicit_consent: bool,
    /// Cookie consent configuration
    pub cookie_config: CookieConsentConfig,
    /// Data subject request handling
    pub request_handling: DataSubjectRequestConfig,
    /// Automated decision making configuration
    pub automated_decisions: AutomatedDecisionConfig,
    /// Data breach notification settings
    pub breach_notification: BreachNotificationConfig,
}

impl Default for GdprConfig {
    fn default() -> Self {
        let mut data_retention_days = HashMap::new();
        data_retention_days.insert(DataType::UserDocument, 2555); // 7 years
        data_retention_days.insert(DataType::UserAccount, 2555);
        data_retention_days.insert(DataType::QueryLog, 365); // 1 year
        data_retention_days.insert(DataType::AuditLog, 2555);
        data_retention_days.insert(DataType::SessionData, 30);

        Self {
            enabled: true,
            data_controller: DataControllerInfo::default(),
            default_lawful_bases: vec![LawfulBasis::LegitimateInterests],
            data_retention_days,
            require_explicit_consent: false,
            cookie_config: CookieConsentConfig::default(),
            request_handling: DataSubjectRequestConfig::default(),
            automated_decisions: AutomatedDecisionConfig::default(),
            breach_notification: BreachNotificationConfig::default(),
        }
    }
}

/// Data controller information as required by GDPR Article 13-14
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataControllerInfo {
    /// Name of the data controller
    pub name: String,
    /// Contact details
    pub contact_email: String,
    pub contact_phone: Option<String>,
    pub contact_address: Option<String>,
    /// Data Protection Officer (DPO) contact if applicable
    pub dpo_contact: Option<String>,
    /// Representative in EU if controller is outside EU
    pub eu_representative: Option<String>,
}

impl Default for DataControllerInfo {
    fn default() -> Self {
        Self {
            name: "RustRAG Service".to_string(),
            contact_email: "privacy@example.com".to_string(),
            contact_phone: None,
            contact_address: None,
            dpo_contact: None,
            eu_representative: None,
        }
    }
}

/// GDPR lawful bases for processing personal data (Article 6)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LawfulBasis {
    /// Consent of the data subject
    Consent,
    /// Processing necessary for performance of a contract
    Contract,
    /// Processing necessary for compliance with legal obligation
    LegalObligation,
    /// Processing necessary to protect vital interests
    VitalInterests,
    /// Processing necessary for performance of task carried out in public interest
    PublicTask,
    /// Processing necessary for legitimate interests pursued by controller
    LegitimateInterests,
}

/// Cookie consent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CookieConsentConfig {
    /// Whether cookie consent banner is enabled
    pub enabled: bool,
    /// Essential cookies that don't require consent
    pub essential_cookies: Vec<String>,
    /// Analytics cookies that require consent
    pub analytics_cookies: Vec<String>,
    /// Marketing cookies that require consent
    pub marketing_cookies: Vec<String>,
    /// Cookie consent expiration in days
    pub consent_expiry_days: u32,
}

impl Default for CookieConsentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            essential_cookies: vec!["session_id".to_string(), "csrf_token".to_string()],
            analytics_cookies: vec!["analytics_id".to_string()],
            marketing_cookies: vec!["marketing_id".to_string()],
            consent_expiry_days: 365,
        }
    }
}

/// Data subject request handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubjectRequestConfig {
    /// Maximum response time in days (GDPR requires 30 days, can be extended to 90)
    pub response_time_days: u32,
    /// Whether to require identity verification
    pub require_identity_verification: bool,
    /// Admin email for request notifications
    pub admin_email: String,
    /// Whether to automatically fulfill some requests
    pub auto_fulfill_access_requests: bool,
}

impl Default for DataSubjectRequestConfig {
    fn default() -> Self {
        Self {
            response_time_days: 30,
            require_identity_verification: true,
            admin_email: "privacy@example.com".to_string(),
            auto_fulfill_access_requests: false,
        }
    }
}

/// Automated decision making configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedDecisionConfig {
    /// Whether automated decisions are made
    pub enabled: bool,
    /// Types of automated decisions
    pub decision_types: Vec<String>,
    /// Whether human review is available
    pub human_review_available: bool,
    /// Explanation mechanism for decisions
    pub explanation_available: bool,
}

impl Default for AutomatedDecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            decision_types: vec!["content_recommendation".to_string()],
            human_review_available: true,
            explanation_available: true,
        }
    }
}

/// Data breach notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachNotificationConfig {
    /// Authority to notify (e.g., national data protection authority)
    pub authority_email: String,
    /// Required notification time in hours (GDPR requires 72 hours)
    pub authority_notification_hours: u32,
    /// Whether to automatically notify data subjects
    pub auto_notify_subjects: bool,
    /// Threshold for notifying subjects (high risk)
    pub subject_notification_threshold: BreachSeverity,
}

impl Default for BreachNotificationConfig {
    fn default() -> Self {
        Self {
            authority_email: "authority@dataprotection.gov".to_string(),
            authority_notification_hours: 72,
            auto_notify_subjects: false,
            subject_notification_threshold: BreachSeverity::High,
        }
    }
}

/// Data breach severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum BreachSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Personal data categories as per GDPR
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PersonalDataCategory {
    /// Basic identifying information
    IdentifyingData,
    /// Contact information
    ContactData,
    /// Technical data (IP addresses, device IDs)
    TechnicalData,
    /// Usage and behavioral data
    UsageData,
    /// Content data
    ContentData,
    /// Special categories (Article 9)
    SpecialCategory,
    /// Criminal convictions data (Article 10)
    CriminalData,
}

/// Processing purpose categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ProcessingPurpose {
    /// Service provision
    ServiceProvision,
    /// User account management
    AccountManagement,
    /// Communication with users
    Communication,
    /// Service improvement and analytics
    Analytics,
    /// Marketing and advertising
    Marketing,
    /// Legal compliance
    LegalCompliance,
    /// Security and fraud prevention
    Security,
    /// Research and development
    Research,
}

/// Data processing record as required by GDPR Article 30
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingRecord {
    /// Unique record ID
    pub id: Uuid,
    /// Name and description of processing activity
    pub activity_name: String,
    pub description: String,
    /// Data controller and processor information
    pub controller: DataControllerInfo,
    pub processor: Option<String>,
    /// Data subjects categories
    pub data_subject_categories: Vec<String>,
    /// Personal data categories
    pub data_categories: Vec<PersonalDataCategory>,
    /// Purposes of processing
    pub purposes: Vec<ProcessingPurpose>,
    /// Lawful basis for processing
    pub lawful_basis: Vec<LawfulBasis>,
    /// Recipients or categories of recipients
    pub recipients: Vec<String>,
    /// Third country transfers
    pub third_country_transfers: Vec<ThirdCountryTransfer>,
    /// Retention periods
    pub retention_periods: HashMap<PersonalDataCategory, u32>,
    /// Technical and organizational measures
    pub security_measures: Vec<String>,
    /// Creation and update timestamps
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl ProcessingRecord {
    pub fn new(activity_name: String, description: String, controller: DataControllerInfo) -> Self {
        Self {
            id: Uuid::new_v4(),
            activity_name,
            description,
            controller,
            processor: None,
            data_subject_categories: Vec::new(),
            data_categories: Vec::new(),
            purposes: Vec::new(),
            lawful_basis: Vec::new(),
            recipients: Vec::new(),
            third_country_transfers: Vec::new(),
            retention_periods: HashMap::new(),
            security_measures: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}

/// Third country transfer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThirdCountryTransfer {
    /// Destination country
    pub country: String,
    /// Transfer mechanism (adequacy decision, SCCs, etc.)
    pub transfer_mechanism: TransferMechanism,
    /// Safeguards in place
    pub safeguards: Vec<String>,
}

/// Transfer mechanisms for third country transfers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferMechanism {
    /// Adequacy decision by European Commission
    AdequacyDecision,
    /// Standard Contractual Clauses
    StandardContractualClauses,
    /// Binding Corporate Rules
    BindingCorporateRules,
    /// Code of conduct or certification
    CodeOfConduct,
    /// Derogations for specific situations
    Derogation(String),
}

/// Consent record for tracking user consent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Unique consent ID
    pub id: Uuid,
    /// User identifier
    pub user_id: String,
    /// What the consent is for
    pub purpose: ProcessingPurpose,
    /// Data categories covered by consent
    pub data_categories: Vec<PersonalDataCategory>,
    /// Whether consent was given
    pub consent_given: bool,
    /// When consent was given/withdrawn
    pub consent_timestamp: DateTime<Utc>,
    /// How consent was collected
    pub consent_method: ConsentMethod,
    /// Consent evidence/proof
    pub evidence: ConsentEvidence,
    /// When consent expires (if applicable)
    pub expires_at: Option<DateTime<Utc>>,
    /// Whether consent has been withdrawn
    pub withdrawn: bool,
    pub withdrawn_at: Option<DateTime<Utc>>,
}

impl ConsentRecord {
    pub fn new(
        user_id: String,
        purpose: ProcessingPurpose,
        data_categories: Vec<PersonalDataCategory>,
        consent_given: bool,
        method: ConsentMethod,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            user_id,
            purpose,
            data_categories,
            consent_given,
            consent_timestamp: Utc::now(),
            consent_method: method,
            evidence: ConsentEvidence::default(),
            expires_at: None,
            withdrawn: false,
            withdrawn_at: None,
        }
    }

    pub fn withdraw(&mut self) {
        self.withdrawn = true;
        self.withdrawn_at = Some(Utc::now());
        self.consent_given = false;
    }

    pub fn is_valid(&self) -> bool {
        if self.withdrawn || !self.consent_given {
            return false;
        }

        if let Some(expires_at) = self.expires_at {
            return Utc::now() <= expires_at;
        }

        true
    }
}

/// Methods for collecting consent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentMethod {
    /// Web form checkbox
    WebForm,
    /// API consent endpoint
    Api,
    /// Email confirmation
    Email,
    /// Phone consent
    Phone,
    /// In-person consent
    InPerson,
    /// Implicit consent (for legitimate interests)
    Implicit,
}

/// Evidence of consent collection
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsentEvidence {
    /// IP address when consent was given
    pub ip_address: Option<String>,
    /// User agent string
    pub user_agent: Option<String>,
    /// Timestamp of consent
    pub timestamp: Option<DateTime<Utc>>,
    /// Form version or API version
    pub version: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Data subject request types as per GDPR Chapter 3
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataSubjectRequestType {
    /// Right of access (Article 15)
    Access,
    /// Right to rectification (Article 16)
    Rectification,
    /// Right to erasure/"right to be forgotten" (Article 17)
    Erasure,
    /// Right to restrict processing (Article 18)
    RestrictionOfProcessing,
    /// Right to data portability (Article 20)
    DataPortability,
    /// Right to object (Article 21)
    ObjectToProcessing,
    /// Rights related to automated decision making (Article 22)
    AutomatedDecisionMaking,
}

/// Data subject request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubjectRequest {
    /// Unique request ID
    pub id: Uuid,
    /// Type of request
    pub request_type: DataSubjectRequestType,
    /// User making the request
    pub user_id: String,
    /// User's email for communication
    pub user_email: String,
    /// Request description
    pub description: String,
    /// Current status
    pub status: RequestStatus,
    /// When request was submitted
    pub submitted_at: DateTime<Utc>,
    /// When request was last updated
    pub updated_at: DateTime<Utc>,
    /// Response deadline (30 days from submission, can be extended)
    pub response_deadline: DateTime<Utc>,
    /// Identity verification status
    pub identity_verified: bool,
    /// Verification method used
    pub verification_method: Option<String>,
    /// Admin handling the request
    pub assigned_to: Option<String>,
    /// Processing notes
    pub notes: Vec<RequestNote>,
    /// Attached files (for evidence, etc.)
    pub attachments: Vec<String>,
    /// Final response to the data subject
    pub response: Option<String>,
    /// Whether request was fulfilled
    pub fulfilled: bool,
    pub fulfilled_at: Option<DateTime<Utc>>,
}

impl DataSubjectRequest {
    pub fn new(
        request_type: DataSubjectRequestType,
        user_id: String,
        user_email: String,
        description: String,
    ) -> Self {
        let submitted_at = Utc::now();
        let response_deadline = submitted_at + chrono::Duration::days(30);

        Self {
            id: Uuid::new_v4(),
            request_type,
            user_id,
            user_email,
            description,
            status: RequestStatus::Submitted,
            submitted_at,
            updated_at: submitted_at,
            response_deadline,
            identity_verified: false,
            verification_method: None,
            assigned_to: None,
            notes: Vec::new(),
            attachments: Vec::new(),
            response: None,
            fulfilled: false,
            fulfilled_at: None,
        }
    }

    pub fn add_note(&mut self, author: String, content: String) {
        self.notes.push(RequestNote {
            id: Uuid::new_v4(),
            author,
            content,
            created_at: Utc::now(),
        });
        self.updated_at = Utc::now();
    }

    pub fn is_overdue(&self) -> bool {
        Utc::now() > self.response_deadline && !self.fulfilled
    }
}

/// Status of a data subject request
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RequestStatus {
    /// Request submitted, awaiting review
    Submitted,
    /// Identity verification required
    VerificationRequired,
    /// Under review by admin
    UnderReview,
    /// Additional information needed from data subject
    MoreInfoRequired,
    /// Request approved and being processed
    Approved,
    /// Request fulfilled
    Fulfilled,
    /// Request rejected with reason
    Rejected,
    /// Request cancelled by data subject
    Cancelled,
}

/// Note attached to a data subject request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestNote {
    pub id: Uuid,
    pub author: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

/// Data breach record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBreachRecord {
    /// Unique breach ID
    pub id: Uuid,
    /// Breach description
    pub description: String,
    /// When breach was discovered
    pub discovered_at: DateTime<Utc>,
    /// When breach occurred (if different from discovery)
    pub occurred_at: Option<DateTime<Utc>>,
    /// Severity assessment
    pub severity: BreachSeverity,
    /// Types of personal data involved
    pub data_categories: Vec<PersonalDataCategory>,
    /// Number of data subjects affected
    pub affected_subjects_count: u32,
    /// Affected data subject details (if known)
    pub affected_subjects: Vec<String>,
    /// Cause of the breach
    pub cause: String,
    /// Current status
    pub status: BreachStatus,
    /// Whether authorities were notified
    pub authority_notified: bool,
    pub authority_notification_date: Option<DateTime<Utc>>,
    /// Whether data subjects were notified
    pub subjects_notified: bool,
    pub subject_notification_date: Option<DateTime<Utc>>,
    /// Mitigation measures taken
    pub mitigation_measures: Vec<String>,
    /// Lessons learned and improvements
    pub lessons_learned: Vec<String>,
    /// Record timestamps
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl DataBreachRecord {
    pub fn new(description: String, severity: BreachSeverity, data_categories: Vec<PersonalDataCategory>) -> Self {
        Self {
            id: Uuid::new_v4(),
            description,
            discovered_at: Utc::now(),
            occurred_at: None,
            severity,
            data_categories,
            affected_subjects_count: 0,
            affected_subjects: Vec::new(),
            cause: String::new(),
            status: BreachStatus::Investigating,
            authority_notified: false,
            authority_notification_date: None,
            subjects_notified: false,
            subject_notification_date: None,
            mitigation_measures: Vec::new(),
            lessons_learned: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn requires_authority_notification(&self) -> bool {
        // GDPR requires notification within 72 hours unless low risk
        matches!(self.severity, BreachSeverity::Medium | BreachSeverity::High | BreachSeverity::Critical)
    }

    pub fn requires_subject_notification(&self) -> bool {
        // High risk breaches require subject notification
        matches!(self.severity, BreachSeverity::High | BreachSeverity::Critical)
    }
}

/// Data breach status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BreachStatus {
    Investigating,
    Contained,
    Resolved,
    Ongoing,
}

/// Main GDPR compliance service
pub struct GdprService {
    config: GdprConfig,
    processing_records: Arc<RwLock<HashMap<Uuid, ProcessingRecord>>>,
    consent_records: Arc<RwLock<HashMap<String, Vec<ConsentRecord>>>>, // user_id -> consents
    data_subject_requests: Arc<RwLock<HashMap<Uuid, DataSubjectRequest>>>,
    data_breaches: Arc<RwLock<HashMap<Uuid, DataBreachRecord>>>,
    audit_logger: Option<Arc<AuditLogger>>,
    retention_service: Option<Arc<RetentionService>>,
}

impl GdprService {
    /// Create a new GDPR service
    pub async fn new(config: GdprConfig) -> Result<Self> {
        Ok(Self {
            config,
            processing_records: Arc::new(RwLock::new(HashMap::new())),
            consent_records: Arc::new(RwLock::new(HashMap::new())),
            data_subject_requests: Arc::new(RwLock::new(HashMap::new())),
            data_breaches: Arc::new(RwLock::new(HashMap::new())),
            audit_logger: None,
            retention_service: None,
        })
    }

    /// Set audit logger for compliance logging
    pub fn set_audit_logger(&mut self, audit_logger: Arc<AuditLogger>) {
        self.audit_logger = Some(audit_logger);
    }

    /// Set retention service for data lifecycle management
    pub fn set_retention_service(&mut self, retention_service: Arc<RetentionService>) {
        self.retention_service = Some(retention_service);
    }

    /// Record consent from a data subject
    pub async fn record_consent(
        &self,
        user_id: String,
        purpose: ProcessingPurpose,
        data_categories: Vec<PersonalDataCategory>,
        consent_given: bool,
        method: ConsentMethod,
        evidence: ConsentEvidence,
    ) -> Result<Uuid> {
        if !self.config.enabled {
            return Err(Error::configuration("GDPR features are disabled"));
        }

        let mut consent = ConsentRecord::new(user_id.clone(), purpose.clone(), data_categories.clone(), consent_given, method);
        consent.evidence = evidence;

        // Set expiry for cookie consent
        if matches!(purpose, ProcessingPurpose::Analytics | ProcessingPurpose::Marketing) {
            consent.expires_at = Some(Utc::now() + chrono::Duration::days(self.config.cookie_config.consent_expiry_days as i64));
        }

        let consent_id = consent.id;

        // Store consent
        {
            let mut consents = self.consent_records.write().await;
            consents.entry(user_id.clone()).or_default().push(consent);
        }

        // Log consent event
        if let Some(audit_logger) = &self.audit_logger {
            let event_type = if consent_given {
                AuditEventType::ConsentGranted
            } else {
                AuditEventType::ConsentRevoked
            };

            audit_logger.log_event(
                crate::security::audit::AuditEvent::new(
                    event_type,
                    format!("Consent {} for purpose: {:?}", if consent_given { "granted" } else { "revoked" }, purpose),
                    "gdpr".to_string()
                )
                .with_user(user_id, None)
                .with_detail("purpose".to_string(), serde_json::json!(purpose))
                .with_detail("data_categories".to_string(), serde_json::json!(data_categories))
            ).await.map_err(|e| Error::internal(format!("Failed to log consent event: {}", e)))?;
        }

        info!("Recorded consent: user={}, purpose={:?}, given={}", user_id, purpose, consent_given);
        Ok(consent_id)
    }

    /// Withdraw consent
    pub async fn withdraw_consent(&self, user_id: &str, consent_id: &Uuid) -> Result<()> {
        let mut consents = self.consent_records.write().await;
        
        if let Some(user_consents) = consents.get_mut(user_id) {
            if let Some(consent) = user_consents.iter_mut().find(|c| c.id == *consent_id) {
                consent.withdraw();

                // Log withdrawal
                if let Some(audit_logger) = &self.audit_logger {
                    audit_logger.log_event(
                        crate::security::audit::AuditEvent::new(
                            AuditEventType::ConsentRevoked,
                            "Consent withdrawn".to_string(),
                            "gdpr".to_string()
                        )
                        .with_user(user_id.to_string(), None)
                        .with_detail("consent_id".to_string(), serde_json::json!(consent_id))
                        .with_detail("purpose".to_string(), serde_json::json!(consent.purpose))
                    ).await.map_err(|e| Error::internal(format!("Failed to log consent withdrawal: {}", e)))?;
                }

                info!("Consent withdrawn: user={}, consent_id={}", user_id, consent_id);
                return Ok(());
            }
        }

        Err(Error::not_found("Consent record not found"))
    }

    /// Check if user has given valid consent for a specific purpose
    pub async fn has_valid_consent(&self, user_id: &str, purpose: &ProcessingPurpose) -> bool {
        let consents = self.consent_records.read().await;
        
        if let Some(user_consents) = consents.get(user_id) {
            return user_consents.iter().any(|consent| {
                consent.purpose == *purpose && consent.is_valid()
            });
        }

        false
    }

    /// Submit a data subject request
    pub async fn submit_data_subject_request(
        &self,
        request_type: DataSubjectRequestType,
        user_id: String,
        user_email: String,
        description: String,
    ) -> Result<Uuid> {
        if !self.config.enabled {
            return Err(Error::configuration("GDPR features are disabled"));
        }

        let request = DataSubjectRequest::new(request_type, user_id.clone(), user_email, description);
        let request_id = request.id;

        // Store request
        {
            let mut requests = self.data_subject_requests.write().await;
            requests.insert(request_id, request);
        }

        // Log request submission
        if let Some(audit_logger) = &self.audit_logger {
            audit_logger.log_event(
                crate::security::audit::AuditEvent::new(
                    AuditEventType::DataSubjectRequest,
                    format!("Data subject request submitted: {:?}", request_type),
                    "gdpr".to_string()
                )
                .with_user(user_id, None)
                .with_detail("request_type".to_string(), serde_json::json!(request_type))
                .with_detail("request_id".to_string(), serde_json::json!(request_id))
            ).await.map_err(|e| Error::internal(format!("Failed to log request submission: {}", e)))?;
        }

        info!("Data subject request submitted: id={}, type={:?}, user={}", request_id, request_type, user_id);
        Ok(request_id)
    }

    /// Process data subject access request (Article 15)
    pub async fn process_access_request(&self, request_id: &Uuid) -> Result<HashMap<String, serde_json::Value>> {
        let request = {
            let requests = self.data_subject_requests.read().await;
            requests.get(request_id).cloned()
        };

        if let Some(request) = request {
            if request.request_type != DataSubjectRequestType::Access {
                return Err(Error::validation("Not an access request"));
            }

            if !request.identity_verified && self.config.request_handling.require_identity_verification {
                return Err(Error::validation("Identity verification required"));
            }

            let mut data = HashMap::new();

            // Collect user data across systems
            data.insert("user_id".to_string(), serde_json::json!(request.user_id));
            data.insert("request_id".to_string(), serde_json::json!(request_id));
            
            // Personal data categories
            let consents = self.consent_records.read().await;
            if let Some(user_consents) = consents.get(&request.user_id) {
                data.insert("consents".to_string(), serde_json::json!(user_consents));
            }

            // Processing activities
            let processing_records = self.processing_records.read().await;
            let relevant_processing: Vec<_> = processing_records.values()
                .filter(|record| {
                    // Filter processing records that might involve this user
                    record.purposes.contains(&ProcessingPurpose::ServiceProvision) ||
                    record.purposes.contains(&ProcessingPurpose::AccountManagement)
                })
                .collect();
            data.insert("processing_activities".to_string(), serde_json::json!(relevant_processing));

            // Data retention information
            if let Some(retention_service) = &self.retention_service {
                // TODO: Get user's retention records
                data.insert("retention_info".to_string(), serde_json::json!("Available upon request"));
            }

            // Controller information
            data.insert("data_controller".to_string(), serde_json::json!(self.config.data_controller));

            info!("Access request processed: request_id={}", request_id);
            Ok(data)
        } else {
            Err(Error::not_found("Request not found"))
        }
    }

    /// Process erasure request (Article 17 - Right to be forgotten)
    pub async fn process_erasure_request(&self, request_id: &Uuid) -> Result<()> {
        let request = {
            let requests = self.data_subject_requests.read().await;
            requests.get(request_id).cloned()
        };

        if let Some(request) = request {
            if request.request_type != DataSubjectRequestType::Erasure {
                return Err(Error::validation("Not an erasure request"));
            }

            if !request.identity_verified && self.config.request_handling.require_identity_verification {
                return Err(Error::validation("Identity verification required"));
            }

            // Remove consent records
            {
                let mut consents = self.consent_records.write().await;
                consents.remove(&request.user_id);
            }

            // Schedule data for deletion through retention service
            if let Some(retention_service) = &self.retention_service {
                // TODO: Mark user data for deletion
                warn!("Erasure request processing not fully implemented for retention service");
            }

            // Log erasure
            if let Some(audit_logger) = &self.audit_logger {
                audit_logger.log_event(
                    crate::security::audit::AuditEvent::new(
                        AuditEventType::DataPurged,
                        "Data erased per subject request".to_string(),
                        "gdpr".to_string()
                    )
                    .with_user(request.user_id.clone(), None)
                    .with_detail("request_id".to_string(), serde_json::json!(request_id))
                ).await.map_err(|e| Error::internal(format!("Failed to log erasure: {}", e)))?;
            }

            info!("Erasure request processed: request_id={}, user={}", request_id, request.user_id);
            Ok(())
        } else {
            Err(Error::not_found("Request not found"))
        }
    }

    /// Record a data breach
    pub async fn record_data_breach(
        &self,
        description: String,
        severity: BreachSeverity,
        data_categories: Vec<PersonalDataCategory>,
        affected_subjects_count: u32,
    ) -> Result<Uuid> {
        if !self.config.enabled {
            return Err(Error::configuration("GDPR features are disabled"));
        }

        let mut breach = DataBreachRecord::new(description, severity, data_categories);
        breach.affected_subjects_count = affected_subjects_count;
        
        let breach_id = breach.id;

        // Store breach record
        {
            let mut breaches = self.data_breaches.write().await;
            breaches.insert(breach_id, breach.clone());
        }

        // Log security violation
        if let Some(audit_logger) = &self.audit_logger {
            audit_logger.log_security_violation(
                None,
                "data_breach".to_string(),
                format!("Data breach recorded: {} (severity: {:?})", description, severity),
                None,
            ).await.map_err(|e| Error::internal(format!("Failed to log breach: {}", e)))?;
        }

        // Check if authority notification is required
        if breach.requires_authority_notification() {
            info!(
                "Data breach requires authority notification within {} hours: breach_id={}",
                self.config.breach_notification.authority_notification_hours,
                breach_id
            );
        }

        // Check if subject notification is required
        if breach.requires_subject_notification() {
            info!(
                "Data breach requires subject notification: breach_id={}, affected_count={}",
                breach_id,
                affected_subjects_count
            );
        }

        info!("Data breach recorded: id={}, severity={:?}", breach_id, severity);
        Ok(breach_id)
    }

    /// Add processing record (Article 30 - Records of processing)
    pub async fn add_processing_record(&self, record: ProcessingRecord) -> Result<Uuid> {
        if !self.config.enabled {
            return Err(Error::configuration("GDPR features are disabled"));
        }

        let record_id = record.id;

        {
            let mut records = self.processing_records.write().await;
            records.insert(record_id, record);
        }

        info!("Processing record added: id={}", record_id);
        Ok(record_id)
    }

    /// Generate GDPR compliance report
    pub async fn generate_compliance_report(&self) -> Result<GdprComplianceReport> {
        let processing_records = self.processing_records.read().await;
        let consents = self.consent_records.read().await;
        let requests = self.data_subject_requests.read().await;
        let breaches = self.data_breaches.read().await;

        let report = GdprComplianceReport {
            generated_at: Utc::now(),
            processing_activities_count: processing_records.len() as u64,
            active_consents_count: consents.values()
                .flatten()
                .filter(|c| c.is_valid())
                .count() as u64,
            pending_requests_count: requests.values()
                .filter(|r| matches!(r.status, RequestStatus::Submitted | RequestStatus::UnderReview))
                .count() as u64,
            overdue_requests_count: requests.values()
                .filter(|r| r.is_overdue())
                .count() as u64,
            data_breaches_count: breaches.len() as u64,
            high_risk_breaches_count: breaches.values()
                .filter(|b| matches!(b.severity, BreachSeverity::High | BreachSeverity::Critical))
                .count() as u64,
            compliance_status: self.assess_compliance_status(&*processing_records, &*requests, &*breaches),
        };

        Ok(report)
    }

    /// Assess overall compliance status
    fn assess_compliance_status(
        &self,
        processing_records: &HashMap<Uuid, ProcessingRecord>,
        requests: &HashMap<Uuid, DataSubjectRequest>,
        breaches: &HashMap<Uuid, DataBreachRecord>,
    ) -> ComplianceStatus {
        let mut issues = Vec::new();

        // Check for overdue requests
        let overdue_count = requests.values().filter(|r| r.is_overdue()).count();
        if overdue_count > 0 {
            issues.push(format!("{} overdue data subject requests", overdue_count));
        }

        // Check for unnotified breaches
        let unnotified_breaches: Vec<_> = breaches.values()
            .filter(|b| b.requires_authority_notification() && !b.authority_notified)
            .collect();
        if !unnotified_breaches.is_empty() {
            issues.push(format!("{} data breaches require authority notification", unnotified_breaches.len()));
        }

        // Check for missing lawful basis
        let missing_basis = processing_records.values()
            .filter(|r| r.lawful_basis.is_empty())
            .count();
        if missing_basis > 0 {
            issues.push(format!("{} processing activities lack lawful basis", missing_basis));
        }

        if issues.is_empty() {
            ComplianceStatus::Compliant
        } else if issues.len() <= 2 {
            ComplianceStatus::MinorIssues(issues)
        } else {
            ComplianceStatus::MajorIssues(issues)
        }
    }
}

/// GDPR compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdprComplianceReport {
    pub generated_at: DateTime<Utc>,
    pub processing_activities_count: u64,
    pub active_consents_count: u64,
    pub pending_requests_count: u64,
    pub overdue_requests_count: u64,
    pub data_breaches_count: u64,
    pub high_risk_breaches_count: u64,
    pub compliance_status: ComplianceStatus,
}

/// Overall compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    Compliant,
    MinorIssues(Vec<String>),
    MajorIssues(Vec<String>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consent_recording() {
        let config = GdprConfig::default();
        let service = GdprService::new(config).await.unwrap();

        let consent_id = service.record_consent(
            "user123".to_string(),
            ProcessingPurpose::Analytics,
            vec![PersonalDataCategory::UsageData],
            true,
            ConsentMethod::WebForm,
            ConsentEvidence::default(),
        ).await.unwrap();

        assert!(!consent_id.is_nil());

        // Check consent is valid
        assert!(service.has_valid_consent("user123", &ProcessingPurpose::Analytics).await);
        assert!(!service.has_valid_consent("user123", &ProcessingPurpose::Marketing).await);
    }

    #[tokio::test]
    async fn test_consent_withdrawal() {
        let config = GdprConfig::default();
        let service = GdprService::new(config).await.unwrap();

        let consent_id = service.record_consent(
            "user123".to_string(),
            ProcessingPurpose::Analytics,
            vec![PersonalDataCategory::UsageData],
            true,
            ConsentMethod::WebForm,
            ConsentEvidence::default(),
        ).await.unwrap();

        // Initially consent is valid
        assert!(service.has_valid_consent("user123", &ProcessingPurpose::Analytics).await);

        // Withdraw consent
        service.withdraw_consent("user123", &consent_id).await.unwrap();

        // Now consent should be invalid
        assert!(!service.has_valid_consent("user123", &ProcessingPurpose::Analytics).await);
    }

    #[tokio::test]
    async fn test_data_subject_request() {
        let config = GdprConfig::default();
        let service = GdprService::new(config).await.unwrap();

        let request_id = service.submit_data_subject_request(
            DataSubjectRequestType::Access,
            "user123".to_string(),
            "user@example.com".to_string(),
            "I want to see my data".to_string(),
        ).await.unwrap();

        assert!(!request_id.is_nil());

        // Check request exists
        let requests = service.data_subject_requests.read().await;
        let request = requests.get(&request_id).unwrap();
        assert_eq!(request.request_type, DataSubjectRequestType::Access);
        assert_eq!(request.status, RequestStatus::Submitted);
    }

    #[tokio::test]
    async fn test_data_breach_recording() {
        let config = GdprConfig::default();
        let service = GdprService::new(config).await.unwrap();

        let breach_id = service.record_data_breach(
            "Unauthorized access to user data".to_string(),
            BreachSeverity::High,
            vec![PersonalDataCategory::ContactData],
            100,
        ).await.unwrap();

        assert!(!breach_id.is_nil());

        // Check breach exists
        let breaches = service.data_breaches.read().await;
        let breach = breaches.get(&breach_id).unwrap();
        assert_eq!(breach.severity, BreachSeverity::High);
        assert!(breach.requires_authority_notification());
        assert!(breach.requires_subject_notification());
    }

    #[test]
    fn test_processing_record_creation() {
        let controller = DataControllerInfo::default();
        let record = ProcessingRecord::new(
            "User account management".to_string(),
            "Managing user accounts and authentication".to_string(),
            controller,
        );

        assert_eq!(record.activity_name, "User account management");
        assert!(!record.id.is_nil());
    }

    #[test]
    fn test_lawful_basis_serialization() {
        let basis = LawfulBasis::Consent;
        let json = serde_json::to_string(&basis).unwrap();
        let deserialized: LawfulBasis = serde_json::from_str(&json).unwrap();
        assert_eq!(basis, deserialized);
    }
}
