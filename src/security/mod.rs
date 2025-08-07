pub mod encryption;
pub mod tls;
pub mod audit;
pub mod retention;
pub mod gdpr;

// Re-exports for easier access
pub use encryption::{
    EncryptionService, EncryptionConfig, EncryptionError,
    DataEncryption, FieldEncryption,
};
pub use tls::{TlsConfig, TlsManager, CertificateInfo};
pub use audit::{
    AuditLogger, AuditEvent, AuditEventType, AuditConfig,
    AuditQuery, AuditReport,
};
pub use retention::{
    RetentionPolicy, RetentionManager, RetentionConfig,
    DataClassification, RetentionRule,
};
pub use gdpr::{
    GdprManager, GdprConfig, DataSubjectRequest,
    ConsentManager, DataPortabilityService,
};
