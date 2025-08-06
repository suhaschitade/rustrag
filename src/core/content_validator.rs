use crate::utils::Result;
use regex::Regex;
use std::collections::HashSet;

/// Configuration for content validation and sanitization
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum allowed document size in bytes
    pub max_document_size: usize,
    /// Minimum allowed document size in bytes
    pub min_document_size: usize,
    /// Maximum allowed line length
    pub max_line_length: usize,
    /// Whether to allow empty documents
    pub allow_empty_content: bool,
    /// Whether to remove sensitive data patterns
    pub sanitize_sensitive_data: bool,
    /// Whether to normalize whitespace
    pub normalize_whitespace: bool,
    /// Whether to remove control characters
    pub remove_control_chars: bool,
    /// Whether to validate encoding
    pub validate_encoding: bool,
    /// Allowed file types for validation
    pub allowed_file_types: HashSet<String>,
    /// Blocked content patterns (regex)
    pub blocked_patterns: Vec<String>,
    /// Maximum word repetition threshold
    pub max_word_repetition: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_document_size: 10 * 1024 * 1024, // 10MB
            min_document_size: 10, // 10 bytes
            max_line_length: 10000,
            allow_empty_content: false,
            sanitize_sensitive_data: true,
            normalize_whitespace: true,
            remove_control_chars: true,
            validate_encoding: true,
            allowed_file_types: ["pdf", "txt", "md", "doc", "docx", "html", "htm"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            blocked_patterns: vec![
                // Common malicious patterns
                r"<script[^>]*>.*?</script>".to_string(),
                r"javascript:".to_string(),
                r"vbscript:".to_string(),
                r"on\w+\s*=".to_string(),
            ],
            max_word_repetition: 50,
        }
    }
}

/// Result of content validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub sanitized_content: Option<String>,
    pub metadata: ValidationMetadata,
}

/// Metadata collected during validation
#[derive(Debug, Clone, Default)]
pub struct ValidationMetadata {
    pub original_size: usize,
    pub sanitized_size: Option<usize>,
    pub encoding_detected: Option<String>,
    pub language_confidence: Option<f32>,
    pub suspicious_patterns_found: Vec<String>,
    pub content_quality_score: Option<f32>,
}

/// Content validator and sanitizer
pub struct ContentValidator {
    config: ValidationConfig,
    // Compiled regex patterns for performance
    sensitive_data_patterns: Vec<Regex>,
    blocked_patterns: Vec<Regex>,
    whitespace_pattern: Regex,
    control_char_pattern: Regex,
}

impl ContentValidator {
    /// Create a new content validator with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ValidationConfig::default())
    }

    /// Create a new content validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Result<Self> {
        let sensitive_data_patterns = Self::compile_sensitive_patterns()?;
        let blocked_patterns = Self::compile_blocked_patterns(&config.blocked_patterns)?;
        
        let whitespace_pattern = Regex::new(r"\s+")
            .map_err(|e| crate::utils::Error::validation_simple(format!("Invalid whitespace regex: {}", e)))?;
        
        let control_char_pattern = Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
            .map_err(|e| crate::utils::Error::validation_simple(format!("Invalid control char regex: {}", e)))?;

        Ok(Self {
            config,
            sensitive_data_patterns,
            blocked_patterns,
            whitespace_pattern,
            control_char_pattern,
        })
    }

    /// Validate and sanitize content
    pub fn validate_and_sanitize(&self, content: &str, file_type: Option<&str>) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            is_valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
            sanitized_content: None,
            metadata: ValidationMetadata {
                original_size: content.len(),
                ..Default::default()
            },
        };

        // 1. Basic validation checks
        self.validate_basic_constraints(content, &mut result);

        // 2. File type validation
        if let Some(ft) = file_type {
            self.validate_file_type(ft, &mut result);
        }

        // 3. Encoding validation
        if self.config.validate_encoding {
            self.validate_encoding(content, &mut result);
        }

        // 4. Content security validation
        self.validate_security(content, &mut result);

        // 5. Content quality assessment
        self.assess_content_quality(content, &mut result);

        // 6. Sanitization (if validation passes or warnings only)
        if result.is_valid || result.errors.is_empty() {
            let sanitized = self.sanitize_content(content)?;
            result.sanitized_content = Some(sanitized.clone());
            result.metadata.sanitized_size = Some(sanitized.len());
        }

        Ok(result)
    }

    /// Basic validation constraints
    fn validate_basic_constraints(&self, content: &str, result: &mut ValidationResult) {
        // Size checks
        if content.len() > self.config.max_document_size {
            result.errors.push(format!(
                "Document size ({} bytes) exceeds maximum allowed size ({} bytes)",
                content.len(),
                self.config.max_document_size
            ));
            result.is_valid = false;
        }

        if content.len() < self.config.min_document_size {
            result.errors.push(format!(
                "Document size ({} bytes) is below minimum required size ({} bytes)",
                content.len(),
                self.config.min_document_size
            ));
            result.is_valid = false;
        }

        // Empty content check
        if content.trim().is_empty() && !self.config.allow_empty_content {
            result.errors.push("Empty content is not allowed".to_string());
            result.is_valid = false;
        }

        // Line length checks
        for (line_num, line) in content.lines().enumerate() {
            if line.len() > self.config.max_line_length {
                result.warnings.push(format!(
                    "Line {} exceeds maximum length ({} > {})",
                    line_num + 1,
                    line.len(),
                    self.config.max_line_length
                ));
            }
        }
    }

    /// Validate file type
    fn validate_file_type(&self, file_type: &str, result: &mut ValidationResult) {
        let normalized_type = file_type.to_lowercase();
        if !self.config.allowed_file_types.contains(&normalized_type) {
            result.errors.push(format!("File type '{}' is not allowed", file_type));
            result.is_valid = false;
        }
    }

    /// Validate text encoding
    fn validate_encoding(&self, content: &str, result: &mut ValidationResult) {
        // Check for valid UTF-8 (Rust strings are always valid UTF-8, but we can check for suspicious patterns)
        let mut suspicious_chars = 0;
        let mut total_chars = 0;

        for ch in content.chars() {
            total_chars += 1;
            
            // Count potentially suspicious characters
            if ch.is_control() && ch != '\n' && ch != '\r' && ch != '\t' {
                suspicious_chars += 1;
            }
        }

        if total_chars > 0 {
            let suspicious_ratio = suspicious_chars as f32 / total_chars as f32;
            if suspicious_ratio > 0.1 {
                result.warnings.push(format!(
                    "High ratio of control characters detected ({:.2}%)",
                    suspicious_ratio * 100.0
                ));
            }
        }

        result.metadata.encoding_detected = Some("UTF-8".to_string());
    }

    /// Validate content security
    fn validate_security(&self, content: &str, result: &mut ValidationResult) {
        // Check for blocked patterns
        for pattern in &self.blocked_patterns {
            if let Some(matches) = pattern.find(content) {
                let pattern_str = &content[matches.start()..matches.end().min(matches.start() + 50)];
                result.errors.push(format!("Blocked content pattern detected: {}", pattern_str));
                result.metadata.suspicious_patterns_found.push(pattern_str.to_string());
                result.is_valid = false;
            }
        }

        // Check for potential sensitive data
        if self.config.sanitize_sensitive_data {
            for pattern in &self.sensitive_data_patterns {
                if pattern.is_match(content) {
                    result.warnings.push("Potential sensitive data detected and will be sanitized".to_string());
                    break;
                }
            }
        }
    }

    /// Assess content quality
    fn assess_content_quality(&self, content: &str, result: &mut ValidationResult) {
        let mut quality_score: f32 = 100.0;
        let mut issues = Vec::new();

        // Word repetition analysis
        let words: Vec<&str> = content.split_whitespace().collect();
        if !words.is_empty() {
            let mut word_counts = std::collections::HashMap::new();
            for word in &words {
                let normalized_word = word.to_lowercase();
                *word_counts.entry(normalized_word).or_insert(0) += 1;
            }

            let max_repetition = word_counts.values().max().copied().unwrap_or(0);
            if max_repetition > self.config.max_word_repetition {
                quality_score -= 20.0;
                issues.push(format!("Excessive word repetition detected (max: {})", max_repetition));
            }

            // Vocabulary diversity
            let unique_words = word_counts.len();
            let diversity_ratio = unique_words as f32 / words.len() as f32;
            if diversity_ratio < 0.3 {
                quality_score -= 15.0;
                issues.push("Low vocabulary diversity".to_string());
            }
        }

        // Sentence structure analysis
        let sentences: Vec<&str> = content.split(&['.', '!', '?'][..]).collect();
        let avg_sentence_length = if !sentences.is_empty() {
            content.len() / sentences.len()
        } else {
            0
        };

        if avg_sentence_length > 500 {
            quality_score -= 10.0;
            issues.push("Very long sentences detected".to_string());
        } else if avg_sentence_length < 10 && !sentences.is_empty() {
            quality_score -= 5.0;
            issues.push("Very short sentences detected".to_string());
        }

        // Character distribution analysis
        let alpha_count = content.chars().filter(|c| c.is_alphabetic()).count();
        let total_chars = content.chars().count();
        
        if total_chars > 0 {
            let alpha_ratio = alpha_count as f32 / total_chars as f32;
            if alpha_ratio < 0.3 {
                quality_score -= 20.0;
                issues.push("Low alphabetic character ratio".to_string());
            }
        }

        result.metadata.content_quality_score = Some(quality_score.max(0.0));
        
        for issue in issues {
            result.warnings.push(format!("Content quality: {}", issue));
        }
    }

    /// Sanitize content based on configuration
    fn sanitize_content(&self, content: &str) -> Result<String> {
        let mut sanitized = content.to_string();

        // Remove control characters
        if self.config.remove_control_chars {
            sanitized = self.control_char_pattern.replace_all(&sanitized, "").to_string();
        }

        // Normalize whitespace
        if self.config.normalize_whitespace {
            sanitized = self.whitespace_pattern.replace_all(&sanitized, " ").to_string();
            sanitized = sanitized.trim().to_string();
        }

        // Sanitize sensitive data
        if self.config.sanitize_sensitive_data {
            for pattern in &self.sensitive_data_patterns {
                sanitized = pattern.replace_all(&sanitized, "[REDACTED]").to_string();
            }
        }

        // Remove any blocked patterns (replace with safe text)
        for pattern in &self.blocked_patterns {
            sanitized = pattern.replace_all(&sanitized, "[REMOVED]").to_string();
        }

        Ok(sanitized)
    }

    /// Compile sensitive data regex patterns
    fn compile_sensitive_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            // Social Security Numbers
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\b\d{9}\b",
            
            // Credit Card Numbers (basic pattern)
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            
            // Email addresses (if configured to redact)
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            
            // Phone Numbers
            r"\b\+?1?[- ]?\(?[0-9]{3}\)?[- ]?[0-9]{3}[- ]?[0-9]{4}\b",
            
            // IP Addresses
            r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            
            // URLs with sensitive info
            r"https?://[^\s]+[?&](password|token|key|secret)=[^\s&]+",
        ];

        let mut compiled_patterns = Vec::new();
        for pattern in patterns {
            match Regex::new(pattern) {
                Ok(regex) => compiled_patterns.push(regex),
                Err(e) => {
                    tracing::warn!("Failed to compile sensitive data pattern '{}': {}", pattern, e);
                }
            }
        }

        Ok(compiled_patterns)
    }

    /// Compile blocked content patterns
    fn compile_blocked_patterns(patterns: &[String]) -> Result<Vec<Regex>> {
        let mut compiled_patterns = Vec::new();
        
        for pattern in patterns {
            match Regex::new(pattern) {
                Ok(regex) => compiled_patterns.push(regex),
                Err(e) => {
                    return Err(crate::utils::Error::validation_simple(
                        format!("Invalid blocked pattern '{}': {}", pattern, e)
                    ));
                }
            }
        }

        Ok(compiled_patterns)
    }

    /// Quick validation check (without sanitization)
    pub fn quick_validate(&self, content: &str, file_type: Option<&str>) -> Result<bool> {
        // Basic size check
        if content.len() > self.config.max_document_size || 
           content.len() < self.config.min_document_size {
            return Ok(false);
        }

        // Empty content check
        if content.trim().is_empty() && !self.config.allow_empty_content {
            return Ok(false);
        }

        // File type check
        if let Some(ft) = file_type {
            let normalized_type = ft.to_lowercase();
            if !self.config.allowed_file_types.contains(&normalized_type) {
                return Ok(false);
            }
        }

        // Security check
        for pattern in &self.blocked_patterns {
            if pattern.is_match(content) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get validation statistics for a batch of documents
    pub fn batch_validate_stats(&self, contents: &[(&str, Option<&str>)]) -> Result<BatchValidationStats> {
        let mut stats = BatchValidationStats::default();
        
        for (content, file_type) in contents {
            let result = self.validate_and_sanitize(content, *file_type)?;
            
            stats.total_documents += 1;
            
            if result.is_valid {
                stats.valid_documents += 1;
            } else {
                stats.invalid_documents += 1;
            }
            
            stats.total_warnings += result.warnings.len();
            stats.total_errors += result.errors.len();
            stats.total_original_size += result.metadata.original_size;
            
            if let Some(sanitized_size) = result.metadata.sanitized_size {
                stats.total_sanitized_size += sanitized_size;
            }
            
            if let Some(quality_score) = result.metadata.content_quality_score {
                stats.quality_scores.push(quality_score);
            }
        }
        
        // Calculate average quality score
        if !stats.quality_scores.is_empty() {
            stats.average_quality_score = stats.quality_scores.iter().sum::<f32>() / stats.quality_scores.len() as f32;
        }
        
        Ok(stats)
    }
}

/// Statistics for batch validation operations
#[derive(Debug, Default)]
pub struct BatchValidationStats {
    pub total_documents: usize,
    pub valid_documents: usize,
    pub invalid_documents: usize,
    pub total_warnings: usize,
    pub total_errors: usize,
    pub total_original_size: usize,
    pub total_sanitized_size: usize,
    pub quality_scores: Vec<f32>,
    pub average_quality_score: f32,
}

impl Default for ContentValidator {
    fn default() -> Self {
        Self::new().expect("Failed to create default ContentValidator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_validation() {
        let validator = ContentValidator::new().unwrap();
        
        // Valid content
        let result = validator.validate_and_sanitize("This is valid content.", Some("txt")).unwrap();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        
        // Empty content
        let result = validator.validate_and_sanitize("", Some("txt")).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_security_validation() {
        let validator = ContentValidator::new().unwrap();
        
        // Content with script tag
        let malicious_content = "Hello <script>alert('xss')</script> world";
        let result = validator.validate_and_sanitize(malicious_content, Some("html")).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_sanitization() {
        let validator = ContentValidator::new().unwrap();
        
        // Content with extra whitespace
        let messy_content = "  Hello    world  \n\n  with   extra   spaces  ";
        let result = validator.validate_and_sanitize(messy_content, Some("txt")).unwrap();
        
        if let Some(sanitized) = result.sanitized_content {
            assert!(sanitized.len() < messy_content.len());
            assert!(!sanitized.contains("    ")); // No multiple spaces
        }
    }

    #[test]
    fn test_sensitive_data_detection() {
        let validator = ContentValidator::new().unwrap();
        
        // Content with SSN
        let sensitive_content = "My SSN is 123-45-6789 and email is test@example.com";
        let result = validator.validate_and_sanitize(sensitive_content, Some("txt")).unwrap();
        
        assert!(!result.warnings.is_empty());
        if let Some(sanitized) = result.sanitized_content {
            assert!(sanitized.contains("[REDACTED]"));
        }
    }

    #[test]
    fn test_quality_assessment() {
        let validator = ContentValidator::new().unwrap();
        
        // Low quality content (repetitive)
        let repetitive_content = "word word word word word word word word word word";
        let result = validator.validate_and_sanitize(repetitive_content, Some("txt")).unwrap();
        
        assert!(result.metadata.content_quality_score.unwrap_or(100.0) < 100.0);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_quick_validate() {
        let validator = ContentValidator::new().unwrap();
        
        assert!(validator.quick_validate("Valid content", Some("txt")).unwrap());
        assert!(!validator.quick_validate("", Some("txt")).unwrap());
        assert!(!validator.quick_validate("Valid content", Some("exe")).unwrap());
    }
}
