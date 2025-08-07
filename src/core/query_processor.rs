use crate::models::{Query, QueryOptions};
use crate::core::query_expansion::{QueryExpansionService, QueryExpansionConfig, ExpansionResult};
use crate::utils::{Error, Result};
use regex::Regex;
use std::collections::HashSet;
use std::sync::Arc;

/// Query processing and validation service
pub struct QueryProcessor {
    config: QueryProcessorConfig,
    malicious_patterns: Vec<Regex>,
    stop_words: HashSet<String>,
}

/// Configuration for query processing
#[derive(Debug, Clone)]
pub struct QueryProcessorConfig {
    /// Maximum query length in characters
    pub max_query_length: usize,
    /// Minimum query length in characters
    pub min_query_length: usize,
    /// Whether to enable query enhancement/expansion
    pub enable_enhancement: bool,
    /// Whether to enable malicious content detection
    pub enable_security_checks: bool,
    /// Whether to normalize unicode characters
    pub normalize_unicode: bool,
    /// Maximum number of terms in a query
    pub max_terms: usize,
}

impl Default for QueryProcessorConfig {
    fn default() -> Self {
        Self {
            max_query_length: 2000,
            min_query_length: 3,
            enable_enhancement: true,
            enable_security_checks: true,
            normalize_unicode: true,
            max_terms: 50,
        }
    }
}

/// Query validation result
#[derive(Debug, Clone)]
pub struct QueryValidationResult {
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub processed_query: String,
    pub detected_language: Option<String>,
    pub query_type: QueryType,
    pub confidence: f32,
}

/// Query validation issues
#[derive(Debug, Clone)]
pub enum ValidationIssue {
    TooShort(usize),
    TooLong(usize),
    TooManyTerms(usize),
    MaliciousContent(String),
    InvalidCharacters(String),
    EmptyQuery,
    ExcessiveWhitespace,
}

/// Type of query detected
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    Question,
    Command,
    Search,
    Conversational,
    Technical,
    Unknown,
}

/// Enhanced query with additional metadata
#[derive(Debug, Clone)]
pub struct ProcessedQuery {
    pub original: String,
    pub processed: String,
    pub enhanced: Option<String>,
    pub tokens: Vec<String>,
    pub key_terms: Vec<String>,
    pub query_type: QueryType,
    pub language: Option<String>,
    pub intent_confidence: f32,
}

impl QueryProcessor {
    /// Create a new query processor with default configuration
    pub fn new() -> Self {
        Self::with_config(QueryProcessorConfig::default())
    }

    /// Create a new query processor with custom configuration
    pub fn with_config(config: QueryProcessorConfig) -> Self {
        let malicious_patterns = Self::build_malicious_patterns();
        let stop_words = Self::build_stop_words();

        Self {
            config,
            malicious_patterns,
            stop_words,
        }
    }

    /// Process and validate a query
    pub async fn process_query(&self, query_text: &str) -> Result<ProcessedQuery> {
        let validation_result = self.validate_query(query_text)?;
        
        if !validation_result.is_valid {
            return Err(Error::validation(
                "query".to_string(),
                format!("Validation failed: {:?}", validation_result.issues)
            ));
        }

        let processed_text = self.sanitize_query(query_text);
        let tokens = self.tokenize_query(&processed_text);
        let key_terms = self.extract_key_terms(&tokens);
        let tokens_str: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let query_type = self.classify_query(&processed_text, &tokens_str);
        let enhanced = if self.config.enable_enhancement {
            Some(self.enhance_query(&processed_text, &query_type).await?)
        } else {
            None
        };

        Ok(ProcessedQuery {
            original: query_text.to_string(),
            processed: processed_text,
            enhanced,
            tokens,
            key_terms,
            query_type: query_type.clone(),
            language: validation_result.detected_language,
            intent_confidence: validation_result.confidence,
        })
    }

    /// Validate a raw query text
    pub fn validate_query(&self, query_text: &str) -> Result<QueryValidationResult> {
        let mut issues = Vec::new();
        let trimmed = query_text.trim();

        // Check if query is empty
        if trimmed.is_empty() {
            issues.push(ValidationIssue::EmptyQuery);
            return Ok(QueryValidationResult {
                is_valid: false,
                issues,
                processed_query: String::new(),
                detected_language: None,
                query_type: QueryType::Unknown,
                confidence: 0.0,
            });
        }

        // Length validation
        if trimmed.len() < self.config.min_query_length {
            issues.push(ValidationIssue::TooShort(trimmed.len()));
        }

        if trimmed.len() > self.config.max_query_length {
            issues.push(ValidationIssue::TooLong(trimmed.len()));
        }

        // Term count validation
        let terms: Vec<&str> = trimmed.split_whitespace().collect();
        if terms.len() > self.config.max_terms {
            issues.push(ValidationIssue::TooManyTerms(terms.len()));
        }

        // Security checks
        if self.config.enable_security_checks {
            for pattern in &self.malicious_patterns {
                if pattern.is_match(trimmed) {
                    issues.push(ValidationIssue::MaliciousContent(
                        "Potentially malicious pattern detected".to_string()
                    ));
                    break;
                }
            }
        }

        // Check for excessive whitespace
        if trimmed.contains("  ") || trimmed != query_text.trim() {
            issues.push(ValidationIssue::ExcessiveWhitespace);
        }

        // Detect query type and language
        let query_type = self.classify_query(trimmed, &terms);
        let detected_language = self.detect_language(trimmed);
        let confidence = self.calculate_confidence(trimmed, &query_type);

        let processed_query = if issues.is_empty() {
            self.sanitize_query(trimmed)
        } else {
            trimmed.to_string()
        };

        Ok(QueryValidationResult {
            is_valid: issues.is_empty(),
            issues,
            processed_query,
            detected_language,
            query_type,
            confidence,
        })
    }

    /// Sanitize query text
    fn sanitize_query(&self, query: &str) -> String {
        let mut sanitized = query.trim().to_string();

        // Normalize Unicode if enabled
        if self.config.normalize_unicode {
            // Basic normalization - can be enhanced with unicode-normalization crate
            sanitized = sanitized
                .chars()
                .filter(|c| !c.is_control() || c.is_whitespace())
                .collect();
        }

        // Normalize whitespace
        let whitespace_regex = Regex::new(r"\s+").unwrap();
        sanitized = whitespace_regex.replace_all(&sanitized, " ").to_string();

        // Remove leading/trailing whitespace
        sanitized.trim().to_string()
    }

    /// Tokenize query into terms
    fn tokenize_query(&self, query: &str) -> Vec<String> {
        // Simple whitespace tokenization - can be enhanced with proper NLP tokenization
        query
            .split_whitespace()
            .map(|token| {
                token
                    .chars()
                    .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '\'' || *c == '-')
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|token| !token.is_empty() && !self.stop_words.contains(token))
            .collect()
    }

    /// Extract key terms from tokens
    fn extract_key_terms(&self, tokens: &[String]) -> Vec<String> {
        // Simple implementation - can be enhanced with TF-IDF or other techniques
        tokens
            .iter()
            .filter(|token| token.len() > 3) // Keep longer terms
            .cloned()
            .collect()
    }

    /// Classify the type of query
    fn classify_query(&self, query: &str, tokens: &[&str]) -> QueryType {
        let query_lower = query.to_lowercase();

        // Check for command patterns first (more specific)
        let command_patterns = [
            "show me", "give me", "search for", "tell me about", "help me",
            "find", "list", "display", "explain", "describe",
        ];

        if command_patterns.iter().any(|&pattern| query_lower.contains(pattern)) {
            return QueryType::Command;
        }

        // Question patterns (less specific than commands)
        let question_patterns = [
            "what", "how", "why", "when", "where", "who", "which", "whose",
            "can you", "could you", "would you", "do you", "did you", "will you",
            "is there", "are there", "does", "doesn't", "did", "didn't",
        ];

        if question_patterns.iter().any(|&pattern| query_lower.contains(pattern)) ||
           query.ends_with('?') {
            return QueryType::Question;
        }

        // Technical patterns
        let technical_patterns = [
            "api", "function", "method", "class", "variable", "error", "exception",
            "algorithm", "implementation", "code", "syntax", "debug",
        ];

        if technical_patterns.iter().any(|&pattern| query_lower.contains(pattern)) {
            return QueryType::Technical;
        }

        // Conversational patterns
        let conversational_patterns = [
            "hello", "hi", "thanks", "thank you", "please", "sorry",
            "i need", "i want", "i would like", "could you help",
        ];

        if conversational_patterns.iter().any(|&pattern| query_lower.contains(pattern)) {
            return QueryType::Conversational;
        }

        // Default to search if none of the above patterns match
        if tokens.len() >= 2 {
            QueryType::Search
        } else {
            QueryType::Unknown
        }
    }

    /// Detect query language (basic implementation)
    fn detect_language(&self, _query: &str) -> Option<String> {
        // Simple implementation - always returns English
        // In a real implementation, you would use a language detection library
        Some("en".to_string())
    }

    /// Calculate confidence score for query classification
    fn calculate_confidence(&self, query: &str, query_type: &QueryType) -> f32 {
        // Simple confidence calculation based on query characteristics
        let mut confidence: f32 = 0.5; // Base confidence

        // Boost confidence for longer queries
        if query.len() > 20 {
            confidence += 0.2;
        }

        // Boost confidence for specific query types
        match query_type {
            QueryType::Question if query.ends_with('?') => confidence += 0.3,
            QueryType::Command if query.to_lowercase().starts_with("show") => confidence += 0.2,
            QueryType::Technical if query.to_lowercase().contains("api") => confidence += 0.2,
            _ => {}
        }

        confidence.min(1.0)
    }

    /// Enhance query with synonyms and related terms
    async fn enhance_query(&self, query: &str, query_type: &QueryType) -> Result<String> {
        // Simple query enhancement - in a real implementation, this would use
        // word embeddings, thesaurus APIs, or domain-specific knowledge bases
        
        let enhancements = match query_type {
            QueryType::Technical => {
                // Add common technical synonyms
                if query.to_lowercase().contains("function") {
                    " method procedure routine".to_string()
                } else if query.to_lowercase().contains("error") {
                    " exception bug issue problem".to_string()
                } else {
                    String::new()
                }
            }
            QueryType::Question => {
                // Add question-related enhancements
                if query.to_lowercase().contains("how to") {
                    " tutorial guide instructions steps".to_string()
                } else {
                    String::new()
                }
            }
            _ => String::new(),
        };

        if enhancements.is_empty() {
            Ok(query.to_string())
        } else {
            Ok(format!("{}{}", query, enhancements))
        }
    }

    /// Build patterns for malicious content detection
    fn build_malicious_patterns() -> Vec<Regex> {
        let patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"data:text/html",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"eval\s*\(",
            r"document\s*\.\s*write",
            // Add more patterns as needed
        ];

        patterns
            .iter()
            .filter_map(|pattern| Regex::new(pattern).ok())
            .collect()
    }

    /// Build set of common stop words
    fn build_stop_words() -> HashSet<String> {
        let stop_words = [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "the", "this", "but", "they", "have",
            "had", "what", "said", "each", "which", "she", "do", "how", "their",
            "if", "up", "out", "many", "then", "them", "these", "so", "some",
        ];

        stop_words.iter().map(|&word| word.to_string()).collect()
    }

    /// Create a Query object from validated text
    pub fn create_query(&self, processed_query: &ProcessedQuery, options: Option<QueryOptions>) -> Query {
        let query_text = processed_query.enhanced
            .as_ref()
            .unwrap_or(&processed_query.processed)
            .clone();

        let mut query_options = options.unwrap_or_default();
        
        // Adjust query options based on query type
        match processed_query.query_type {
            QueryType::Technical => {
                // Technical queries might need more precise results
                query_options.similarity_threshold = Some(0.8);
                query_options.max_chunks = Some(5);
            }
            QueryType::Question => {
                // Questions might benefit from more context
                query_options.max_chunks = Some(10);
                query_options.include_citations = true;
            }
            QueryType::Search => {
                // Search queries might need broader results
                query_options.similarity_threshold = Some(0.6);
                query_options.max_chunks = Some(15);
            }
            _ => {}
        }

        Query::new_with_options(query_text, query_options)
    }
}

impl Default for QueryProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_validation() {
        let processor = QueryProcessor::new();
        
        // Valid query
        let result = processor.validate_query("What is machine learning?").unwrap();
        assert!(result.is_valid);
        assert_eq!(result.query_type, QueryType::Question);
        
        // Too short query
        let result = processor.validate_query("Hi").unwrap();
        assert!(!result.is_valid);
        assert!(matches!(result.issues[0], ValidationIssue::TooShort(_)));
        
        // Empty query
        let result = processor.validate_query("").unwrap();
        assert!(!result.is_valid);
        assert!(matches!(result.issues[0], ValidationIssue::EmptyQuery));
    }

    #[tokio::test]
    async fn test_query_processing() {
        let processor = QueryProcessor::new();
        
        let processed = processor.process_query("How to implement a REST API?").await.unwrap();
        assert_eq!(processed.query_type, QueryType::Question);
        assert!(!processed.tokens.is_empty());
        assert!(processed.enhanced.is_some());
        assert!(processed.intent_confidence > 0.5);
    }

    #[test]
    fn test_query_classification() {
        let processor = QueryProcessor::new();
        
        assert_eq!(
            processor.classify_query("What is Rust?", &["what", "is", "rust"]),
            QueryType::Question
        );
        
        assert_eq!(
            processor.classify_query("Show me all documents", &["show", "me", "all", "documents"]),
            QueryType::Command
        );
        
        assert_eq!(
            processor.classify_query("API documentation error", &["api", "documentation", "error"]),
            QueryType::Technical
        );
    }

    #[test]
    fn test_sanitization() {
        let processor = QueryProcessor::new();
        
        let sanitized = processor.sanitize_query("  Hello    world  ");
        assert_eq!(sanitized, "Hello world");
        
        let sanitized = processor.sanitize_query("Test\t\nquery");
        assert_eq!(sanitized, "Test query");
    }

    #[test]
    fn test_tokenization() {
        let processor = QueryProcessor::new();
        
        let tokens = processor.tokenize_query("How to implement machine learning algorithms?");
        assert!(!tokens.contains(&"to".to_string())); // Stop word should be removed
        assert!(tokens.contains(&"implement".to_string()));
        assert!(tokens.contains(&"machine".to_string()));
        assert!(tokens.contains(&"learning".to_string()));
        assert!(tokens.contains(&"algorithms".to_string()));
    }

    #[test]
    fn test_malicious_content_detection() {
        let processor = QueryProcessor::new();
        
        let result = processor.validate_query("<script>alert('xss')</script>").unwrap();
        assert!(!result.is_valid);
        assert!(result.issues.iter().any(|issue| matches!(issue, ValidationIssue::MaliciousContent(_))));
    }
}
