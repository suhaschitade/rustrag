use crate::models::{Query, QueryOptions};
// Note: QueryProcessor types would be imported when implemented
// use crate::core::query_processor::{QueryProcessor, QueryProcessorConfig, ProcessedQuery, QueryType};
use crate::core::query_expansion::{QueryExpansionService, QueryExpansionConfig, ExpansionResult};
use crate::utils::{Error, Result};
use std::sync::Arc;

/// Enhanced query processor that combines validation, processing, and expansion
pub struct EnhancedQueryProcessor {
    /// Basic query processor for validation and classification
    query_processor: QueryProcessor,
    /// Query expansion service for enhancing queries
    expansion_service: Arc<QueryExpansionService>,
    config: EnhancedQueryProcessorConfig,
}

/// Configuration for enhanced query processor
#[derive(Debug, Clone)]
pub struct EnhancedQueryProcessorConfig {
    /// Basic query processor configuration
    pub processor_config: QueryProcessorConfig,
    /// Query expansion configuration
    pub expansion_config: QueryExpansionConfig,
    /// Whether to enable advanced expansion (default: true)
    pub enable_advanced_expansion: bool,
    /// Whether to use expansion alternatives for query generation
    pub use_expansion_alternatives: bool,
    /// Whether to apply term weights from expansion
    pub apply_term_weights: bool,
}

impl Default for EnhancedQueryProcessorConfig {
    fn default() -> Self {
        Self {
            processor_config: QueryProcessorConfig::default(),
            expansion_config: QueryExpansionConfig::default(),
            enable_advanced_expansion: true,
            use_expansion_alternatives: true,
            apply_term_weights: true,
        }
    }
}

/// Enhanced processed query with expansion information
#[derive(Debug, Clone)]
pub struct EnhancedProcessedQuery {
    /// Basic processed query information
    pub basic: ProcessedQuery,
    /// Query expansion results
    pub expansion: Option<ExpansionResult>,
    /// Final query text to use for search (may be expanded)
    pub final_query: String,
    /// Alternative query formulations
    pub alternatives: Vec<String>,
    /// Term weights for ranking
    pub term_weights: std::collections::HashMap<String, f32>,
    /// Overall processing confidence
    pub overall_confidence: f32,
}

impl EnhancedQueryProcessor {
    /// Create new enhanced query processor with default configuration
    pub fn new() -> Self {
        Self::with_config(EnhancedQueryProcessorConfig::default())
    }

    /// Create enhanced query processor with custom configuration
    pub fn with_config(config: EnhancedQueryProcessorConfig) -> Self {
        let query_processor = QueryProcessor::with_config(config.processor_config.clone());
        let expansion_service = Arc::new(QueryExpansionService::with_config(config.expansion_config.clone()));

        Self {
            query_processor,
            expansion_service,
            config,
        }
    }

    /// Process and enhance a query with full expansion capabilities
    pub async fn process_query_enhanced(&self, query_text: &str) -> Result<EnhancedProcessedQuery> {
        tracing::info!("Processing enhanced query: {}", query_text);

        // Step 1: Basic query processing and validation
        let basic_processed = self.query_processor.process_query(query_text).await?;
        tracing::debug!("Basic processing completed for query type: {:?}", basic_processed.query_type);

        // Step 2: Advanced query expansion (if enabled)
        let expansion_result = if self.config.enable_advanced_expansion {
            match self.expansion_service.expand_query(query_text).await {
                Ok(expansion) => {
                    tracing::info!("Query expansion successful. Original: '{}', Expanded: '{}'", 
                                  expansion.original_query, expansion.expanded_query);
                    Some(expansion)
                }
                Err(e) => {
                    tracing::warn!("Query expansion failed: {}. Continuing with basic processing.", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 3: Determine final query text
        let final_query = self.determine_final_query(&basic_processed, &expansion_result);

        // Step 4: Collect alternatives
        let mut alternatives = Vec::new();
        
        // Add basic enhanced query if available
        if let Some(enhanced) = &basic_processed.enhanced {
            if enhanced != &final_query {
                alternatives.push(enhanced.clone());
            }
        }

        // Add expansion alternatives if enabled
        if self.config.use_expansion_alternatives {
            if let Some(ref expansion) = expansion_result {
                for alt in &expansion.alternatives {
                    if !alternatives.contains(alt) && alt != &final_query {
                        alternatives.push(alt.clone());
                    }
                }
            }
        }

        // Step 5: Determine term weights
        let term_weights = self.calculate_term_weights(&basic_processed, &expansion_result);

        // Step 6: Calculate overall confidence
        let overall_confidence = self.calculate_overall_confidence(&basic_processed, &expansion_result);

        tracing::info!("Enhanced query processing completed. Final query: '{}'", final_query);

        Ok(EnhancedProcessedQuery {
            basic: basic_processed,
            expansion: expansion_result,
            final_query,
            alternatives,
            term_weights,
            overall_confidence,
        })
    }

    /// Create optimized Query object from enhanced processing
    pub fn create_optimized_query(&self, enhanced: &EnhancedProcessedQuery, options: Option<QueryOptions>) -> Query {
        let mut query_options = options.unwrap_or_default();

        // Apply query type optimizations from basic processing
        self.apply_query_type_optimizations(&mut query_options, &enhanced.basic.query_type);

        // Apply expansion-based optimizations
        if let Some(ref expansion) = enhanced.expansion {
            self.apply_expansion_optimizations(&mut query_options, expansion);
        }

        // Apply confidence-based optimizations
        self.apply_confidence_optimizations(&mut query_options, enhanced.overall_confidence);

        Query::new_with_options(enhanced.final_query.clone(), query_options)
    }

    /// Determine the final query text to use
    fn determine_final_query(&self, basic: &ProcessedQuery, expansion: &Option<ExpansionResult>) -> String {
        if let Some(expansion_result) = expansion {
            // Use expanded query if expansion was successful and confident
            if expansion_result.confidence > 0.6 {
                expansion_result.expanded_query.clone()
            } else if expansion_result.confidence > 0.4 {
                // Use refined query for moderate confidence
                expansion_result.refined_query.clone()
            } else {
                // Fall back to basic enhanced query
                basic.enhanced.clone().unwrap_or_else(|| basic.processed.clone())
            }
        } else {
            // Use basic enhanced query if available
            basic.enhanced.clone().unwrap_or_else(|| basic.processed.clone())
        }
    }

    /// Calculate combined term weights
    fn calculate_term_weights(
        &self,
        basic: &ProcessedQuery,
        expansion: &Option<ExpansionResult>,
    ) -> std::collections::HashMap<String, f32> {
        let mut weights = std::collections::HashMap::new();

        if !self.config.apply_term_weights {
            return weights;
        }

        // Add weights for original terms (highest priority)
        for term in basic.processed.split_whitespace() {
            weights.insert(term.to_lowercase(), 1.0);
        }

        // Add weights for key terms (high priority)
        for term in &basic.key_terms {
            weights.insert(term.clone(), 0.9);
        }

        // Add weights from expansion if available
        if let Some(expansion_result) = expansion {
            for (term, weight) in &expansion_result.term_weights {
                // Don't override higher weights
                let current_weight = weights.get(term).cloned().unwrap_or(0.0);
                if *weight > current_weight {
                    weights.insert(term.clone(), *weight);
                }
            }
        }

        weights
    }

    /// Calculate overall processing confidence
    fn calculate_overall_confidence(
        &self,
        basic: &ProcessedQuery,
        expansion: &Option<ExpansionResult>,
    ) -> f32 {
        let basic_confidence = basic.intent_confidence;
        
        if let Some(expansion_result) = expansion {
            // Weighted average of basic and expansion confidence
            (basic_confidence * 0.6) + (expansion_result.confidence * 0.4)
        } else {
            basic_confidence
        }
    }

    /// Apply query type-based optimizations
    fn apply_query_type_optimizations(&self, options: &mut QueryOptions, query_type: &QueryType) {
        match query_type {
            QueryType::Technical => {
                // Technical queries need precise results
                options.similarity_threshold = Some(options.similarity_threshold.unwrap_or(0.8).max(0.8));
                options.max_chunks = Some(options.max_chunks.unwrap_or(5).min(5));
            }
            QueryType::Question => {
                // Questions benefit from more context and citations
                options.max_chunks = Some(options.max_chunks.unwrap_or(10).max(8));
                options.include_citations = true;
            }
            QueryType::Search => {
                // Search queries can have broader results
                options.similarity_threshold = Some(options.similarity_threshold.unwrap_or(0.6).min(0.7));
                options.max_chunks = Some(options.max_chunks.unwrap_or(15).max(10));
            }
            QueryType::Command => {
                // Commands should be precise and actionable
                options.similarity_threshold = Some(options.similarity_threshold.unwrap_or(0.75));
                options.max_chunks = Some(options.max_chunks.unwrap_or(7));
            }
            _ => {
                // Default optimizations for other types
                options.max_chunks = Some(options.max_chunks.unwrap_or(10));
            }
        }
    }

    /// Apply expansion-based optimizations
    fn apply_expansion_optimizations(&self, options: &mut QueryOptions, expansion: &ExpansionResult) {
        // If we have many synonyms, we might want more results to capture semantic variations
        let synonym_boost = (expansion.synonyms.len() as f32 * 0.1).min(0.3);
        let semantic_boost = (expansion.semantic_terms.len() as f32 * 0.05).min(0.2);
        
        if synonym_boost > 0.0 || semantic_boost > 0.0 {
            let current_max = options.max_chunks.unwrap_or(10) as f32;
            let new_max = (current_max * (1.0 + synonym_boost + semantic_boost)) as usize;
            options.max_chunks = Some(new_max.min(20)); // Cap at 20
        }

        // If we handled negations, adjust similarity threshold
        if !expansion.negations.is_empty() {
            // Negations require more precise matching
            let current_threshold = options.similarity_threshold.unwrap_or(0.7);
            options.similarity_threshold = Some((current_threshold + 0.1).min(0.9));
        }

        // If expansion confidence is low, be more conservative
        if expansion.confidence < 0.5 {
            let current_threshold = options.similarity_threshold.unwrap_or(0.7);
            options.similarity_threshold = Some((current_threshold + 0.05).min(0.85));
        }
    }

    /// Apply confidence-based optimizations
    fn apply_confidence_optimizations(&self, options: &mut QueryOptions, confidence: f32) {
        if confidence < 0.5 {
            // Low confidence: be more inclusive
            let current_threshold = options.similarity_threshold.unwrap_or(0.7);
            options.similarity_threshold = Some((current_threshold - 0.1).max(0.5));
            
            let current_max = options.max_chunks.unwrap_or(10);
            options.max_chunks = Some((current_max + 3).min(20));
        } else if confidence > 0.8 {
            // High confidence: be more selective
            let current_threshold = options.similarity_threshold.unwrap_or(0.7);
            options.similarity_threshold = Some((current_threshold + 0.05).min(0.9));
        }
    }

    /// Get query processing statistics
    pub fn get_processing_stats(&self, enhanced: &EnhancedProcessedQuery) -> QueryProcessingStats {
        QueryProcessingStats {
            original_term_count: enhanced.basic.original.split_whitespace().count(),
            processed_term_count: enhanced.basic.tokens.len(),
            key_terms_count: enhanced.basic.key_terms.len(),
            synonyms_added: enhanced.expansion.as_ref().map(|e| e.synonyms.len()).unwrap_or(0),
            semantic_terms_added: enhanced.expansion.as_ref().map(|e| e.semantic_terms.len()).unwrap_or(0),
            negations_handled: enhanced.expansion.as_ref().map(|e| e.negations.len()).unwrap_or(0),
            alternatives_generated: enhanced.alternatives.len(),
            expansion_confidence: enhanced.expansion.as_ref().map(|e| e.confidence).unwrap_or(0.0),
            overall_confidence: enhanced.overall_confidence,
            query_type: enhanced.basic.query_type.clone(),
        }
    }
}

/// Statistics about query processing
#[derive(Debug, Clone)]
pub struct QueryProcessingStats {
    pub original_term_count: usize,
    pub processed_term_count: usize,
    pub key_terms_count: usize,
    pub synonyms_added: usize,
    pub semantic_terms_added: usize,
    pub negations_handled: usize,
    pub alternatives_generated: usize,
    pub expansion_confidence: f32,
    pub overall_confidence: f32,
    pub query_type: QueryType,
}

impl Default for EnhancedQueryProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_enhanced_query_processing() {
        let processor = EnhancedQueryProcessor::new();
        
        let result = processor.process_query_enhanced("How to implement REST API?").await.unwrap();
        
        assert_eq!(result.basic.query_type, QueryType::Question);
        assert!(!result.final_query.is_empty());
        assert!(result.overall_confidence > 0.0);
        
        // Should have expansion results
        assert!(result.expansion.is_some());
        
        // Should handle acronym expansion
        if let Some(expansion) = &result.expansion {
            assert!(expansion.refined_query.contains("representational state transfer"));
        }
    }

    #[tokio::test]
    async fn test_technical_query_optimization() {
        let processor = EnhancedQueryProcessor::new();
        
        let enhanced = processor.process_query_enhanced("database connection error").await.unwrap();
        let query = processor.create_optimized_query(&enhanced, None);
        
        // Technical queries should have high similarity threshold
        assert!(query.options.similarity_threshold.unwrap_or(0.0) >= 0.8);
        // And fewer chunks for precision
        assert!(query.options.max_chunks.unwrap_or(100) <= 5);
    }

    #[tokio::test]
    async fn test_question_query_optimization() {
        let processor = EnhancedQueryProcessor::new();
        
        let enhanced = processor.process_query_enhanced("What is machine learning?").await.unwrap();
        let query = processor.create_optimized_query(&enhanced, None);
        
        // Questions should include citations
        assert!(query.options.include_citations);
        // And have more chunks for context
        assert!(query.options.max_chunks.unwrap_or(0) >= 8);
    }

    #[tokio::test] 
    async fn test_negation_handling() {
        let processor = EnhancedQueryProcessor::new();
        
        let enhanced = processor.process_query_enhanced("not an error but a feature").await.unwrap();
        
        if let Some(expansion) = &enhanced.expansion {
            assert!(!expansion.negations.is_empty());
        }
        
        // Should adjust similarity threshold for negations
        let query = processor.create_optimized_query(&enhanced, None);
        assert!(query.options.similarity_threshold.unwrap_or(0.0) > 0.7);
    }

    #[test]
    fn test_processing_stats() {
        let enhanced = EnhancedProcessedQuery {
            basic: ProcessedQuery {
                original: "test query".to_string(),
                processed: "test query".to_string(),
                enhanced: None,
                tokens: vec!["test".to_string(), "query".to_string()],
                key_terms: vec!["query".to_string()],
                query_type: QueryType::Search,
                language: Some("en".to_string()),
                intent_confidence: 0.8,
            },
            expansion: None,
            final_query: "test query".to_string(),
            alternatives: vec![],
            term_weights: std::collections::HashMap::new(),
            overall_confidence: 0.8,
        };

        let processor = EnhancedQueryProcessor::new();
        let stats = processor.get_processing_stats(&enhanced);

        assert_eq!(stats.original_term_count, 2);
        assert_eq!(stats.processed_term_count, 2);
        assert_eq!(stats.key_terms_count, 1);
        assert_eq!(stats.query_type, QueryType::Search);
    }
}
