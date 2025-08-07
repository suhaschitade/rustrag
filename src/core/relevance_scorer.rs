use crate::models::{Query, DocumentChunk};
use crate::utils::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Configuration for relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceConfig {
    /// Weight for semantic similarity (0.0-1.0)
    pub semantic_weight: f32,
    /// Weight for keyword relevance (0.0-1.0)
    pub keyword_weight: f32,
    /// Weight for document quality (0.0-1.0)
    pub quality_weight: f32,
    /// Weight for freshness/recency (0.0-1.0)
    pub freshness_weight: f32,
    /// Weight for user context relevance (0.0-1.0)
    pub context_weight: f32,
    /// Whether to enable query-specific boosting
    pub enable_query_boosting: bool,
    /// Whether to enable document authority scoring
    pub enable_authority_scoring: bool,
    /// Whether to enable diversity penalty
    pub enable_diversity_penalty: bool,
    /// Threshold for considering a match relevant
    pub relevance_threshold: f32,
}

impl Default for RelevanceConfig {
    fn default() -> Self {
        Self {
            semantic_weight: 0.4,
            keyword_weight: 0.3,
            quality_weight: 0.15,
            freshness_weight: 0.05,
            context_weight: 0.1,
            enable_query_boosting: true,
            enable_authority_scoring: true,
            enable_diversity_penalty: true,
            relevance_threshold: 0.5,
        }
    }
}

/// Query analysis for relevance scoring
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub query_type: QueryType,
    pub intent: QueryIntent,
    pub complexity: QueryComplexity,
    pub key_terms: Vec<String>,
    pub named_entities: Vec<NamedEntity>,
    pub temporal_context: Option<TemporalContext>,
}

/// Query type classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryType {
    Factual,        // What is X?
    Procedural,     // How to do X?
    Comparative,    // X vs Y
    Analytical,     // Why/explain X?
    Navigational,   // Find specific document/section
    Conversational, // Casual question
}

/// Query intent classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryIntent {
    InformationSeeking,
    ProblemSolving,
    TaskCompletion,
    Learning,
    Decision,
    Exploration,
}

/// Query complexity level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryComplexity {
    Simple,     // Single concept
    Moderate,   // Multiple concepts
    Complex,    // Multi-faceted with relationships
    Expert,     // Domain-specific, technical
}

/// Named entity in query
#[derive(Debug, Clone)]
pub struct NamedEntity {
    pub text: String,
    pub entity_type: EntityType,
    pub confidence: f32,
}

/// Entity types
#[derive(Debug, Clone, PartialEq)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Technology,
    Product,
    Concept,
    Date,
    Metric,
}

/// Temporal context for time-sensitive queries
#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub time_sensitivity: f32,  // 0-1, how time-sensitive is this query
    pub preferred_recency: Option<chrono::Duration>, // How recent should results be
}

/// Document quality metrics
#[derive(Debug, Clone, Default)]
pub struct DocumentQuality {
    pub readability_score: f32,     // 0-1, how readable is the content
    pub completeness_score: f32,    // 0-1, how complete is the information
    pub authority_score: f32,       // 0-1, source authority/credibility
    pub structure_score: f32,       // 0-1, how well-structured
    pub depth_score: f32,          // 0-1, depth of information
}

/// Comprehensive relevance score with explanations
#[derive(Debug, Clone)]
pub struct RelevanceScore {
    pub overall_score: f32,
    pub semantic_score: f32,
    pub keyword_score: f32,
    pub quality_score: f32,
    pub freshness_score: f32,
    pub context_score: f32,
    pub boost_multiplier: f32,
    pub diversity_penalty: f32,
    pub confidence: f32,
    pub explanation: RelevanceExplanation,
}

/// Detailed explanation of relevance scoring
#[derive(Debug, Clone)]
pub struct RelevanceExplanation {
    pub main_factors: Vec<String>,
    pub positive_signals: Vec<String>,
    pub negative_signals: Vec<String>,
    pub recommendations: Vec<String>,
    pub debug_info: HashMap<String, f32>,
}

/// Advanced relevance scorer
pub struct RelevanceScorer {
    config: RelevanceConfig,
    term_statistics: Arc<TermStatistics>,
    document_graph: Arc<DocumentGraph>,
}

/// Term statistics for advanced scoring
#[derive(Debug, Default)]
pub struct TermStatistics {
    pub term_frequencies: HashMap<String, usize>,
    pub document_frequencies: HashMap<String, usize>,
    pub co_occurrence_matrix: HashMap<(String, String), f32>,
    pub term_importance: HashMap<String, f32>,
    pub total_documents: usize,
}

/// Document relationship graph for authority scoring
#[derive(Debug, Default)]
pub struct DocumentGraph {
    pub citations: HashMap<uuid::Uuid, Vec<uuid::Uuid>>,
    pub references: HashMap<uuid::Uuid, Vec<uuid::Uuid>>,
    pub authority_scores: HashMap<uuid::Uuid, f32>,
    pub topic_clusters: HashMap<String, Vec<uuid::Uuid>>,
}

impl RelevanceScorer {
    /// Create a new relevance scorer
    pub fn new(config: RelevanceConfig) -> Self {
        Self {
            config,
            term_statistics: Arc::new(TermStatistics::default()),
            document_graph: Arc::new(DocumentGraph::default()),
        }
    }

    /// Create with pre-computed statistics
    pub fn with_statistics(
        config: RelevanceConfig,
        term_statistics: Arc<TermStatistics>,
        document_graph: Arc<DocumentGraph>,
    ) -> Self {
        Self {
            config,
            term_statistics,
            document_graph,
        }
    }

    /// Analyze query to understand intent and context
    pub fn analyze_query(&self, query: &Query) -> QueryAnalysis {
        let text = query.text.to_lowercase();
        let tokens: Vec<&str> = text.split_whitespace().collect();

        // Classify query type
        let query_type = self.classify_query_type(&text, &tokens);

        // Determine intent
        let intent = self.determine_intent(&text, &tokens, &query_type);

        // Assess complexity
        let complexity = self.assess_complexity(&tokens, &query_type);

        // Extract key terms (removing stop words and common terms)
        let key_terms = self.extract_key_terms(&tokens);

        // Extract named entities (simplified implementation)
        let named_entities = self.extract_named_entities(&text);

        // Determine temporal context
        let temporal_context = self.extract_temporal_context(&text);

        QueryAnalysis {
            query_type,
            intent,
            complexity,
            key_terms,
            named_entities,
            temporal_context,
        }
    }

    /// Calculate comprehensive relevance score
    pub async fn calculate_relevance(
        &self,
        query_analysis: &QueryAnalysis,
        chunk: &DocumentChunk,
        semantic_similarity: f32,
        keyword_score: f32,
    ) -> Result<RelevanceScore> {
        debug!("Calculating relevance for chunk {}", chunk.id);

        // Calculate individual score components
        let semantic_score = self.calculate_semantic_score(semantic_similarity, query_analysis);
        let keyword_adjusted_score = self.calculate_keyword_relevance(keyword_score, query_analysis, chunk);
        let quality_score = self.calculate_document_quality(chunk).await;
        let freshness_score = self.calculate_freshness_score(chunk, query_analysis);
        let context_score = self.calculate_context_relevance(query_analysis, chunk);

        // Calculate boost multiplier
        let boost_multiplier = if self.config.enable_query_boosting {
            self.calculate_query_boost(query_analysis, chunk)
        } else {
            1.0
        };

        // Calculate diversity penalty (will be applied later in ranking)
        let diversity_penalty = 0.0; // Calculated during ranking phase

        // Combine scores using weighted average
        let base_score = (semantic_score * self.config.semantic_weight) +
                        (keyword_adjusted_score * self.config.keyword_weight) +
                        (quality_score * self.config.quality_weight) +
                        (freshness_score * self.config.freshness_weight) +
                        (context_score * self.config.context_weight);

        let overall_score = (base_score * boost_multiplier) - diversity_penalty;
        
        // Calculate confidence based on score distribution
        let confidence = self.calculate_confidence(&[
            semantic_score,
            keyword_adjusted_score,
            quality_score,
            freshness_score,
            context_score,
        ]);

        // Generate explanation
        let explanation = self.generate_explanation(
            query_analysis,
            chunk,
            semantic_score,
            keyword_adjusted_score,
            quality_score,
            freshness_score,
            context_score,
            boost_multiplier,
        );

        Ok(RelevanceScore {
            overall_score,
            semantic_score,
            keyword_score: keyword_adjusted_score,
            quality_score,
            freshness_score,
            context_score,
            boost_multiplier,
            diversity_penalty,
            confidence,
            explanation,
        })
    }

    /// Rank multiple chunks with diversity consideration
    pub async fn rank_chunks(
        &self,
        query_analysis: &QueryAnalysis,
        chunks_with_scores: Vec<(DocumentChunk, f32, f32)>, // (chunk, semantic_sim, keyword_score)
    ) -> Result<Vec<(DocumentChunk, RelevanceScore)>> {
        info!("Ranking {} chunks with advanced relevance scoring", chunks_with_scores.len());

        let mut scored_chunks = Vec::new();

        // Calculate initial relevance scores
        for (chunk, semantic_sim, keyword_score) in chunks_with_scores {
            let relevance = self.calculate_relevance(
                query_analysis,
                &chunk,
                semantic_sim,
                keyword_score,
            ).await?;

            scored_chunks.push((chunk, relevance));
        }

        // Apply diversity penalty using Maximal Marginal Relevance approach
        if self.config.enable_diversity_penalty {
            scored_chunks = self.apply_diversity_penalty(scored_chunks, query_analysis).await;
        }

        // Sort by final relevance score
        scored_chunks.sort_by(|a, b| {
            b.1.overall_score
                .partial_cmp(&a.1.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored_chunks)
    }

    /// Calculate semantic relevance score with query-specific adjustments
    fn calculate_semantic_score(&self, similarity: f32, query_analysis: &QueryAnalysis) -> f32 {
        let mut score = similarity;

        // Adjust based on query complexity
        match query_analysis.complexity {
            QueryComplexity::Simple => score *= 1.0,     // No adjustment
            QueryComplexity::Moderate => score *= 1.05,  // Slight boost
            QueryComplexity::Complex => score *= 1.1,    // Moderate boost
            QueryComplexity::Expert => score *= 1.15,    // Higher boost for expert queries
        }

        // Adjust based on query type
        match query_analysis.query_type {
            QueryType::Factual => score *= 1.1,      // Favor semantic similarity for facts
            QueryType::Analytical => score *= 1.05,  // Moderate boost for analysis
            QueryType::Procedural => score *= 0.95,  // Slight penalty (favor keyword matching)
            _ => score *= 1.0,
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate enhanced keyword relevance
    fn calculate_keyword_relevance(
        &self,
        base_keyword_score: f32,
        query_analysis: &QueryAnalysis,
        chunk: &DocumentChunk,
    ) -> f32 {
        let mut score = base_keyword_score;

        // Boost for key term matches
        let content_lower = chunk.content.to_lowercase();
        let key_term_matches = query_analysis.key_terms.iter()
            .filter(|term| content_lower.contains(*term))
            .count();

        if key_term_matches > 0 {
            let boost = (key_term_matches as f32 / query_analysis.key_terms.len() as f32) * 0.2;
            score *= 1.0 + boost;
        }

        // Boost for named entity matches
        for entity in &query_analysis.named_entities {
            if content_lower.contains(&entity.text.to_lowercase()) {
                score *= 1.0 + (entity.confidence * 0.15);
            }
        }

        // Adjust based on query type preferences
        match query_analysis.query_type {
            QueryType::Procedural => score *= 1.2,   // Favor exact matches for procedures
            QueryType::Navigational => score *= 1.25, // Strong preference for exact matches
            QueryType::Comparative => score *= 1.1,  // Moderate boost for comparisons
            _ => score *= 1.0,
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate document quality score
    async fn calculate_document_quality(&self, chunk: &DocumentChunk) -> f32 {
        let content = &chunk.content;
        let mut quality = DocumentQuality::default();

        // Readability: sentence length and complexity
        quality.readability_score = self.calculate_readability(content);

        // Completeness: content length and structure indicators
        quality.completeness_score = self.calculate_completeness(content);

        // Authority: use pre-computed authority scores or heuristics
        quality.authority_score = self.document_graph
            .authority_scores
            .get(&chunk.document_id)
            .copied()
            .unwrap_or(0.5);

        // Structure: presence of headings, lists, etc.
        quality.structure_score = self.calculate_structure_score(content);

        // Depth: information density and detail level
        quality.depth_score = self.calculate_depth_score(content);

        // Weighted average of quality factors
        (quality.readability_score * 0.2) +
        (quality.completeness_score * 0.3) +
        (quality.authority_score * 0.3) +
        (quality.structure_score * 0.1) +
        (quality.depth_score * 0.1)
    }

    /// Calculate freshness/recency score
    fn calculate_freshness_score(&self, chunk: &DocumentChunk, query_analysis: &QueryAnalysis) -> f32 {
        let now = chrono::Utc::now();
        let age = now.signed_duration_since(chunk.created_at);
        
        // Base freshness score (exponential decay)
        let days_old = age.num_days() as f32;
        let mut freshness = (-days_old / 365.0).exp(); // Decay over a year

        // Adjust based on temporal context
        if let Some(temporal) = &query_analysis.temporal_context {
            freshness *= temporal.time_sensitivity;
            
            if let Some(preferred_recency) = temporal.preferred_recency {
                let preferred_days = preferred_recency.num_days() as f32;
                if days_old <= preferred_days {
                    freshness *= 1.2; // Boost for recently created content
                }
            }
        }

        freshness.clamp(0.0, 1.0)
    }

    /// Calculate context relevance score
    fn calculate_context_relevance(&self, query_analysis: &QueryAnalysis, chunk: &DocumentChunk) -> f32 {
        let mut context_score = 0.5; // Base score

        // Check metadata for context clues
        if let Ok(metadata) = serde_json::from_value::<HashMap<String, serde_json::Value>>(chunk.metadata.clone()) {
            // Topic relevance
            if let Some(topic) = metadata.get("topic") {
                if let Ok(topic_str) = serde_json::from_value::<String>(topic.clone()) {
                    context_score += self.calculate_topic_relevance(&topic_str, query_analysis) * 0.3;
                }
            }

            // Source type relevance
            if let Some(source_type) = metadata.get("source_type") {
                if let Ok(source_str) = serde_json::from_value::<String>(source_type.clone()) {
                    context_score += self.calculate_source_type_relevance(&source_str, query_analysis) * 0.2;
                }
            }
        }

        context_score.clamp(0.0, 1.0)
    }

    /// Calculate query-specific boost
    fn calculate_query_boost(&self, query_analysis: &QueryAnalysis, chunk: &DocumentChunk) -> f32 {
        let mut boost = 1.0;

        // Boost based on query intent
        match query_analysis.intent {
            QueryIntent::ProblemSolving => {
                if chunk.content.to_lowercase().contains("solution") || 
                   chunk.content.to_lowercase().contains("solve") ||
                   chunk.content.to_lowercase().contains("fix") {
                    boost *= 1.15;
                }
            },
            QueryIntent::Learning => {
                if chunk.content.to_lowercase().contains("example") ||
                   chunk.content.to_lowercase().contains("tutorial") ||
                   chunk.content.to_lowercase().contains("guide") {
                    boost *= 1.1;
                }
            },
            QueryIntent::Decision => {
                if chunk.content.to_lowercase().contains("compare") ||
                   chunk.content.to_lowercase().contains("versus") ||
                   chunk.content.to_lowercase().contains("pros and cons") {
                    boost *= 1.12;
                }
            },
            _ => {}
        }

        boost
    }

    /// Apply diversity penalty to reduce redundancy
    async fn apply_diversity_penalty(
        &self,
        mut scored_chunks: Vec<(DocumentChunk, RelevanceScore)>,
        _query_analysis: &QueryAnalysis,
    ) -> Vec<(DocumentChunk, RelevanceScore)> {
        if scored_chunks.len() <= 1 {
            return scored_chunks;
        }

        // Calculate similarity between chunks and penalize similar ones
        for i in 0..scored_chunks.len() {
            let mut max_similarity: f32 = 0.0;
            
            for j in 0..i {
                if let (Some(emb1), Some(emb2)) = (&scored_chunks[i].0.embedding, &scored_chunks[j].0.embedding) {
                    let similarity = self.calculate_cosine_similarity(emb1, emb2);
                    max_similarity = max_similarity.max(similarity);
                }
            }

            // Apply penalty based on maximum similarity to previously selected chunks
            let penalty = max_similarity * 0.3; // Up to 30% penalty
            scored_chunks[i].1.diversity_penalty = penalty;
            scored_chunks[i].1.overall_score -= penalty;
        }

        scored_chunks
    }

    /// Helper methods for scoring components
    fn classify_query_type(&self, text: &str, tokens: &[&str]) -> QueryType {
        if text.starts_with("what is") || text.starts_with("define") {
            QueryType::Factual
        } else if text.starts_with("how to") || text.starts_with("how do") {
            QueryType::Procedural
        } else if text.contains(" vs ") || text.contains(" versus ") || text.contains("compare") {
            QueryType::Comparative
        } else if text.starts_with("why") || text.starts_with("explain") {
            QueryType::Analytical
        } else if text.starts_with("find") || text.contains("document") || text.contains("page") {
            QueryType::Navigational
        } else if tokens.len() <= 5 && !text.contains('?') {
            QueryType::Conversational
        } else {
            QueryType::Factual // Default
        }
    }

    fn determine_intent(&self, _text: &str, tokens: &[&str], query_type: &QueryType) -> QueryIntent {
        match query_type {
            QueryType::Procedural => QueryIntent::ProblemSolving,
            QueryType::Comparative => QueryIntent::Decision,
            QueryType::Analytical => QueryIntent::Learning,
            QueryType::Factual => {
                if tokens.len() > 5 {
                    QueryIntent::InformationSeeking
                } else {
                    QueryIntent::Exploration
                }
            },
            _ => QueryIntent::InformationSeeking,
        }
    }

    fn assess_complexity(&self, tokens: &[&str], _query_type: &QueryType) -> QueryComplexity {
        match tokens.len() {
            1..=3 => QueryComplexity::Simple,
            4..=7 => QueryComplexity::Moderate,
            8..=15 => QueryComplexity::Complex,
            _ => QueryComplexity::Expert,
        }
    }

    fn extract_key_terms(&self, tokens: &[&str]) -> Vec<String> {
        let stop_words: std::collections::HashSet<&str> = [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with", "would",
            "what", "how", "why", "when", "where", "who"
        ].iter().cloned().collect();

        tokens.iter()
            .filter(|token| !stop_words.contains(*token) && token.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }

    fn extract_named_entities(&self, text: &str) -> Vec<NamedEntity> {
        // Simplified named entity extraction
        let mut entities = Vec::new();

        // Look for capitalized words (potential proper nouns)
        let words: Vec<&str> = text.split_whitespace().collect();
        for window in words.windows(2) {
            if let [word1, word2] = window {
                if word1.chars().next().unwrap_or('a').is_uppercase() &&
                   word2.chars().next().unwrap_or('a').is_uppercase() {
                    entities.push(NamedEntity {
                        text: format!("{} {}", word1, word2),
                        entity_type: EntityType::Organization, // Default classification
                        confidence: 0.7,
                    });
                }
            }
        }

        entities
    }

    fn extract_temporal_context(&self, text: &str) -> Option<TemporalContext> {
        let time_indicators = ["recent", "latest", "new", "current", "today", "now"];
        let has_temporal = time_indicators.iter().any(|&indicator| text.contains(indicator));

        if has_temporal {
            Some(TemporalContext {
                time_sensitivity: 0.8,
                preferred_recency: Some(chrono::Duration::days(30)),
            })
        } else {
            None
        }
    }

    // Additional helper methods for quality scoring
    fn calculate_readability(&self, content: &str) -> f32 {
        let sentences: Vec<&str> = content.split('.').collect();
        let words: Vec<&str> = content.split_whitespace().collect();
        
        if sentences.is_empty() || words.is_empty() {
            return 0.5;
        }

        let avg_sentence_length = words.len() as f32 / sentences.len() as f32;
        
        // Flesch reading ease approximation (simplified)
        let score = 206.835 - (1.015 * avg_sentence_length);
        (score / 100.0).clamp(0.0, 1.0)
    }

    fn calculate_completeness(&self, content: &str) -> f32 {
        let word_count = content.split_whitespace().count();
        
        // Score based on content length (assuming 50-500 words is optimal)
        let length_score = match word_count {
            0..=20 => 0.3,
            21..=50 => 0.6,
            51..=200 => 1.0,
            201..=500 => 0.9,
            _ => 0.7,
        };

        length_score
    }

    fn calculate_structure_score(&self, content: &str) -> f32 {
        let mut score: f32 = 0.5; // Base score

        // Check for structural elements
        if content.contains('\n') {
            score += 0.1; // Has line breaks
        }
        if content.contains("- ") || content.contains("* ") {
            score += 0.1; // Has bullet points
        }
        if content.contains('#') {
            score += 0.1; // Has headings (markdown)
        }
        if content.contains(':') && content.matches(':').count() > 2 {
            score += 0.1; // Has definitions or structured content
        }

        score.clamp(0.0, 1.0)
    }

    fn calculate_depth_score(&self, content: &str) -> f32 {
        let word_count = content.split_whitespace().count();
        let unique_words: std::collections::HashSet<&str> = content.split_whitespace().collect();
        
        if word_count == 0 {
            return 0.0;
        }

        // Vocabulary richness
        let vocabulary_richness = unique_words.len() as f32 / word_count as f32;
        
        // Information density heuristics
        let info_indicators = ["because", "therefore", "however", "although", "example", "specifically"];
        let info_density = info_indicators.iter()
            .filter(|&indicator| content.to_lowercase().contains(indicator))
            .count() as f32 / 10.0; // Normalize

        (vocabulary_richness + info_density).clamp(0.0, 1.0)
    }

    fn calculate_topic_relevance(&self, topic: &str, query_analysis: &QueryAnalysis) -> f32 {
        // Simple topic matching (can be enhanced with topic models)
        let topic_lower = topic.to_lowercase();
        let relevance = query_analysis.key_terms.iter()
            .any(|term| topic_lower.contains(&term.to_lowercase()));

        if relevance { 0.8 } else { 0.2 }
    }

    fn calculate_source_type_relevance(&self, source_type: &str, query_analysis: &QueryAnalysis) -> f32 {
        match query_analysis.intent {
            QueryIntent::Learning => {
                if source_type == "tutorial" || source_type == "guide" { 0.9 } else { 0.5 }
            },
            QueryIntent::ProblemSolving => {
                if source_type == "solution" || source_type == "troubleshooting" { 0.9 } else { 0.5 }
            },
            QueryIntent::InformationSeeking => {
                if source_type == "reference" || source_type == "documentation" { 0.9 } else { 0.5 }
            },
            _ => 0.5,
        }
    }

    fn calculate_confidence(&self, scores: &[f32]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }

        // Calculate variance to determine confidence
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f32>() / scores.len() as f32;

        // High variance = low confidence, low variance = high confidence
        let confidence = 1.0 - variance.sqrt();
        confidence.clamp(0.0, 1.0)
    }

    fn calculate_cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }

        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        (dot_product / (norm1 * norm2)).clamp(-1.0, 1.0)
    }

    fn generate_explanation(
        &self,
        query_analysis: &QueryAnalysis,
        chunk: &DocumentChunk,
        semantic_score: f32,
        keyword_score: f32,
        quality_score: f32,
        freshness_score: f32,
        context_score: f32,
        boost_multiplier: f32,
    ) -> RelevanceExplanation {
        let mut main_factors = Vec::new();
        let mut positive_signals = Vec::new();
        let mut negative_signals = Vec::new();
        let mut recommendations = Vec::new();
        let mut debug_info = HashMap::new();

        // Store debug information
        debug_info.insert("semantic_score".to_string(), semantic_score);
        debug_info.insert("keyword_score".to_string(), keyword_score);
        debug_info.insert("quality_score".to_string(), quality_score);
        debug_info.insert("freshness_score".to_string(), freshness_score);
        debug_info.insert("context_score".to_string(), context_score);
        debug_info.insert("boost_multiplier".to_string(), boost_multiplier);

        // Identify main contributing factors
        let scores = [
            ("semantic similarity", semantic_score),
            ("keyword relevance", keyword_score),
            ("document quality", quality_score),
            ("content freshness", freshness_score),
            ("context alignment", context_score),
        ];

        // Find top contributing factors
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (factor, score) in sorted_scores.iter().take(3) {
            main_factors.push(format!("{}: {:.3}", factor, score));
        }

        // Identify positive signals
        if semantic_score > 0.7 {
            positive_signals.push("Strong semantic match with query".to_string());
        }
        if keyword_score > 0.6 {
            positive_signals.push("Good keyword coverage".to_string());
        }
        if quality_score > 0.7 {
            positive_signals.push("High-quality document structure".to_string());
        }
        if boost_multiplier > 1.1 {
            positive_signals.push("Query-specific relevance boost applied".to_string());
        }

        // Identify negative signals
        if semantic_score < 0.3 {
            negative_signals.push("Low semantic similarity".to_string());
        }
        if keyword_score < 0.2 {
            negative_signals.push("Poor keyword match".to_string());
        }
        if freshness_score < 0.3 {
            negative_signals.push("Content may be outdated".to_string());
        }

        // Generate recommendations
        match query_analysis.query_type {
            QueryType::Procedural => {
                if !chunk.content.to_lowercase().contains("step") {
                    recommendations.push("Consider documents with step-by-step instructions".to_string());
                }
            },
            QueryType::Factual => {
                if quality_score < 0.5 {
                    recommendations.push("Look for more authoritative sources".to_string());
                }
            },
            _ => {}
        }

        if positive_signals.is_empty() {
            recommendations.push("Refine query terms for better matches".to_string());
        }

        RelevanceExplanation {
            main_factors,
            positive_signals,
            negative_signals,
            recommendations,
            debug_info,
        }
    }
}

/// Factory for creating relevance scorers with different configurations
pub struct RelevanceScorerFactory;

impl RelevanceScorerFactory {
    /// Create a general-purpose relevance scorer
    pub fn create_general() -> RelevanceScorer {
        RelevanceScorer::new(RelevanceConfig::default())
    }

    /// Create a semantic-focused scorer (higher weight on vector similarity)
    pub fn create_semantic_focused() -> RelevanceScorer {
        let config = RelevanceConfig {
            semantic_weight: 0.6,
            keyword_weight: 0.2,
            quality_weight: 0.1,
            freshness_weight: 0.05,
            context_weight: 0.05,
            ..Default::default()
        };
        RelevanceScorer::new(config)
    }

    /// Create a keyword-focused scorer (higher weight on exact matches)
    pub fn create_keyword_focused() -> RelevanceScorer {
        let config = RelevanceConfig {
            semantic_weight: 0.2,
            keyword_weight: 0.5,
            quality_weight: 0.15,
            freshness_weight: 0.05,
            context_weight: 0.1,
            ..Default::default()
        };
        RelevanceScorer::new(config)
    }

    /// Create a quality-focused scorer (emphasizes document quality)
    pub fn create_quality_focused() -> RelevanceScorer {
        let config = RelevanceConfig {
            semantic_weight: 0.3,
            keyword_weight: 0.25,
            quality_weight: 0.3,
            freshness_weight: 0.05,
            context_weight: 0.1,
            ..Default::default()
        };
        RelevanceScorer::new(config)
    }

    /// Create a time-sensitive scorer (emphasizes freshness)
    pub fn create_time_sensitive() -> RelevanceScorer {
        let config = RelevanceConfig {
            semantic_weight: 0.3,
            keyword_weight: 0.3,
            quality_weight: 0.1,
            freshness_weight: 0.25,
            context_weight: 0.05,
            ..Default::default()
        };
        RelevanceScorer::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    use chrono::Utc;

    fn create_test_query(text: &str) -> Query {
        Query::new(text.to_string())
    }

    fn create_test_chunk(content: &str) -> DocumentChunk {
        DocumentChunk {
            id: Uuid::new_v4(),
            document_id: Uuid::new_v4(),
            content: content.to_string(),
            chunk_index: 0,
            embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_query_analysis() {
        let scorer = RelevanceScorerFactory::create_general();
        let query = create_test_query("How to implement machine learning algorithms?");
        let analysis = scorer.analyze_query(&query);

        assert_eq!(analysis.query_type, QueryType::Procedural);
        assert_eq!(analysis.intent, QueryIntent::ProblemSolving);
        assert!(!analysis.key_terms.is_empty());
    }

    #[test]
    fn test_query_type_classification() {
        let scorer = RelevanceScorerFactory::create_general();

        let factual = create_test_query("What is machine learning?");
        let factual_analysis = scorer.analyze_query(&factual);
        assert_eq!(factual_analysis.query_type, QueryType::Factual);

        let procedural = create_test_query("How to train a neural network?");
        let procedural_analysis = scorer.analyze_query(&procedural);
        assert_eq!(procedural_analysis.query_type, QueryType::Procedural);

        let comparative = create_test_query("Python vs Java for machine learning");
        let comparative_analysis = scorer.analyze_query(&comparative);
        assert_eq!(comparative_analysis.query_type, QueryType::Comparative);
    }

    #[tokio::test]
    async fn test_relevance_calculation() {
        let scorer = RelevanceScorerFactory::create_general();
        let query = create_test_query("machine learning algorithms");
        let analysis = scorer.analyze_query(&query);
        let chunk = create_test_chunk("Machine learning algorithms are powerful tools for data analysis.");

        let relevance = scorer.calculate_relevance(&analysis, &chunk, 0.8, 0.7).await.unwrap();

        assert!(relevance.overall_score > 0.0);
        assert!(relevance.confidence > 0.0);
        assert!(!relevance.explanation.main_factors.is_empty());
    }

    #[test]
    fn test_scorer_factory() {
        let general = RelevanceScorerFactory::create_general();
        let semantic = RelevanceScorerFactory::create_semantic_focused();
        let keyword = RelevanceScorerFactory::create_keyword_focused();

        // Test different weight configurations
        assert!(semantic.config.semantic_weight > general.config.semantic_weight);
        assert!(keyword.config.keyword_weight > general.config.keyword_weight);
    }

    #[test]
    fn test_document_quality_scoring() {
        let scorer = RelevanceScorerFactory::create_quality_focused();
        
        let high_quality_content = "# Machine Learning Guide\n\nMachine learning is a method of data analysis. For example:\n\n- Supervised learning\n- Unsupervised learning\n\nTherefore, understanding these concepts is crucial.";
        let readability = scorer.calculate_readability(high_quality_content);
        let structure = scorer.calculate_structure_score(high_quality_content);
        
        assert!(readability > 0.0);
        assert!(structure > 0.5); // Should detect structural elements
    }

    #[test]
    fn test_temporal_context_extraction() {
        let scorer = RelevanceScorerFactory::create_time_sensitive();
        
        let temporal_query = create_test_query("What are the latest developments in AI?");
        let analysis = scorer.analyze_query(&temporal_query);
        
        assert!(analysis.temporal_context.is_some());
        assert!(analysis.temporal_context.unwrap().time_sensitivity > 0.5);
    }
}
