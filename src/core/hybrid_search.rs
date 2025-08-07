use crate::models::{Query, DocumentChunk};
use crate::utils::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Configuration for hybrid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// Weight for vector similarity (0.0-1.0)
    pub vector_weight: f32,
    /// Weight for keyword matching (0.0-1.0)
    pub keyword_weight: f32,
    /// BM25 k1 parameter (term frequency saturation)
    pub bm25_k1: f32,
    /// BM25 b parameter (length normalization)
    pub bm25_b: f32,
    /// Minimum term frequency to consider
    pub min_term_frequency: usize,
    /// Whether to use stemming for keyword search
    pub use_stemming: bool,
    /// Whether to remove stop words
    pub remove_stop_words: bool,
    /// Custom stop words list (if empty, uses default English stop words)
    pub custom_stop_words: Vec<String>,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.7,
            keyword_weight: 0.3,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            min_term_frequency: 1,
            use_stemming: false, // Keep simple for now
            remove_stop_words: true,
            custom_stop_words: Vec::new(),
        }
    }
}

/// Document statistics for BM25 calculation
#[derive(Debug, Clone)]
pub struct DocumentStats {
    /// Total number of documents in collection
    pub total_documents: usize,
    /// Average document length in the collection
    pub average_document_length: f32,
    /// Term frequencies across all documents
    pub term_document_frequencies: HashMap<String, usize>,
}

/// Processed query terms with statistics
#[derive(Debug, Clone)]
pub struct ProcessedQuery {
    pub original_text: String,
    pub terms: Vec<String>,
    pub term_frequencies: HashMap<String, usize>,
}

/// Keyword search result with BM25 score
#[derive(Debug, Clone)]
pub struct KeywordSearchResult {
    pub chunk_id: uuid::Uuid,
    pub bm25_score: f32,
    pub term_matches: HashMap<String, usize>,
    pub explanation: String,
}

/// Hybrid search scorer combining vector and keyword search
pub struct HybridSearchScorer {
    config: HybridSearchConfig,
    document_stats: Arc<DocumentStats>,
    stop_words: std::collections::HashSet<String>,
}

impl HybridSearchScorer {
    /// Create a new hybrid search scorer
    pub fn new(config: HybridSearchConfig, document_stats: Arc<DocumentStats>) -> Self {
        let stop_words = if config.custom_stop_words.is_empty() {
            Self::default_english_stop_words()
        } else {
            config.custom_stop_words.iter().cloned().collect()
        };

        Self {
            config,
            document_stats,
            stop_words,
        }
    }

    /// Process query text into structured terms
    pub fn process_query(&self, query_text: &str) -> ProcessedQuery {
        let normalized_text = query_text.to_lowercase();
        let raw_terms: Vec<String> = normalized_text
            .split_whitespace()
            .map(|term| self.clean_term(term))
            .filter(|term| !term.is_empty())
            .collect();

        // Remove stop words if enabled
        let filtered_terms: Vec<String> = if self.config.remove_stop_words {
            raw_terms
                .into_iter()
                .filter(|term| !self.stop_words.contains(term))
                .collect()
        } else {
            raw_terms
        };

        // Count term frequencies
        let mut term_frequencies = HashMap::new();
        for term in &filtered_terms {
            *term_frequencies.entry(term.clone()).or_insert(0) += 1;
        }

        ProcessedQuery {
            original_text: query_text.to_string(),
            terms: filtered_terms,
            term_frequencies,
        }
    }

    /// Calculate BM25 score for a document chunk
    pub fn calculate_bm25_score(
        &self,
        query: &ProcessedQuery,
        chunk: &DocumentChunk,
    ) -> KeywordSearchResult {
        let chunk_text = chunk.content.to_lowercase();
        let chunk_terms: Vec<String> = chunk_text
            .split_whitespace()
            .map(|term| self.clean_term(term))
            .filter(|term| !term.is_empty())
            .collect();

        let chunk_length = chunk_terms.len() as f32;
        let mut chunk_term_frequencies = HashMap::new();
        let mut term_matches = HashMap::new();

        // Count term frequencies in the document
        for term in &chunk_terms {
            *chunk_term_frequencies.entry(term.clone()).or_insert(0) += 1;
        }

        let mut bm25_score = 0.0;
        let mut explanations = Vec::new();

        for (query_term, query_tf) in &query.term_frequencies {
            let doc_tf = *chunk_term_frequencies.get(query_term).unwrap_or(&0);
            
            if doc_tf >= self.config.min_term_frequency {
                term_matches.insert(query_term.clone(), doc_tf);

                // Calculate IDF (Inverse Document Frequency)
                let df = self.document_stats
                    .term_document_frequencies
                    .get(query_term)
                    .copied()
                    .unwrap_or(1); // Smoothing for unseen terms

                let idf = ((self.document_stats.total_documents as f32 - df as f32 + 0.5) / 
                          (df as f32 + 0.5)).ln();

                // Calculate BM25 term score
                let tf_component = (doc_tf as f32 * (self.config.bm25_k1 + 1.0)) /
                    (doc_tf as f32 + self.config.bm25_k1 * (1.0 - self.config.bm25_b + 
                     self.config.bm25_b * (chunk_length / self.document_stats.average_document_length)));

                let term_score = idf * tf_component * (*query_tf as f32);
                bm25_score += term_score;

                explanations.push(format!(
                    "{}: tf={}, df={}, idf={:.3}, score={:.3}",
                    query_term, doc_tf, df, idf, term_score
                ));
            }
        }

        let explanation = if explanations.is_empty() {
            "No matching terms found".to_string()
        } else {
            format!("BM25={:.3} [{}]", bm25_score, explanations.join(", "))
        };

        KeywordSearchResult {
            chunk_id: chunk.id,
            bm25_score,
            term_matches,
            explanation,
        }
    }

    /// Calculate combined hybrid score
    pub fn calculate_hybrid_score(
        &self,
        vector_score: f32,
        keyword_result: &KeywordSearchResult,
    ) -> f32 {
        // Normalize BM25 score (simple approach - can be enhanced)
        let normalized_bm25 = (keyword_result.bm25_score / 10.0).tanh(); // Sigmoid-like normalization

        (vector_score * self.config.vector_weight) + 
        (normalized_bm25 * self.config.keyword_weight)
    }

    /// Perform hybrid search on a set of document chunks
    pub async fn search_chunks(
        &self,
        query: &Query,
        chunks: &[DocumentChunk],
        query_embedding: &[f32],
    ) -> Result<Vec<HybridSearchResult>> {
        info!("Performing hybrid search on {} chunks", chunks.len());
        
        let processed_query = self.process_query(&query.text);
        let mut results = Vec::new();

        for chunk in chunks {
            // Calculate vector similarity
            let vector_score = if let Some(chunk_embedding) = &chunk.embedding {
                calculate_cosine_similarity(query_embedding, chunk_embedding)
            } else {
                0.0
            };

            // Calculate BM25 keyword score
            let keyword_result = self.calculate_bm25_score(&processed_query, chunk);

            // Calculate combined hybrid score
            let hybrid_score = self.calculate_hybrid_score(vector_score, &keyword_result);

            let explanation = format!(
                "Hybrid: {:.3} (Vector: {:.3} × {:.1}% + Keyword: {:.3} × {:.1}%) | {}",
                hybrid_score,
                vector_score,
                self.config.vector_weight * 100.0,
                keyword_result.bm25_score,
                self.config.keyword_weight * 100.0,
                keyword_result.explanation
            );

            results.push(HybridSearchResult {
                chunk: chunk.clone(),
                vector_score,
                keyword_score: keyword_result.bm25_score,
                hybrid_score,
                term_matches: keyword_result.term_matches,
                explanation,
            });
        }

        // Sort by hybrid score (descending)
        results.sort_by(|a, b| b.hybrid_score.partial_cmp(&a.hybrid_score).unwrap());

        debug!("Hybrid search completed, top score: {:.3}", 
               results.first().map(|r| r.hybrid_score).unwrap_or(0.0));

        Ok(results)
    }

    /// Clean and normalize a term
    fn clean_term(&self, term: &str) -> String {
        term.chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
            .collect::<String>()
    }

    /// Default English stop words
    fn default_english_stop_words() -> std::collections::HashSet<String> {
        [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", 
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with", "would",
            "i", "you", "we", "they", "them", "their", "this", "these", "those", "what", "which",
            "who", "when", "where", "why", "how", "do", "does", "did", "can", "could", "should",
            "would", "may", "might", "must", "shall", "will", "have", "had", "been", "being"
        ]
        .iter()
        .map(|&s| s.to_string())
        .collect()
    }
}

/// Result of hybrid search combining vector and keyword scores
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub chunk: DocumentChunk,
    pub vector_score: f32,
    pub keyword_score: f32,
    pub hybrid_score: f32,
    pub term_matches: HashMap<String, usize>,
    pub explanation: String,
}

/// Calculate cosine similarity between two vectors
pub fn calculate_cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
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

/// Build document statistics for BM25 calculation
pub async fn build_document_stats(chunks: &[DocumentChunk]) -> DocumentStats {
    let mut term_document_frequencies = HashMap::new();
    let mut total_length = 0;

    for chunk in chunks {
        let chunk_text = chunk.content.to_lowercase();
        let terms: std::collections::HashSet<String> = chunk_text
            .split_whitespace()
            .map(|term| term.chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect::<String>())
            .filter(|term| !term.is_empty())
            .collect();

        total_length += terms.len();

        // Count unique terms per document for IDF calculation
        for term in terms {
            *term_document_frequencies.entry(term).or_insert(0) += 1;
        }
    }

    let average_document_length = if chunks.is_empty() {
        0.0
    } else {
        total_length as f32 / chunks.len() as f32
    };

    DocumentStats {
        total_documents: chunks.len(),
        average_document_length,
        term_document_frequencies,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    use chrono::Utc;

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

    #[tokio::test]
    async fn test_hybrid_search_scoring() {
        let chunks = vec![
            create_test_chunk("The quick brown fox jumps over the lazy dog"),
            create_test_chunk("Machine learning algorithms are powerful tools"),
            create_test_chunk("Natural language processing enables text analysis"),
        ];

        let stats = Arc::new(build_document_stats(&chunks).await);
        let config = HybridSearchConfig::default();
        let scorer = HybridSearchScorer::new(config, stats);

        let query = Query::new("machine learning algorithms".to_string());
        let query_embedding = vec![0.1, 0.2, 0.3, 0.4];

        let results = scorer.search_chunks(&query, &chunks, &query_embedding).await.unwrap();

        assert_eq!(results.len(), 3);
        // The chunk with "Machine learning algorithms" should score highest
        assert!(results[0].chunk.content.contains("Machine learning"));
    }

    #[test]
    fn test_query_processing() {
        let config = HybridSearchConfig::default();
        let stats = Arc::new(DocumentStats {
            total_documents: 10,
            average_document_length: 50.0,
            term_document_frequencies: HashMap::new(),
        });
        
        let scorer = HybridSearchScorer::new(config, stats);
        let processed = scorer.process_query("The quick brown fox");

        // Stop words should be removed
        assert!(!processed.terms.contains(&"the".to_string()));
        assert!(processed.terms.contains(&"quick".to_string()));
        assert!(processed.terms.contains(&"brown".to_string()));
        assert!(processed.terms.contains(&"fox".to_string()));
    }

    #[test]
    fn test_cosine_similarity() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        assert!((calculate_cosine_similarity(&vec1, &vec2) - 1.0).abs() < 1e-6);

        let vec3 = vec![1.0, 0.0, 0.0];
        let vec4 = vec![0.0, 1.0, 0.0];
        assert!((calculate_cosine_similarity(&vec3, &vec4) - 0.0).abs() < 1e-6);
    }
}
