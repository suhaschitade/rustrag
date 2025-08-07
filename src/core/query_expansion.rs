use crate::models::{Query, QueryOptions};
use crate::utils::{Result, Error};
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for query expansion and refinement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExpansionConfig {
    /// Enable synonym expansion
    pub enable_synonyms: bool,
    /// Enable semantic expansion using related terms
    pub enable_semantic_expansion: bool,
    /// Enable query refinement (fixing grammar, structure)
    pub enable_refinement: bool,
    /// Maximum number of expanded terms to add
    pub max_expanded_terms: usize,
    /// Minimum similarity threshold for related terms
    pub semantic_threshold: f32,
    /// Enable domain-specific expansion
    pub enable_domain_expansion: bool,
    /// Enable query negation detection and handling
    pub enable_negation_handling: bool,
    /// Enable query intent refinement
    pub enable_intent_refinement: bool,
    /// Weight for original terms vs expanded terms
    pub original_term_weight: f32,
    /// Weight for expanded terms
    pub expanded_term_weight: f32,
}

impl Default for QueryExpansionConfig {
    fn default() -> Self {
        Self {
            enable_synonyms: true,
            enable_semantic_expansion: true,
            enable_refinement: true,
            max_expanded_terms: 10,
            semantic_threshold: 0.7,
            enable_domain_expansion: true,
            enable_negation_handling: true,
            enable_intent_refinement: true,
            original_term_weight: 1.0,
            expanded_term_weight: 0.6,
        }
    }
}

/// Result of query expansion process
#[derive(Debug, Clone)]
pub struct ExpansionResult {
    /// Original query text
    pub original_query: String,
    /// Expanded query with additional terms
    pub expanded_query: String,
    /// Refined query with improved structure
    pub refined_query: String,
    /// List of added synonyms
    pub synonyms: Vec<String>,
    /// List of semantically related terms
    pub semantic_terms: Vec<String>,
    /// Detected negations and how they're handled
    pub negations: Vec<NegationInfo>,
    /// Term weights for ranking
    pub term_weights: HashMap<String, f32>,
    /// Alternative query formulations
    pub alternatives: Vec<String>,
    /// Expansion confidence score
    pub confidence: f32,
}

/// Information about detected negations in query
#[derive(Debug, Clone)]
pub struct NegationInfo {
    /// The negated term or phrase
    pub negated_term: String,
    /// Position in original query
    pub position: usize,
    /// How the negation is handled
    pub handling_strategy: NegationHandling,
}

/// Strategies for handling negations in queries
#[derive(Debug, Clone)]
pub enum NegationHandling {
    /// Remove the negated term from positive search
    RemoveFromPositive,
    /// Create explicit negative filter
    ExplicitNegativeFilter,
    /// Rephrase to positive equivalent
    RephraseToPositive(String),
    /// Keep as context information
    KeepAsContext,
}

/// Query expansion service
pub struct QueryExpansionService {
    config: QueryExpansionConfig,
    synonym_dictionary: Arc<SynonymDictionary>,
    domain_knowledge: Arc<DomainKnowledgeBase>,
    semantic_relations: Arc<SemanticRelations>,
}

/// Synonym dictionary with domain-specific mappings
#[derive(Debug)]
pub struct SynonymDictionary {
    /// General synonyms mapping
    general_synonyms: HashMap<String, Vec<String>>,
    /// Technical domain synonyms
    technical_synonyms: HashMap<String, Vec<String>>,
    /// Business domain synonyms
    business_synonyms: HashMap<String, Vec<String>>,
    /// Academic domain synonyms
    academic_synonyms: HashMap<String, Vec<String>>,
}

/// Domain-specific knowledge base for expansion
#[derive(Debug)]
pub struct DomainKnowledgeBase {
    /// Technology stack mappings (e.g., "React" -> ["JavaScript", "frontend", "component"])
    tech_mappings: HashMap<String, Vec<String>>,
    /// Concept hierarchies (e.g., "ML" -> ["machine learning", "AI", "algorithms"])
    concept_hierarchies: HashMap<String, Vec<String>>,
    /// Acronym expansions
    acronyms: HashMap<String, String>,
    /// Common misspellings and corrections
    spell_corrections: HashMap<String, String>,
}

/// Semantic relationship mappings
#[derive(Debug)]
pub struct SemanticRelations {
    /// Hypernyms (more general terms)
    hypernyms: HashMap<String, Vec<String>>,
    /// Hyponyms (more specific terms)
    hyponyms: HashMap<String, Vec<String>>,
    /// Related concepts
    related_concepts: HashMap<String, Vec<String>>,
    /// Part-of relationships
    meronyms: HashMap<String, Vec<String>>,
}

impl QueryExpansionService {
    /// Create new query expansion service with default configuration
    pub fn new() -> Self {
        Self::with_config(QueryExpansionConfig::default())
    }

    /// Create query expansion service with custom configuration
    pub fn with_config(config: QueryExpansionConfig) -> Self {
        let synonym_dictionary = Arc::new(SynonymDictionary::new());
        let domain_knowledge = Arc::new(DomainKnowledgeBase::new());
        let semantic_relations = Arc::new(SemanticRelations::new());

        Self {
            config,
            synonym_dictionary,
            domain_knowledge,
            semantic_relations,
        }
    }

    /// Expand and refine a query
    pub async fn expand_query(&self, query: &str) -> Result<ExpansionResult> {
        tracing::info!("Expanding query: {}", query);

        let mut result = ExpansionResult {
            original_query: query.to_string(),
            expanded_query: query.to_string(),
            refined_query: query.to_string(),
            synonyms: Vec::new(),
            semantic_terms: Vec::new(),
            negations: Vec::new(),
            term_weights: HashMap::new(),
            alternatives: Vec::new(),
            confidence: 0.5,
        };

        // Step 1: Detect and handle negations
        if self.config.enable_negation_handling {
            result.negations = self.detect_negations(query);
            result.refined_query = self.handle_negations(&result.refined_query, &result.negations);
        }

        // Step 2: Spell correction and normalization
        result.refined_query = self.apply_spell_corrections(&result.refined_query);

        // Step 3: Expand acronyms
        result.refined_query = self.expand_acronyms(&result.refined_query);

        // Step 4: Add synonyms
        if self.config.enable_synonyms {
            let synonyms = self.find_synonyms(&result.refined_query);
            result.synonyms = synonyms.clone();
            result.expanded_query = self.add_terms_to_query(&result.expanded_query, &synonyms);
        }

        // Step 5: Add semantic expansions
        if self.config.enable_semantic_expansion {
            let semantic_terms = self.find_semantic_terms(&result.refined_query);
            result.semantic_terms = semantic_terms.clone();
            result.expanded_query = self.add_terms_to_query(&result.expanded_query, &semantic_terms);
        }

        // Step 6: Domain-specific expansion
        if self.config.enable_domain_expansion {
            let domain_terms = self.find_domain_terms(&result.refined_query);
            result.expanded_query = self.add_terms_to_query(&result.expanded_query, &domain_terms);
        }

        // Step 7: Calculate term weights
        result.term_weights = self.calculate_term_weights(&result);

        // Step 8: Generate alternative formulations
        result.alternatives = self.generate_alternatives(&result.refined_query);

        // Step 9: Calculate confidence
        result.confidence = self.calculate_expansion_confidence(&result);

        tracing::info!("Query expansion completed. Original: '{}', Expanded: '{}'", 
                      result.original_query, result.expanded_query);

        Ok(result)
    }

    /// Detect negations in the query
    fn detect_negations(&self, query: &str) -> Vec<NegationInfo> {
        let mut negations = Vec::new();
        let query_lower = query.to_lowercase();

        // Negation patterns
        let negation_patterns = [
            ("not ", 4),
            ("don't ", 6),
            ("doesn't ", 8),
            ("didn't ", 7),
            ("won't ", 6),
            ("wouldn't ", 9),
            ("can't ", 6),
            ("cannot ", 7),
            ("shouldn't ", 10),
            ("isn't ", 6),
            ("aren't ", 7),
            ("wasn't ", 7),
            ("weren't ", 8),
            ("without ", 8),
            ("exclude ", 8),
            ("except ", 7),
            ("no ", 3),
        ];

        for (pattern, length) in negation_patterns {
            if let Some(pos) = query_lower.find(pattern) {
                // Find the term being negated
                let after_negation = &query_lower[pos + length..];
                if let Some(space_pos) = after_negation.find(' ') {
                    let negated_term = after_negation[..space_pos].trim().to_string();
                    if !negated_term.is_empty() {
                        negations.push(NegationInfo {
                            negated_term: negated_term.clone(),
                            position: pos,
                            handling_strategy: self.determine_negation_strategy(&negated_term),
                        });
                    }
                }
            }
        }

        negations
    }

    /// Determine how to handle a specific negation
    fn determine_negation_strategy(&self, negated_term: &str) -> NegationHandling {
        // For demo purposes, use simple rules
        // In production, this could use ML models or more sophisticated rules
        
        if negated_term.len() < 3 {
            NegationHandling::KeepAsContext
        } else if negated_term == "error" {
            NegationHandling::RephraseToPositive("success".to_string())
        } else if negated_term == "problem" {
            NegationHandling::RephraseToPositive("solution".to_string())
        } else {
            NegationHandling::ExplicitNegativeFilter
        }
    }

    /// Handle negations in the query
    fn handle_negations(&self, query: &str, negations: &[NegationInfo]) -> String {
        let mut refined = query.to_string();

        for negation in negations {
            match &negation.handling_strategy {
                NegationHandling::RephraseToPositive(positive_term) => {
                    refined = refined.replace(&negation.negated_term, positive_term);
                }
                NegationHandling::RemoveFromPositive => {
                    // Remove negation patterns but keep for filtering
                    refined = refined.replace(&format!("not {}", negation.negated_term), "");
                    refined = refined.replace(&format!("no {}", negation.negated_term), "");
                }
                _ => {
                    // Keep as-is for other strategies
                }
            }
        }

        // Clean up extra spaces
        refined.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Apply spell corrections to the query
    fn apply_spell_corrections(&self, query: &str) -> String {
        let mut corrected = query.to_string();
        
        for (misspelling, correction) in &self.domain_knowledge.spell_corrections {
            // Use word boundaries to avoid partial matches
            let pattern = format!(r"\b{}\b", regex::escape(misspelling));
            if let Ok(regex) = regex::Regex::new(&pattern) {
                corrected = regex.replace_all(&corrected, correction).to_string();
            }
        }

        corrected
    }

    /// Expand acronyms in the query
    fn expand_acronyms(&self, query: &str) -> String {
        let mut expanded = query.to_string();

        for (acronym, expansion) in &self.domain_knowledge.acronyms {
            // Check for exact word matches
            let pattern = format!(r"\b{}\b", regex::escape(acronym));
            if let Ok(regex) = regex::Regex::new(&pattern) {
                expanded = regex.replace_all(&expanded, &format!("{} {}", acronym, expansion)).to_string();
            }
        }

        expanded
    }

    /// Find synonyms for terms in the query
    fn find_synonyms(&self, query: &str) -> Vec<String> {
        let mut synonyms = Vec::new();
        let words: Vec<&str> = query.split_whitespace().collect();

        for word in words {
            let word_lower = word.to_lowercase();
            
            // Check general synonyms
            if let Some(word_synonyms) = self.synonym_dictionary.general_synonyms.get(&word_lower) {
                for synonym in word_synonyms {
                    if !synonyms.contains(synonym) && synonyms.len() < self.config.max_expanded_terms {
                        synonyms.push(synonym.clone());
                    }
                }
            }

            // Check technical synonyms
            if let Some(tech_synonyms) = self.synonym_dictionary.technical_synonyms.get(&word_lower) {
                for synonym in tech_synonyms {
                    if !synonyms.contains(synonym) && synonyms.len() < self.config.max_expanded_terms {
                        synonyms.push(synonym.clone());
                    }
                }
            }
        }

        synonyms
    }

    /// Find semantically related terms
    fn find_semantic_terms(&self, query: &str) -> Vec<String> {
        let mut semantic_terms = Vec::new();
        let words: Vec<&str> = query.split_whitespace().collect();

        for word in words {
            let word_lower = word.to_lowercase();

            // Add hypernyms (more general terms)
            if let Some(hypernyms) = self.semantic_relations.hypernyms.get(&word_lower) {
                for hypernym in hypernyms {
                    if !semantic_terms.contains(hypernym) && semantic_terms.len() < self.config.max_expanded_terms {
                        semantic_terms.push(hypernym.clone());
                    }
                }
            }

            // Add related concepts
            if let Some(related) = self.semantic_relations.related_concepts.get(&word_lower) {
                for concept in related {
                    if !semantic_terms.contains(concept) && semantic_terms.len() < self.config.max_expanded_terms {
                        semantic_terms.push(concept.clone());
                    }
                }
            }
        }

        semantic_terms
    }

    /// Find domain-specific terms
    fn find_domain_terms(&self, query: &str) -> Vec<String> {
        let mut domain_terms = Vec::new();
        let words: Vec<&str> = query.split_whitespace().collect();

        for word in words {
            let word_lower = word.to_lowercase();

            // Check tech mappings
            if let Some(tech_terms) = self.domain_knowledge.tech_mappings.get(&word_lower) {
                for term in tech_terms {
                    if !domain_terms.contains(term) && domain_terms.len() < self.config.max_expanded_terms {
                        domain_terms.push(term.clone());
                    }
                }
            }

            // Check concept hierarchies
            if let Some(concepts) = self.domain_knowledge.concept_hierarchies.get(&word_lower) {
                for concept in concepts {
                    if !domain_terms.contains(concept) && domain_terms.len() < self.config.max_expanded_terms {
                        domain_terms.push(concept.clone());
                    }
                }
            }
        }

        domain_terms
    }

    /// Add terms to the query
    fn add_terms_to_query(&self, query: &str, terms: &[String]) -> String {
        if terms.is_empty() {
            return query.to_string();
        }

        // Add terms with lower weight indicators
        let additional_terms = terms.join(" ");
        format!("{} {}", query, additional_terms)
    }

    /// Calculate weights for different terms
    fn calculate_term_weights(&self, result: &ExpansionResult) -> HashMap<String, f32> {
        let mut weights = HashMap::new();

        // Weight original terms highest
        for term in result.original_query.split_whitespace() {
            weights.insert(term.to_lowercase(), self.config.original_term_weight);
        }

        // Weight synonyms moderately
        for synonym in &result.synonyms {
            weights.insert(synonym.clone(), self.config.expanded_term_weight);
        }

        // Weight semantic terms slightly lower
        for semantic_term in &result.semantic_terms {
            weights.insert(semantic_term.clone(), self.config.expanded_term_weight * 0.8);
        }

        weights
    }

    /// Generate alternative query formulations
    fn generate_alternatives(&self, query: &str) -> Vec<String> {
        let mut alternatives = Vec::new();

        // Question to statement conversion
        if query.contains('?') {
            let statement = query.replace('?', "").trim().to_string();
            if !statement.is_empty() {
                alternatives.push(statement);
            }
        }

        // Add "how to" prefix for procedural queries
        let query_lower = query.to_lowercase();
        if query_lower.contains("implement") || query_lower.contains("create") || 
           query_lower.contains("build") || query_lower.contains("make") {
            if !query_lower.starts_with("how to") {
                alternatives.push(format!("how to {}", query));
            }
        }

        // Add "what is" prefix for definition queries
        if !query_lower.starts_with("what is") && !query.contains('?') {
            alternatives.push(format!("what is {}", query));
        }

        alternatives
    }

    /// Calculate confidence in the expansion quality
    fn calculate_expansion_confidence(&self, result: &ExpansionResult) -> f32 {
        let mut confidence = 0.5; // Base confidence

        // Boost confidence based on number of expansions found
        let total_expansions = result.synonyms.len() + result.semantic_terms.len();
        if total_expansions > 0 {
            confidence += 0.2 * (total_expansions as f32 / self.config.max_expanded_terms as f32);
        }

        // Boost confidence if negations were handled
        if !result.negations.is_empty() {
            confidence += 0.1;
        }

        // Boost confidence if alternatives were generated
        if !result.alternatives.is_empty() {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }
}

// Implementation of knowledge bases with sample data
impl SynonymDictionary {
    fn new() -> Self {
        let mut general_synonyms = HashMap::new();
        let mut technical_synonyms = HashMap::new();
        let mut business_synonyms = HashMap::new();
        let mut academic_synonyms = HashMap::new();

        // General synonyms
        general_synonyms.insert("big".to_string(), vec!["large".to_string(), "huge".to_string(), "massive".to_string()]);
        general_synonyms.insert("small".to_string(), vec!["tiny".to_string(), "little".to_string(), "minor".to_string()]);
        general_synonyms.insert("fast".to_string(), vec!["quick".to_string(), "rapid".to_string(), "speedy".to_string()]);
        general_synonyms.insert("slow".to_string(), vec!["gradual".to_string(), "leisurely".to_string()]);

        // Technical synonyms
        technical_synonyms.insert("function".to_string(), vec!["method".to_string(), "procedure".to_string(), "routine".to_string()]);
        technical_synonyms.insert("error".to_string(), vec!["exception".to_string(), "bug".to_string(), "issue".to_string()]);
        technical_synonyms.insert("database".to_string(), vec!["db".to_string(), "datastore".to_string(), "repository".to_string()]);
        technical_synonyms.insert("algorithm".to_string(), vec!["procedure".to_string(), "method".to_string(), "approach".to_string()]);
        technical_synonyms.insert("api".to_string(), vec!["interface".to_string(), "endpoint".to_string(), "service".to_string()]);

        // Business synonyms
        business_synonyms.insert("profit".to_string(), vec!["revenue".to_string(), "income".to_string(), "earnings".to_string()]);
        business_synonyms.insert("customer".to_string(), vec!["client".to_string(), "user".to_string(), "consumer".to_string()]);

        // Academic synonyms
        academic_synonyms.insert("research".to_string(), vec!["study".to_string(), "investigation".to_string(), "analysis".to_string()]);
        academic_synonyms.insert("theory".to_string(), vec!["concept".to_string(), "principle".to_string(), "framework".to_string()]);

        Self {
            general_synonyms,
            technical_synonyms,
            business_synonyms,
            academic_synonyms,
        }
    }
}

impl DomainKnowledgeBase {
    fn new() -> Self {
        let mut tech_mappings = HashMap::new();
        let mut concept_hierarchies = HashMap::new();
        let mut acronyms = HashMap::new();
        let mut spell_corrections = HashMap::new();

        // Tech mappings
        tech_mappings.insert("react".to_string(), vec!["javascript".to_string(), "frontend".to_string(), "component".to_string(), "jsx".to_string()]);
        tech_mappings.insert("python".to_string(), vec!["programming".to_string(), "language".to_string(), "script".to_string()]);
        tech_mappings.insert("rust".to_string(), vec!["systems programming".to_string(), "memory safe".to_string(), "performance".to_string()]);
        tech_mappings.insert("docker".to_string(), vec!["container".to_string(), "containerization".to_string(), "deployment".to_string()]);

        // Concept hierarchies
        concept_hierarchies.insert("ml".to_string(), vec!["machine learning".to_string(), "artificial intelligence".to_string(), "algorithms".to_string()]);
        concept_hierarchies.insert("ai".to_string(), vec!["artificial intelligence".to_string(), "machine learning".to_string(), "neural networks".to_string()]);
        concept_hierarchies.insert("db".to_string(), vec!["database".to_string(), "data storage".to_string(), "sql".to_string()]);

        // Acronyms
        acronyms.insert("ML".to_string(), "machine learning".to_string());
        acronyms.insert("AI".to_string(), "artificial intelligence".to_string());
        acronyms.insert("API".to_string(), "application programming interface".to_string());
        acronyms.insert("REST".to_string(), "representational state transfer".to_string());
        acronyms.insert("CRUD".to_string(), "create read update delete".to_string());
        acronyms.insert("SQL".to_string(), "structured query language".to_string());
        acronyms.insert("HTTP".to_string(), "hypertext transfer protocol".to_string());
        acronyms.insert("JSON".to_string(), "javascript object notation".to_string());

        // Spell corrections
        spell_corrections.insert("algoritm".to_string(), "algorithm".to_string());
        spell_corrections.insert("machien".to_string(), "machine".to_string());
        spell_corrections.insert("leanring".to_string(), "learning".to_string());
        spell_corrections.insert("databse".to_string(), "database".to_string());
        spell_corrections.insert("progrmming".to_string(), "programming".to_string());

        Self {
            tech_mappings,
            concept_hierarchies,
            acronyms,
            spell_corrections,
        }
    }
}

impl SemanticRelations {
    fn new() -> Self {
        let mut hypernyms = HashMap::new();
        let mut hyponyms = HashMap::new();
        let mut related_concepts = HashMap::new();
        let mut meronyms = HashMap::new();

        // Hypernyms (more general)
        hypernyms.insert("python".to_string(), vec!["programming language".to_string(), "language".to_string()]);
        hypernyms.insert("algorithm".to_string(), vec!["method".to_string(), "approach".to_string()]);
        hypernyms.insert("database".to_string(), vec!["storage".to_string(), "system".to_string()]);

        // Hyponyms (more specific)
        hyponyms.insert("programming".to_string(), vec!["python".to_string(), "rust".to_string(), "javascript".to_string()]);
        hyponyms.insert("database".to_string(), vec!["postgresql".to_string(), "mysql".to_string(), "mongodb".to_string()]);

        // Related concepts
        related_concepts.insert("machine learning".to_string(), vec!["data science".to_string(), "statistics".to_string(), "neural networks".to_string()]);
        related_concepts.insert("web development".to_string(), vec!["frontend".to_string(), "backend".to_string(), "full stack".to_string()]);
        related_concepts.insert("api".to_string(), vec!["rest".to_string(), "graphql".to_string(), "endpoint".to_string()]);

        // Meronyms (part-of)
        meronyms.insert("web application".to_string(), vec!["frontend".to_string(), "backend".to_string(), "database".to_string()]);
        meronyms.insert("machine learning".to_string(), vec!["training".to_string(), "inference".to_string(), "model".to_string()]);

        Self {
            hypernyms,
            hyponyms,
            related_concepts,
            meronyms,
        }
    }
}

impl Default for QueryExpansionService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_expansion() {
        let service = QueryExpansionService::new();
        let result = service.expand_query("machine learning algorithms").await.unwrap();
        
        assert_eq!(result.original_query, "machine learning algorithms");
        assert!(!result.expanded_query.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[tokio::test] 
    async fn test_negation_detection() {
        let service = QueryExpansionService::new();
        let result = service.expand_query("not an error in the code").await.unwrap();
        
        assert!(!result.negations.is_empty());
        assert_eq!(result.negations[0].negated_term, "an");
    }

    #[tokio::test]
    async fn test_acronym_expansion() {
        let service = QueryExpansionService::new();
        let result = service.expand_query("REST API documentation").await.unwrap();
        
        assert!(result.refined_query.contains("representational state transfer"));
        assert!(result.refined_query.contains("application programming interface"));
    }

    #[tokio::test] 
    async fn test_synonym_expansion() {
        let service = QueryExpansionService::new();
        let result = service.expand_query("database error").await.unwrap();
        
        assert!(!result.synonyms.is_empty());
        assert!(!result.expanded_query.is_empty());
    }

    #[tokio::test]
    async fn test_alternative_generation() {
        let service = QueryExpansionService::new();
        let result = service.expand_query("How to implement API?").await.unwrap();
        
        assert!(!result.alternatives.is_empty());
    }
}
