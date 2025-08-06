use crate::core::retrieval::RankedChunk;
use crate::models::{Citation, Query, QueryResponse};
use crate::utils::Result;

#[cfg(feature = "embeddings-openai")]
use crate::utils::Error;
#[cfg(feature = "embeddings-openai")]
use async_openai::types::{CreateChatCompletionRequest, ChatCompletionRequestUserMessage, Role};
#[cfg(feature = "embeddings-openai")]
use async_openai::{Client, config::OpenAIConfig};
#[cfg(feature = "embeddings-openai")]
use std::sync::Arc;

/// Configuration for generation service
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// LLM provider (e.g., "openai", "claude", "local")
    pub provider: String,
    /// Model name to use for generation
    pub model_name: String,
    /// API key for the LLM provider
    pub api_key: Option<String>,
    /// Maximum number of tokens for the generated response
    pub max_tokens: u32,
    /// Temperature setting for generation
    pub temperature: f32,
    /// System prompt to guide the LLM
    pub system_prompt: String,
    /// Whether to include citations in the response
    pub include_citations: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model_name: "gpt-4".to_string(),
            api_key: None,
            max_tokens: 1000,
            temperature: 0.7,
            system_prompt: "You are an AI assistant. Answer the user\'s question based on the provided context. If the answer is not in the context, say you don\'t know.".to_string(),
            include_citations: true,
        }
    }
}

/// Generation service for creating responses using LLMs
pub struct GenerationService {
    config: GenerationConfig,
    #[cfg(feature = "embeddings-openai")]
    client: Arc<Client<OpenAIConfig>>,
}

impl GenerationService {
    /// Create a new generation service
    pub fn new() -> Result<Self> {
        Self::with_config(GenerationConfig::default())
    }

    /// Create a new generation service with custom configuration
    pub fn with_config(config: GenerationConfig) -> Result<Self> {
        #[cfg(feature = "embeddings-openai")]
        let client = {
            let client_config = if let Some(api_key) = &config.api_key {
                OpenAIConfig::new().with_api_key(api_key)
            } else {
                OpenAIConfig::new()
            };
            Client::with_config(client_config)
        };
        
        Ok(Self {
            config,
            #[cfg(feature = "embeddings-openai")]
            client: Arc::new(client),
        })
    }

    /// Generate response for a query using retrieved chunks
    pub async fn generate_response(
        &self,
        query: &Query,
        retrieved_chunks: Vec<RankedChunk>,
    ) -> Result<QueryResponse> {
        let start_time = std::time::Instant::now();

        // Assemble context from retrieved chunks
        let context = self.assemble_context(&retrieved_chunks);

        // Generate prompt for the LLM
        let prompt = self.build_prompt(&query.text, &context);

        // Generate answer using the LLM
        let answer = self.generate_answer(&prompt).await?;

        // Create citations if enabled
        let citations = if self.config.include_citations {
            self.create_citations(&retrieved_chunks)
        } else {
            Vec::new()
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(QueryResponse {
            id: uuid::Uuid::new_v4(),
            query_id: query.id,
            answer,
            retrieved_chunks: retrieved_chunks.into_iter().map(|r| r.into()).collect(),
            citations,
            processing_time_ms: processing_time,
            created_at: chrono::Utc::now(),
        })
    }

    /// Assemble context from retrieved chunks
    fn assemble_context(&self, chunks: &[RankedChunk]) -> String {
        if chunks.is_empty() {
            return "No relevant information found in the knowledge base.".to_string();
        }

        let mut context = String::new();
        context.push_str("Based on the following information:\n\n");
        
        for (i, chunk) in chunks.iter().enumerate() {
            context.push_str(&format!(
                "[{}] {}\n\n",
                i + 1,
                chunk.chunk.content
            ));
        }
        
        context
    }

    /// Build prompt for the LLM
    fn build_prompt(&self, query: &str, context: &str) -> String {
        format!(
            "{}\n\n{}\n\nQuestion: {}\n\nAnswer:",
            self.config.system_prompt,
            context,
            query
        )
    }

    /// Generate answer using LLM
    async fn generate_answer(&self, prompt: &str) -> Result<String> {
        match self.config.provider.as_str() {
            "openai" => self.generate_with_openai(prompt).await,
            "mock" | _ => Ok(self.generate_mock_answer(prompt)),
        }
    }

    /// Generate answer using OpenAI
    #[cfg(feature = "embeddings-openai")]
    async fn generate_with_openai(&self, prompt: &str) -> Result<String> {
        use async_openai::types::{ChatCompletionRequestMessage, CreateChatCompletionRequestArgs};
        
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.config.model_name)
            .messages(vec![
                ChatCompletionRequestMessage::System(
                    async_openai::types::ChatCompletionRequestSystemMessage {
                        content: self.config.system_prompt.clone(),
                        name: None,
                    }
                ),
                ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(prompt.to_string()),
                        name: None,
                    }
                ),
            ])
            .temperature(self.config.temperature as f64)
            .max_tokens(self.config.max_tokens as u16)
            .build()
            .map_err(|e| Error::llm_api(format!("Failed to build OpenAI request: {}", e)))?;

        let response = self.client
            .chat()
            .create(request)
            .await
            .map_err(|e| Error::llm_api(format!("OpenAI API error: {}", e)))?;

        let answer = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .ok_or_else(|| Error::llm_api("No content in OpenAI response".to_string()))?
            .clone();

        tracing::info!("Generated answer using OpenAI with {} tokens", 
                      response.usage.map(|u| u.total_tokens).unwrap_or(0));

        Ok(answer)
    }
    
    /// Generate answer using OpenAI (fallback when feature not enabled)
    #[cfg(not(feature = "embeddings-openai"))]
    async fn generate_with_openai(&self, prompt: &str) -> Result<String> {
        tracing::warn!("OpenAI feature not enabled, falling back to mock generation");
        Ok(self.generate_mock_answer(prompt))
    }

    /// Generate mock answer for testing/fallback
    fn generate_mock_answer(&self, prompt: &str) -> String {
        tracing::info!("Generating mock answer for prompt of length {}", prompt.len());
        
        // Simple mock implementation
        if prompt.to_lowercase().contains("what is") {
            "Based on the provided context, this appears to be a definition or explanation query. \n\nThe information available in the knowledge base suggests various aspects of the topic you're asking about. For a more specific answer, please refer to the context above.".to_string()
        } else if prompt.to_lowercase().contains("how to") {
            "Based on the provided context, here are the key steps and information available:\n\n1. Review the relevant documentation or resources\n2. Follow the best practices outlined in the context\n3. Consider the specific requirements of your use case\n\nFor detailed implementation guidance, please refer to the context provided above.".to_string()
        } else {
            "Based on the available context and information in the knowledge base, I can provide relevant insights related to your query. Please refer to the numbered references above for specific details and supporting information.".to_string()
        }
    }

    /// Create citations from retrieved chunks
    fn create_citations(&self, chunks: &[RankedChunk]) -> Vec<Citation> {
        chunks
            .iter()
            .enumerate()
            .map(|(_i, chunk)| Citation {
                document_id: chunk.chunk.document_id,
                document_title: format!("Document {}", chunk.chunk.document_id), // TODO: Get actual title from metadata
                chunk_id: chunk.chunk.id,
                page_number: None, // TODO: Extract from metadata if available
                excerpt: chunk.chunk.content
                    .chars()
                    .take(200)
                    .collect::<String>()
                    .trim()
                    .to_string() + if chunk.chunk.content.len() > 200 { "..." } else { "" },
            })
            .collect()
    }

    /// Get generation configuration
    pub fn get_config(&self) -> &GenerationConfig {
        &self.config
    }

    /// Update generation configuration
    pub fn update_config(&mut self, config: GenerationConfig) -> Result<()> {
        // Recreate client if API key changed
        #[cfg(feature = "embeddings-openai")]
        if config.api_key != self.config.api_key {
            let client_config = if let Some(api_key) = &config.api_key {
                OpenAIConfig::new().with_api_key(api_key)
            } else {
                OpenAIConfig::new()
            };
            self.client = Arc::new(Client::with_config(client_config));
        }
        
        self.config = config;
        Ok(())
    }
}

// Implement conversion from RankedChunk to RetrievedChunk for compatibility
impl From<RankedChunk> for crate::models::RetrievedChunk {
    fn from(ranked: RankedChunk) -> Self {
        crate::models::RetrievedChunk {
            id: ranked.chunk.id,
            chunk_id: ranked.chunk.id,
            document_id: ranked.chunk.document_id,
            document_title: format!("Document {}", ranked.chunk.document_id), // TODO: Get from metadata
            content: ranked.chunk.content,
            similarity_score: ranked.combined_score,
            chunk_index: ranked.chunk.chunk_index,
            embedding: ranked.chunk.embedding,
            metadata: ranked.chunk.metadata,
            created_at: ranked.chunk.created_at,
        }
    }
}

impl Default for GenerationService {
    fn default() -> Self {
        Self::new().unwrap()
    }
}
