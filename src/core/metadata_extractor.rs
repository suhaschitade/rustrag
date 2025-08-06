use crate::models::DocumentMetadata;
use crate::utils::Result;
use chrono::{DateTime, Utc};
use std::path::Path;
use std::fs;

/// Metadata extractor for various document formats
pub struct MetadataExtractor;

impl MetadataExtractor {
    /// Extract metadata from a file path
    pub async fn extract_from_file(file_path: &Path) -> Result<DocumentMetadata> {
        let file_metadata = fs::metadata(file_path)
            .map_err(|e| crate::utils::Error::document_processing(
                format!("Failed to read file metadata: {}", e)
            ))?;

        let file_extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        let mut metadata = DocumentMetadata {
            file_type: Some(file_extension.clone()),
            file_size: Some(file_metadata.len()),
            ..Default::default()
        };

        // Set file timestamps
        if let Ok(created) = file_metadata.created() {
            metadata.created_date = Some(created.into());
        }
        if let Ok(modified) = file_metadata.modified() {
            metadata.modified_date = Some(modified.into());
        }

        // Extract format-specific metadata
        match file_extension.as_str() {
            "pdf" => Self::extract_pdf_metadata(file_path, &mut metadata).await?,
            "docx" | "doc" => Self::extract_docx_metadata(file_path, &mut metadata).await?,
            "txt" | "md" | "markdown" => Self::extract_text_metadata(file_path, &mut metadata).await?,
            "html" | "htm" => Self::extract_html_metadata(file_path, &mut metadata).await?,
            _ => Self::extract_generic_metadata(file_path, &mut metadata).await?,
        }

        Ok(metadata)
    }

    /// Extract metadata from text content
    pub fn extract_from_content(content: &str, file_type: Option<&str>) -> Result<DocumentMetadata> {
        let mut metadata = DocumentMetadata {
            word_count: Some(Self::count_words(content)),
            file_type: file_type.map(|s| s.to_string()),
            language: Some(Self::detect_language(content)),
            ..Default::default()
        };

        // Extract content-based metadata
        match file_type {
            Some("md") | Some("markdown") => Self::extract_markdown_content_metadata(content, &mut metadata)?,
            Some("html") | Some("htm") => Self::extract_html_content_metadata(content, &mut metadata)?,
            _ => Self::extract_generic_content_metadata(content, &mut metadata)?,
        }

        Ok(metadata)
    }

    /// Extract PDF-specific metadata
    async fn extract_pdf_metadata(file_path: &Path, metadata: &mut DocumentMetadata) -> Result<()> {
        // Using lopdf for PDF metadata extraction
        match lopdf::Document::load(file_path) {
            Ok(doc) => {
                // Extract PDF info dictionary
                if let Ok(info_obj) = doc.trailer.get(b"Info") {
                    if let Ok(info_ref) = info_obj.as_reference() {
                        if let Ok(info_dict) = doc.get_dictionary(info_ref) {
                    // Extract title
                    if let Ok(title) = info_dict.get(b"Title").and_then(|obj| obj.as_str()) {
                        if let Ok(title_str) = std::str::from_utf8(title) {
                            metadata.custom_fields = serde_json::json!({
                                "pdf_title": title_str
                            });
                        }
                    }

                    // Extract author
                    if let Ok(author) = info_dict.get(b"Author").and_then(|obj| obj.as_str()) {
                        if let Ok(author_str) = std::str::from_utf8(author) {
                            metadata.author = Some(author_str.to_string());
                        }
                    }

                    // Extract creation date
                    if let Ok(creation_date) = info_dict.get(b"CreationDate").and_then(|obj| obj.as_str()) {
                        if let Ok(date_str) = std::str::from_utf8(creation_date) {
                            // PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
                            if let Ok(parsed_date) = Self::parse_pdf_date(date_str) {
                                metadata.created_date = Some(parsed_date);
                            }
                        }
                    }

                    // Extract modification date
                    if let Ok(mod_date) = info_dict.get(b"ModDate").and_then(|obj| obj.as_str()) {
                        if let Ok(date_str) = std::str::from_utf8(mod_date) {
                            if let Ok(parsed_date) = Self::parse_pdf_date(date_str) {
                                metadata.modified_date = Some(parsed_date);
                            }
                        }
                    }
                        }
                    }
                }

                // Get page count
                metadata.page_count = Some(doc.get_pages().len() as u32);
            }
            Err(e) => {
                tracing::warn!("Failed to extract PDF metadata: {}", e);
                // Continue with basic metadata
            }
        }

        Ok(())
    }

    /// Extract DOCX-specific metadata
    async fn extract_docx_metadata(file_path: &Path, metadata: &mut DocumentMetadata) -> Result<()> {
        // For now, extract basic metadata from file system
        // TODO: Implement proper DOCX metadata extraction using docx-rs or similar
        tracing::info!("DOCX metadata extraction - using basic file metadata");
        
        // Estimate page count based on content length (rough approximation)
        if let Ok(content) = fs::read_to_string(file_path) {
            let word_count = Self::count_words(&content);
            metadata.word_count = Some(word_count);
            // Rough estimate: 250-300 words per page
            metadata.page_count = Some((word_count / 275).max(1));
        }

        Ok(())
    }

    /// Extract text file metadata
    async fn extract_text_metadata(file_path: &Path, metadata: &mut DocumentMetadata) -> Result<()> {
        if let Ok(content) = fs::read_to_string(file_path) {
            let word_count = Self::count_words(&content);
            metadata.word_count = Some(word_count);
            
            // For markdown files, extract frontmatter if present
            if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
                if ext.to_lowercase() == "md" || ext.to_lowercase() == "markdown" {
                    Self::extract_markdown_content_metadata(&content, metadata)?;
                }
            }
        }

        Ok(())
    }

    /// Extract HTML metadata
    async fn extract_html_metadata(file_path: &Path, metadata: &mut DocumentMetadata) -> Result<()> {
        if let Ok(content) = fs::read_to_string(file_path) {
            Self::extract_html_content_metadata(&content, metadata)?;
        }

        Ok(())
    }

    /// Extract generic metadata for unknown file types
    async fn extract_generic_metadata(file_path: &Path, metadata: &mut DocumentMetadata) -> Result<()> {
        if let Ok(content) = fs::read_to_string(file_path) {
            let word_count = Self::count_words(&content);
            metadata.word_count = Some(word_count);
            metadata.language = Some(Self::detect_language(&content));
        }

        Ok(())
    }

    /// Extract markdown-specific metadata from content
    fn extract_markdown_content_metadata(content: &str, metadata: &mut DocumentMetadata) -> Result<()> {
        // Extract YAML frontmatter if present
        if content.starts_with("---\n") {
            if let Some(end_pos) = content[4..].find("\n---\n") {
                let frontmatter = &content[4..end_pos + 4];
                
                // Parse YAML frontmatter (basic implementation)
                for line in frontmatter.lines() {
                    if let Some((key, value)) = line.split_once(':') {
                        let key = key.trim();
                        let value = value.trim().trim_matches('"').trim_matches('\'');
                        
                        match key.to_lowercase().as_str() {
                            "title" => {
                                if let Ok(mut custom) = serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(metadata.custom_fields.clone()) {
                                    custom.insert("title".to_string(), serde_json::Value::String(value.to_string()));
                                    metadata.custom_fields = serde_json::Value::Object(custom);
                                }
                            }
                            "author" => metadata.author = Some(value.to_string()),
                            "date" => {
                                if let Ok(date) = chrono::DateTime::parse_from_rfc3339(value) {
                                    metadata.created_date = Some(date.with_timezone(&Utc));
                                }
                            }
                            "tags" => {
                                let tags: Vec<String> = value
                                    .split(',')
                                    .map(|s| s.trim().to_string())
                                    .collect();
                                metadata.tags = tags;
                            }
                            "category" => metadata.category = Some(value.to_string()),
                            _ => {
                                // Store other fields in custom_fields
                                if let Ok(mut custom) = serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(metadata.custom_fields.clone()) {
                                    custom.insert(key.to_string(), serde_json::Value::String(value.to_string()));
                                    metadata.custom_fields = serde_json::Value::Object(custom);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract HTML metadata from content
    fn extract_html_content_metadata(content: &str, metadata: &mut DocumentMetadata) -> Result<()> {
        use scraper::{Html, Selector};

        let document = Html::parse_document(content);
        
        // Extract title
        if let Ok(title_selector) = Selector::parse("title") {
            if let Some(title_element) = document.select(&title_selector).next() {
                if let Ok(mut custom) = serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(metadata.custom_fields.clone()) {
                    custom.insert("html_title".to_string(), serde_json::Value::String(title_element.inner_html()));
                    metadata.custom_fields = serde_json::Value::Object(custom);
                }
            }
        }

        // Extract meta tags
        if let Ok(meta_selector) = Selector::parse("meta") {
            let mut custom_fields = serde_json::Map::new();
            
            for element in document.select(&meta_selector) {
                if let (Some(name), Some(content)) = (element.value().attr("name"), element.value().attr("content")) {
                    match name.to_lowercase().as_str() {
                        "author" => metadata.author = Some(content.to_string()),
                        "keywords" => {
                            metadata.tags = content.split(',').map(|s| s.trim().to_string()).collect();
                        }
                        "description" => {
                            custom_fields.insert("description".to_string(), serde_json::Value::String(content.to_string()));
                        }
                        _ => {
                            custom_fields.insert(name.to_string(), serde_json::Value::String(content.to_string()));
                        }
                    }
                }
            }
            
            if !custom_fields.is_empty() {
                metadata.custom_fields = serde_json::Value::Object(custom_fields);
            }
        }

        // Count words in text content
        let text_content = Self::extract_text_from_html(&document);
        metadata.word_count = Some(Self::count_words(&text_content));

        Ok(())
    }

    /// Extract generic content metadata
    fn extract_generic_content_metadata(content: &str, metadata: &mut DocumentMetadata) -> Result<()> {
        // Try to extract title from first line if it looks like a title
        let lines: Vec<&str> = content.lines().collect();
        if !lines.is_empty() {
            let first_line = lines[0].trim();
            if first_line.len() < 100 && first_line.len() > 3 {
                // Looks like a title
                if let Ok(mut custom) = serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(metadata.custom_fields.clone()) {
                    custom.insert("inferred_title".to_string(), serde_json::Value::String(first_line.to_string()));
                    metadata.custom_fields = serde_json::Value::Object(custom);
                }
            }
        }

        Ok(())
    }

    /// Count words in text
    fn count_words(text: &str) -> u32 {
        text.split_whitespace().count() as u32
    }

    /// Simple language detection (basic implementation)
    fn detect_language(text: &str) -> String {
        // This is a very basic implementation
        // In production, you'd use a proper language detection library
        
        let sample = text.chars().take(1000).collect::<String>().to_lowercase();
        
        // Check for common English words
        let english_indicators = ["the", "and", "to", "of", "a", "in", "for", "is", "on", "that"];
        let english_count = english_indicators.iter()
            .map(|&word| sample.matches(word).count())
            .sum::<usize>();

        // Check for common Spanish words
        let spanish_indicators = ["el", "de", "que", "y", "a", "en", "un", "es", "se", "no"];
        let spanish_count = spanish_indicators.iter()
            .map(|&word| sample.matches(word).count())
            .sum::<usize>();

        // Check for common French words
        let french_indicators = ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"];
        let french_count = french_indicators.iter()
            .map(|&word| sample.matches(word).count())
            .sum::<usize>();

        if english_count > spanish_count && english_count > french_count {
            "en".to_string()
        } else if spanish_count > french_count {
            "es".to_string()
        } else if french_count > 0 {
            "fr".to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Parse PDF date format
    fn parse_pdf_date(date_str: &str) -> Result<DateTime<Utc>> {
        // PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
        if date_str.starts_with("D:") && date_str.len() >= 14 {
            let date_part = &date_str[2..];
            
            if date_part.len() >= 14 {
                let year: i32 = date_part[0..4].parse()
                    .map_err(|_| crate::utils::Error::document_processing("Invalid PDF date format".to_string()))?;
                let month: u32 = date_part[4..6].parse()
                    .map_err(|_| crate::utils::Error::document_processing("Invalid PDF date format".to_string()))?;
                let day: u32 = date_part[6..8].parse()
                    .map_err(|_| crate::utils::Error::document_processing("Invalid PDF date format".to_string()))?;
                let hour: u32 = date_part[8..10].parse().unwrap_or(0);
                let minute: u32 = date_part[10..12].parse().unwrap_or(0);
                let second: u32 = date_part[12..14].parse().unwrap_or(0);

                use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
                
                let naive_date = NaiveDate::from_ymd_opt(year, month, day)
                    .ok_or_else(|| crate::utils::Error::document_processing("Invalid PDF date".to_string()))?;
                let naive_time = NaiveTime::from_hms_opt(hour, minute, second)
                    .ok_or_else(|| crate::utils::Error::document_processing("Invalid PDF time".to_string()))?;
                let naive_datetime = NaiveDateTime::new(naive_date, naive_time);
                
                Ok(naive_datetime.and_utc())
            } else {
                Err(crate::utils::Error::document_processing("PDF date too short".to_string()))
            }
        } else {
            Err(crate::utils::Error::document_processing("Invalid PDF date prefix".to_string()))
        }
    }

    /// Extract text content from HTML document
    fn extract_text_from_html(document: &scraper::Html) -> String {
        use scraper::Selector;
        
        // Remove script and style elements
        let text_selector = Selector::parse("body").unwrap_or_else(|_| Selector::parse("*").unwrap());
        
        document
            .select(&text_selector)
            .map(|element| element.text().collect::<Vec<_>>().join(" "))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_count() {
        assert_eq!(MetadataExtractor::count_words("hello world"), 2);
        assert_eq!(MetadataExtractor::count_words("  hello   world  test  "), 3);
        assert_eq!(MetadataExtractor::count_words(""), 0);
    }

    #[test]
    fn test_language_detection() {
        let english_text = "The quick brown fox jumps over the lazy dog";
        assert_eq!(MetadataExtractor::detect_language(english_text), "en");
        
        let spanish_text = "El perro come en la casa y el gato duerme";
        assert_eq!(MetadataExtractor::detect_language(spanish_text), "es");
    }

    #[test]
    fn test_pdf_date_parsing() {
        let pdf_date = "D:20240101120000";
        let result = MetadataExtractor::parse_pdf_date(pdf_date);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_extract_from_content() {
        let content = "This is a test document with some content.";
        let metadata = MetadataExtractor::extract_from_content(content, Some("txt")).unwrap();
        
        assert_eq!(metadata.word_count, Some(9));
        assert_eq!(metadata.file_type, Some("txt".to_string()));
        assert_eq!(metadata.language, Some("en".to_string()));
    }
}
