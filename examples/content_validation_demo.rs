use rustrag::core::{DocumentProcessor, ContentValidator, ValidationConfig, ChunkingStrategy};
use std::collections::HashSet;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🛡️  RustRAG Content Validation & Sanitization Demo");
    println!("=================================================\n");

    // Demo 1: Basic validation and sanitization
    demo_basic_validation().await?;
    
    // Demo 2: Security validation
    demo_security_validation().await?;
    
    // Demo 3: Content quality assessment
    demo_quality_assessment().await?;
    
    // Demo 4: Custom validation configuration
    demo_custom_validation().await?;
    
    // Demo 5: Integrated document processing with validation
    demo_integrated_processing().await?;

    // Demo 6: Batch validation statistics
    demo_batch_validation().await?;

    Ok(())
}

async fn demo_basic_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Demo 1: Basic Content Validation");
    println!("----------------------------------");

    let validator = ContentValidator::new()?;

    // Test various content scenarios
    let test_cases = [
        ("Valid content", "This is a well-formed document with proper content.", Some("txt")),
        ("Empty content", "", Some("txt")),
        ("Too short", "Hi", Some("txt")),
        ("Unknown file type", "Valid content", Some("exe")),
        ("Very long line", &"A".repeat(15000), Some("txt")),
    ];

    for (name, content, file_type) in test_cases {
        println!("\n🔸 Test Case: {}", name);
        
        match validator.validate_and_sanitize(content, file_type) {
            Ok(result) => {
                println!("  ✅ Validation Status: {}", if result.is_valid { "VALID" } else { "INVALID" });
                
                if !result.errors.is_empty() {
                    println!("  ❌ Errors:");
                    for error in &result.errors {
                        println!("    - {}", error);
                    }
                }
                
                if !result.warnings.is_empty() {
                    println!("  ⚠️  Warnings:");
                    for warning in &result.warnings {
                        println!("    - {}", warning);
                    }
                }
                
                if let Some(sanitized) = &result.sanitized_content {
                    println!("  📏 Original Size: {} bytes", result.metadata.original_size);
                    println!("  📏 Sanitized Size: {} bytes", sanitized.len());
                    if sanitized.len() != result.metadata.original_size {
                        println!("  🧹 Content was sanitized!");
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Validation Error: {}", e);
            }
        }
    }

    Ok(())
}

async fn demo_security_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n🔒 Demo 2: Security Validation");
    println!("-----------------------------");

    let validator = ContentValidator::new()?;

    let security_test_cases = [
        ("Malicious Script", r#"<script>alert('XSS attack!')</script>"#, Some("html")),
        ("JavaScript URL", r#"<a href="javascript:alert('evil')">Click me</a>"#, Some("html")),
        ("VBScript", r#"<div onload="vbscript:MsgBox('Bad')">Content</div>"#, Some("html")),
        ("Event Handler", r#"<img src="x" onerror="alert('hack')">"#, Some("html")),
    ];

    for (name, content, file_type) in security_test_cases {
        println!("\n🔸 Security Test: {}", name);
        
        match validator.validate_and_sanitize(content, file_type) {
            Ok(result) => {
                println!("  🛡️  Security Status: {}", if result.is_valid { "SAFE" } else { "BLOCKED" });
                
                if !result.metadata.suspicious_patterns_found.is_empty() {
                    println!("  🚨 Suspicious Patterns Found:");
                    for pattern in &result.metadata.suspicious_patterns_found {
                        println!("    - {}", pattern);
                    }
                }
                
                if let Some(sanitized) = &result.sanitized_content {
                    if sanitized != content {
                        println!("  🧹 Sanitized Content: {}", sanitized);
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Security Check Error: {}", e);
            }
        }
    }

    Ok(())
}

async fn demo_quality_assessment() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n📊 Demo 3: Content Quality Assessment");
    println!("------------------------------------");

    let validator = ContentValidator::new()?;

    let quality_test_cases = [
        ("High Quality", "This is a well-written document with diverse vocabulary, proper sentence structure, and meaningful content that provides value to readers.", Some("txt")),
        ("Repetitive Content", "word word word word word word word word word word word word word word word", Some("txt")),
        ("Short Sentences", "Hi. Ok. Yes. No. Maybe. Sure. Fine. Good. Bad.", Some("txt")),
        ("Long Sentence", &format!("This is an extremely long sentence that goes on and on without proper punctuation and structure which makes it very difficult to read and understand and reduces the overall quality of the document {}", "and it continues ".repeat(50)), Some("txt")),
        ("Low Alpha Ratio", "123 456 789 !@# $%^ &*() 123 456 789", Some("txt")),
    ];

    for (name, content, file_type) in quality_test_cases {
        println!("\n🔸 Quality Test: {}", name);
        
        match validator.validate_and_sanitize(content, file_type) {
            Ok(result) => {
                if let Some(quality_score) = result.metadata.content_quality_score {
                    println!("  📈 Quality Score: {:.1}/100", quality_score);
                    
                    let quality_rating = match quality_score as i32 {
                        90..=100 => "Excellent",
                        80..=89 => "Good",
                        70..=79 => "Fair",
                        60..=69 => "Poor",
                        _ => "Very Poor",
                    };
                    println!("  🏆 Quality Rating: {}", quality_rating);
                }
                
                let quality_warnings: Vec<&String> = result.warnings
                    .iter()
                    .filter(|w| w.starts_with("Content quality:"))
                    .collect();
                    
                if !quality_warnings.is_empty() {
                    println!("  ⚠️  Quality Issues:");
                    for warning in quality_warnings {
                        println!("    - {}", warning.strip_prefix("Content quality: ").unwrap_or(warning));
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Quality Assessment Error: {}", e);
            }
        }
    }

    Ok(())
}

async fn demo_custom_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n⚙️  Demo 4: Custom Validation Configuration");
    println!("------------------------------------------");

    // Create a custom validation configuration
    let mut custom_config = ValidationConfig {
        max_document_size: 5000,  // Smaller limit
        min_document_size: 5,     // More lenient minimum
        allow_empty_content: true, // Allow empty content
        sanitize_sensitive_data: true,
        normalize_whitespace: true,
        remove_control_chars: true,
        validate_encoding: true,
        max_word_repetition: 10,  // More strict repetition limit
        allowed_file_types: ["txt", "md", "custom"].iter().map(|s| s.to_string()).collect(),
        blocked_patterns: vec![
            r"FORBIDDEN".to_string(),
            r"SECRET_DATA_\d+".to_string(),
        ],
        ..Default::default()
    };

    let custom_validator = ContentValidator::with_config(custom_config)?;

    let custom_test_cases = [
        ("Empty content (allowed)", "", Some("txt")),
        ("Custom file type", "Custom content", Some("custom")),
        ("Blocked pattern", "This contains FORBIDDEN content", Some("txt")),
        ("Secret data pattern", "User SECRET_DATA_12345 found", Some("txt")),
        ("Excessive repetition", "test test test test test test test test test test test", Some("txt")),
    ];

    for (name, content, file_type) in custom_test_cases {
        println!("\n🔸 Custom Test: {}", name);
        
        match custom_validator.validate_and_sanitize(content, file_type) {
            Ok(result) => {
                println!("  ✅ Status: {}", if result.is_valid { "VALID" } else { "INVALID" });
                
                if !result.errors.is_empty() {
                    for error in &result.errors {
                        println!("  ❌ {}", error);
                    }
                }
                
                if !result.warnings.is_empty() {
                    for warning in &result.warnings {
                        println!("  ⚠️  {}", warning);
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Error: {}", e);
            }
        }
    }

    Ok(())
}

async fn demo_integrated_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n🔄 Demo 5: Integrated Document Processing with Validation");
    println!("--------------------------------------------------------");

    let processor = DocumentProcessor::new_with_config(300, 50);  // Smaller chunks for demo

    // Test content with various issues
    let test_content = r#"# Document Title

This is a sample document with some issues:

- My SSN is 123-45-6789
- Contact me at john@example.com
- Phone: 555-123-4567

<script>alert('This should be removed')</script>

Some    extra    whitespace     and control characters.

This document contains    repetitive words    repetitive words    repetitive words."#;

    println!("📄 Processing document with validation enabled...");
    
    match processor.process_document_with_validation(
        test_content.to_string(),
        "Sample Document".to_string(),
        Some("md"),
        ChunkingStrategy::Semantic,
        None, // Use default validation config
    ).await {
        Ok((document, chunks, validation_result)) => {
            println!("  ✅ Document processed successfully!");
            println!("  📊 Document ID: {}", document.id);
            println!("  📝 Title: {}", document.title);
            println!("  📏 Original Content: {} chars", validation_result.metadata.original_size);
            
            if let Some(sanitized_size) = validation_result.metadata.sanitized_size {
                println!("  🧹 Sanitized Content: {} chars", sanitized_size);
                println!("  📉 Size Reduction: {} chars", 
                    validation_result.metadata.original_size as i32 - sanitized_size as i32);
            }
            
            println!("  📦 Generated Chunks: {}", chunks.len());
            
            println!("  🛡️  Validation Results:");
            println!("    - Valid: {}", validation_result.is_valid);
            println!("    - Warnings: {}", validation_result.warnings.len());
            println!("    - Errors: {}", validation_result.errors.len());
            
            if let Some(quality_score) = validation_result.metadata.content_quality_score {
                println!("    - Quality Score: {:.1}/100", quality_score);
            }
            
            if !validation_result.warnings.is_empty() {
                println!("  ⚠️  Validation Warnings:");
                for warning in &validation_result.warnings {
                    println!("    - {}", warning);
                }
            }
        }
        Err(e) => {
            println!("  ❌ Processing failed: {}", e);
        }
    }

    Ok(())
}

async fn demo_batch_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n📊 Demo 6: Batch Validation Statistics");
    println!("-------------------------------------");

    let processor = DocumentProcessor::new();

    let document_batch = vec![
        ("This is a high-quality document with excellent content and proper structure.".to_string(), Some("txt".to_string())),
        ("word word word word word".to_string(), Some("txt".to_string())),
        ("<script>alert('xss')</script>".to_string(), Some("html".to_string())),
        ("My SSN is 123-45-6789 and credit card is 1234-5678-9012-3456".to_string(), Some("txt".to_string())),
        ("".to_string(), Some("txt".to_string())),
        ("A".repeat(50).to_string(), Some("txt".to_string())),
        ("Normal document content here.".to_string(), Some("txt".to_string())),
    ];

    println!("📋 Validating batch of {} documents...", document_batch.len());

    match processor.batch_validate_documents(&document_batch) {
        Ok(stats) => {
            println!("  📊 Batch Validation Statistics:");
            println!("    - Total Documents: {}", stats.total_documents);
            println!("    - Valid Documents: {}", stats.valid_documents);
            println!("    - Invalid Documents: {}", stats.invalid_documents);
            println!("    - Success Rate: {:.1}%", 
                (stats.valid_documents as f32 / stats.total_documents as f32) * 100.0);
            println!("    - Total Warnings: {}", stats.total_warnings);
            println!("    - Total Errors: {}", stats.total_errors);
            println!("    - Original Size: {} bytes", stats.total_original_size);
            println!("    - Sanitized Size: {} bytes", stats.total_sanitized_size);
            println!("    - Size Reduction: {} bytes", 
                stats.total_original_size as i32 - stats.total_sanitized_size as i32);
            
            if stats.average_quality_score > 0.0 {
                println!("    - Average Quality Score: {:.1}/100", stats.average_quality_score);
            }
        }
        Err(e) => {
            println!("  ❌ Batch validation failed: {}", e);
        }
    }

    Ok(())
}
