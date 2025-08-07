use axum::{
    http::StatusCode,
    response::{Html, Json},
    routing::get,
    Router,
};
use serde_json::json;
use tracing::info;

use crate::performance::metrics::{global_metrics, MetricValue, MetricsSummary};

/// Get all metrics as JSON
pub async fn get_metrics_json() -> Result<Json<serde_json::Value>, StatusCode> {
    let metrics = global_metrics();
    let all_metrics = metrics.get_all_metrics();
    let summary = metrics.get_metrics_summary();

    let response = json!({
        "summary": summary,
        "metrics": all_metrics
    });

    Ok(Json(response))
}

/// Get metrics summary only
pub async fn get_metrics_summary() -> Result<Json<MetricsSummary>, StatusCode> {
    let metrics = global_metrics();
    let summary = metrics.get_metrics_summary();
    
    info!("Metrics summary requested: {} total metrics", summary.total_metrics);
    Ok(Json(summary))
}

/// Get metrics in Prometheus format (simplified)
pub async fn get_metrics_prometheus() -> Result<String, StatusCode> {
    let metrics = global_metrics();
    let all_metrics = metrics.get_all_metrics();
    
    let mut prometheus_output = String::new();
    
    // Add help and type information
    prometheus_output.push_str("# HELP system_uptime_seconds System uptime in seconds\n");
    prometheus_output.push_str("# TYPE system_uptime_seconds gauge\n");
    
    prometheus_output.push_str("# HELP requests_total Total number of requests\n");
    prometheus_output.push_str("# TYPE requests_total counter\n");
    
    prometheus_output.push_str("# HELP cache_hits_total Total number of cache hits\n");
    prometheus_output.push_str("# TYPE cache_hits_total counter\n");
    
    prometheus_output.push_str("# HELP cache_misses_total Total number of cache misses\n");
    prometheus_output.push_str("# TYPE cache_misses_total counter\n");
    
    prometheus_output.push_str("# HELP active_connections Current active connections\n");
    prometheus_output.push_str("# TYPE active_connections gauge\n");
    
    prometheus_output.push_str("# HELP memory_usage_bytes Current memory usage in bytes\n");
    prometheus_output.push_str("# TYPE memory_usage_bytes gauge\n");
    
    // Output metric values
    for metric in all_metrics {
        let labels_str = if metric.labels.is_empty() {
            String::new()
        } else {
            let labels: Vec<String> = metric.labels.iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", labels.join(","))
        };
        
        prometheus_output.push_str(&format!("{}{} {} {}\n", 
            metric.name, labels_str, metric.value, metric.timestamp));
    }
    
    Ok(prometheus_output)
}

/// Get metrics as HTML page (simple dashboard)
pub async fn get_metrics_html() -> Result<Html<String>, StatusCode> {
    let metrics = global_metrics();
    let all_metrics = metrics.get_all_metrics();
    let summary = metrics.get_metrics_summary();
    
    let html = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>RustRAG Metrics Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .summary {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
        .counter {{ background-color: #e6f3ff; }}
        .gauge {{ background-color: #fff2e6; }}
        .histogram {{ background-color: #e6ffe6; }}
        .metric-name {{ font-weight: bold; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .labels {{ font-size: 0.9em; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>RustRAG Metrics Dashboard</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Metrics:</strong> {}</p>
        <p><strong>Counters:</strong> {}</p>
        <p><strong>Gauges:</strong> {}</p>
        <p><strong>Histograms:</strong> {}</p>
        <p><strong>Uptime:</strong> {:.2} seconds</p>
    </div>
    
    <h2>All Metrics</h2>
    <table>
        <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Value</th>
            <th>Labels</th>
            <th>Timestamp</th>
        </tr>
        {}
    </table>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function(){{ window.location.reload(); }}, 30000);
    </script>
</body>
</html>
"#, 
        summary.total_metrics,
        summary.counters_count,
        summary.gauges_count,
        summary.histograms_count,
        summary.uptime_seconds,
        all_metrics.iter().map(|m| {
            let type_class = match m.metric_type {
                crate::performance::metrics::MetricType::Counter => "counter",
                crate::performance::metrics::MetricType::Gauge => "gauge",
                crate::performance::metrics::MetricType::Histogram => "histogram",
            };
            
            let labels_str = if m.labels.is_empty() {
                "-".to_string()
            } else {
                m.labels.iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            
            format!(r#"<tr class="{}">
                <td>{}</td>
                <td>{:?}</td>
                <td>{:.2}</td>
                <td>{}</td>
                <td>{}</td>
            </tr>"#, type_class, m.name, m.metric_type, m.value, labels_str, m.timestamp)
        }).collect::<Vec<_>>().join("\n")
    );
    
    Ok(Html(html))
}

/// Health check endpoint with basic metrics
pub async fn health_with_metrics() -> Result<Json<serde_json::Value>, StatusCode> {
    let metrics = global_metrics();
    let summary = metrics.get_metrics_summary();
    
    let health_status = if summary.uptime_seconds > 0.0 {
        "healthy"
    } else {
        "starting"
    };
    
    let response = json!({
        "status": health_status,
        "uptime_seconds": summary.uptime_seconds,
        "metrics_count": summary.total_metrics,
        "timestamp": chrono::Utc::now().timestamp()
    });
    
    Ok(Json(response))
}

/// Create metrics router
pub fn create_metrics_router() -> Router {
    Router::new()
        .route("/metrics", get(get_metrics_json))
        .route("/metrics/summary", get(get_metrics_summary))
        .route("/metrics/prometheus", get(get_metrics_prometheus))
        .route("/metrics/dashboard", get(get_metrics_html))
        .route("/health/metrics", get(health_with_metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::metrics::global_metrics;

    #[tokio::test]
    async fn test_metrics_functionality() {
        // Initialize some test metrics
        let metrics = global_metrics();
        metrics.increment_counter("test_requests_total");
        metrics.set_gauge("test_active_connections", 5.0);
        metrics.observe_histogram("test_request_duration_seconds", 0.1);

        // Test getting all metrics
        let all_metrics = metrics.get_all_metrics();
        assert!(!all_metrics.is_empty());
        
        // Test metrics summary
        let summary = metrics.get_metrics_summary();
        assert!(summary.total_metrics > 0);
    }

    #[tokio::test]
    async fn test_prometheus_format() {
        let metrics = global_metrics();
        metrics.increment_counter("test_counter");
        
        let prometheus_text = get_metrics_prometheus().await.unwrap();
        
        assert!(prometheus_text.contains("# HELP"));
        assert!(prometheus_text.contains("# TYPE"));
        assert!(prometheus_text.contains("test_counter"));
    }

    #[tokio::test]
    async fn test_html_dashboard() {
        let metrics = global_metrics();
        metrics.increment_counter("dashboard_test_counter");
        
        let html_result = get_metrics_html().await;
        assert!(html_result.is_ok());
        
        let html = html_result.unwrap().0;
        assert!(html.contains("RustRAG Metrics Dashboard"));
        assert!(html.contains("dashboard_test_counter"));
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let health_result = health_with_metrics().await;
        assert!(health_result.is_ok());
        
        let response = health_result.unwrap().0;
        assert!(response["status"].as_str().unwrap() == "healthy");
        assert!(response["uptime_seconds"].as_f64().unwrap() >= 0.0);
    }
}
