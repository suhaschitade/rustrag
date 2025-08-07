use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Metric types supported by the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
}

/// Individual metric value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub name: String,
    pub metric_type: MetricType,
    pub value: f64,
    pub timestamp: u64,
    pub labels: HashMap<String, String>,
}

/// Histogram bucket for tracking value distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    pub le: f64,  // "less than or equal"
    pub count: u64,
}

/// Histogram metric data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    pub buckets: Vec<HistogramBucket>,
    pub sum: f64,
    pub count: u64,
}

/// Counter metric - monotonically increasing values
#[derive(Debug, Clone, Default)]
pub struct Counter {
    value: f64,
    labels: HashMap<String, String>,
}

impl Counter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_labels(labels: HashMap<String, String>) -> Self {
        Self { value: 0.0, labels }
    }

    pub fn increment(&mut self) {
        self.value += 1.0;
    }

    pub fn add(&mut self, value: f64) {
        self.value += value;
    }

    pub fn get(&self) -> f64 {
        self.value
    }

    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }
}

/// Gauge metric - can go up and down
#[derive(Debug, Clone, Default)]
pub struct Gauge {
    value: f64,
    labels: HashMap<String, String>,
}

impl Gauge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_labels(labels: HashMap<String, String>) -> Self {
        Self { value: 0.0, labels }
    }

    pub fn set(&mut self, value: f64) {
        self.value = value;
    }

    pub fn increment(&mut self) {
        self.value += 1.0;
    }

    pub fn decrement(&mut self) {
        self.value -= 1.0;
    }

    pub fn add(&mut self, value: f64) {
        self.value += value;
    }

    pub fn get(&self) -> f64 {
        self.value
    }

    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }
}

/// Histogram metric - tracks distribution of values
#[derive(Debug, Clone)]
pub struct Histogram {
    buckets: Vec<f64>,
    counts: Vec<u64>,
    sum: f64,
    count: u64,
    labels: HashMap<String, String>,
}

impl Histogram {
    pub fn new(buckets: Vec<f64>) -> Self {
        let bucket_count = buckets.len();
        Self {
            buckets,
            counts: vec![0; bucket_count],
            sum: 0.0,
            count: 0,
            labels: HashMap::new(),
        }
    }

    pub fn with_labels(buckets: Vec<f64>, labels: HashMap<String, String>) -> Self {
        let bucket_count = buckets.len();
        Self {
            buckets,
            counts: vec![0; bucket_count],
            sum: 0.0,
            count: 0,
            labels,
        }
    }

    pub fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;

        for (i, &bucket) in self.buckets.iter().enumerate() {
            if value <= bucket {
                self.counts[i] += 1;
            }
        }
    }

    pub fn get_data(&self) -> HistogramData {
        let buckets = self.buckets.iter()
            .zip(self.counts.iter())
            .map(|(&le, &count)| HistogramBucket { le, count })
            .collect();

        HistogramData {
            buckets,
            sum: self.sum,
            count: self.count,
        }
    }

    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }
}

/// Main metrics registry
#[derive(Debug, Clone)]
pub struct MetricsRegistry {
    counters: Arc<RwLock<HashMap<String, Counter>>>,
    gauges: Arc<RwLock<HashMap<String, Gauge>>>,
    histograms: Arc<RwLock<HashMap<String, Histogram>>>,
    start_time: Instant,
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsRegistry {
    pub fn new() -> Self {
        let registry = Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        };

        // Initialize system metrics
        registry.init_system_metrics();
        registry
    }

    fn init_system_metrics(&self) {
        // Initialize basic system counters
        self.register_counter("system_uptime_seconds", HashMap::new());
        self.register_counter("requests_total", HashMap::new());
        self.register_counter("cache_hits_total", HashMap::new());
        self.register_counter("cache_misses_total", HashMap::new());
        
        // Initialize basic gauges
        self.register_gauge("active_connections", HashMap::new());
        self.register_gauge("memory_usage_bytes", HashMap::new());
        
        // Initialize histograms with default buckets
        let default_buckets = vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0];
        self.register_histogram("request_duration_seconds", default_buckets, HashMap::new());
        
        info!("Initialized metrics registry with system metrics");
    }

    pub fn register_counter(&self, name: &str, labels: HashMap<String, String>) {
        let mut counters = self.counters.write().unwrap();
        counters.insert(name.to_string(), Counter::with_labels(labels));
        debug!("Registered counter: {}", name);
    }

    pub fn register_gauge(&self, name: &str, labels: HashMap<String, String>) {
        let mut gauges = self.gauges.write().unwrap();
        gauges.insert(name.to_string(), Gauge::with_labels(labels));
        debug!("Registered gauge: {}", name);
    }

    pub fn register_histogram(&self, name: &str, buckets: Vec<f64>, labels: HashMap<String, String>) {
        let mut histograms = self.histograms.write().unwrap();
        histograms.insert(name.to_string(), Histogram::with_labels(buckets, labels));
        debug!("Registered histogram: {}", name);
    }

    pub fn increment_counter(&self, name: &str) {
        if let Ok(mut counters) = self.counters.write() {
            if let Some(counter) = counters.get_mut(name) {
                counter.increment();
            }
        }
    }

    pub fn add_to_counter(&self, name: &str, value: f64) {
        if let Ok(mut counters) = self.counters.write() {
            if let Some(counter) = counters.get_mut(name) {
                counter.add(value);
            }
        }
    }

    pub fn set_gauge(&self, name: &str, value: f64) {
        if let Ok(mut gauges) = self.gauges.write() {
            if let Some(gauge) = gauges.get_mut(name) {
                gauge.set(value);
            }
        }
    }

    pub fn increment_gauge(&self, name: &str) {
        if let Ok(mut gauges) = self.gauges.write() {
            if let Some(gauge) = gauges.get_mut(name) {
                gauge.increment();
            }
        }
    }

    pub fn decrement_gauge(&self, name: &str) {
        if let Ok(mut gauges) = self.gauges.write() {
            if let Some(gauge) = gauges.get_mut(name) {
                gauge.decrement();
            }
        }
    }

    pub fn observe_histogram(&self, name: &str, value: f64) {
        if let Ok(mut histograms) = self.histograms.write() {
            if let Some(histogram) = histograms.get_mut(name) {
                histogram.observe(value);
            }
        }
    }

    pub fn record_duration<F, R>(&self, histogram_name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed().as_secs_f64();
        self.observe_histogram(histogram_name, duration);
        result
    }

    pub fn get_all_metrics(&self) -> Vec<MetricValue> {
        let mut metrics = Vec::new();
        let current_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Update uptime
        self.set_gauge("system_uptime_seconds", self.start_time.elapsed().as_secs_f64());

        // Collect counters
        if let Ok(counters) = self.counters.read() {
            for (name, counter) in counters.iter() {
                metrics.push(MetricValue {
                    name: name.clone(),
                    metric_type: MetricType::Counter,
                    value: counter.get(),
                    timestamp: current_timestamp,
                    labels: counter.labels().clone(),
                });
            }
        }

        // Collect gauges
        if let Ok(gauges) = self.gauges.read() {
            for (name, gauge) in gauges.iter() {
                metrics.push(MetricValue {
                    name: name.clone(),
                    metric_type: MetricType::Gauge,
                    value: gauge.get(),
                    timestamp: current_timestamp,
                    labels: gauge.labels().clone(),
                });
            }
        }

        // Collect histograms (simplified - just return count)
        if let Ok(histograms) = self.histograms.read() {
            for (name, histogram) in histograms.iter() {
                let data = histogram.get_data();
                metrics.push(MetricValue {
                    name: format!("{}_count", name),
                    metric_type: MetricType::Histogram,
                    value: data.count as f64,
                    timestamp: current_timestamp,
                    labels: histogram.labels().clone(),
                });
                metrics.push(MetricValue {
                    name: format!("{}_sum", name),
                    metric_type: MetricType::Histogram,
                    value: data.sum,
                    timestamp: current_timestamp,
                    labels: histogram.labels().clone(),
                });
            }
        }

        metrics
    }

    pub fn get_metrics_summary(&self) -> MetricsSummary {
        let counters_count = self.counters.read().map(|c| c.len()).unwrap_or(0);
        let gauges_count = self.gauges.read().map(|g| g.len()).unwrap_or(0);
        let histograms_count = self.histograms.read().map(|h| h.len()).unwrap_or(0);

        MetricsSummary {
            total_metrics: counters_count + gauges_count + histograms_count,
            counters_count,
            gauges_count,
            histograms_count,
            uptime_seconds: self.start_time.elapsed().as_secs_f64(),
        }
    }
}

/// Summary of metrics registry state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_metrics: usize,
    pub counters_count: usize,
    pub gauges_count: usize,
    pub histograms_count: usize,
    pub uptime_seconds: f64,
}

/// Global metrics registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<MetricsRegistry> = std::sync::OnceLock::new();

/// Get or initialize the global metrics registry
pub fn global_metrics() -> &'static MetricsRegistry {
    GLOBAL_REGISTRY.get_or_init(|| {
        info!("Initializing global metrics registry");
        MetricsRegistry::new()
    })
}

/// Convenience macros for common metric operations
#[macro_export]
macro_rules! increment_counter {
    ($name:expr) => {
        $crate::performance::metrics::global_metrics().increment_counter($name)
    };
}

#[macro_export]
macro_rules! set_gauge {
    ($name:expr, $value:expr) => {
        $crate::performance::metrics::global_metrics().set_gauge($name, $value)
    };
}

#[macro_export]
macro_rules! observe_histogram {
    ($name:expr, $value:expr) => {
        $crate::performance::metrics::global_metrics().observe_histogram($name, $value)
    };
}

#[macro_export]
macro_rules! time_histogram {
    ($name:expr, $body:block) => {
        $crate::performance::metrics::global_metrics().record_duration($name, || $body)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_counter() {
        let mut counter = Counter::new();
        assert_eq!(counter.get(), 0.0);
        
        counter.increment();
        assert_eq!(counter.get(), 1.0);
        
        counter.add(5.0);
        assert_eq!(counter.get(), 6.0);
    }

    #[test]
    fn test_gauge() {
        let mut gauge = Gauge::new();
        assert_eq!(gauge.get(), 0.0);
        
        gauge.set(10.0);
        assert_eq!(gauge.get(), 10.0);
        
        gauge.increment();
        assert_eq!(gauge.get(), 11.0);
        
        gauge.decrement();
        assert_eq!(gauge.get(), 10.0);
    }

    #[test]
    fn test_histogram() {
        let mut histogram = Histogram::new(vec![1.0, 5.0, 10.0]);
        
        histogram.observe(0.5);
        histogram.observe(3.0);
        histogram.observe(7.0);
        
        let data = histogram.get_data();
        assert_eq!(data.count, 3);
        assert_eq!(data.sum, 10.5);
        
        // Check bucket counts
        assert_eq!(data.buckets[0].count, 1); // <= 1.0
        assert_eq!(data.buckets[1].count, 2); // <= 5.0
        assert_eq!(data.buckets[2].count, 3); // <= 10.0
    }

    #[test]
    fn test_metrics_registry() {
        let registry = MetricsRegistry::new();
        
        // Test counter operations
        registry.increment_counter("test_counter");
        registry.add_to_counter("test_counter", 5.0);
        
        // Test gauge operations
        registry.set_gauge("test_gauge", 42.0);
        registry.increment_gauge("test_gauge");
        
        // Test histogram operations
        registry.observe_histogram("request_duration_seconds", 0.1);
        
        let metrics = registry.get_all_metrics();
        assert!(!metrics.is_empty());
        
        let summary = registry.get_metrics_summary();
        assert!(summary.total_metrics > 0);
    }

    #[test]
    fn test_record_duration() {
        let registry = MetricsRegistry::new();
        
        let result = registry.record_duration("test_duration", || {
            thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        
        // Verify the histogram was updated
        let metrics = registry.get_all_metrics();
        let duration_count = metrics.iter()
            .find(|m| m.name == "test_duration_count")
            .map(|m| m.value)
            .unwrap_or(0.0);
        
        assert!(duration_count > 0.0);
    }

    #[test]
    fn test_global_metrics() {
        let metrics = global_metrics();
        
        metrics.increment_counter("global_test_counter");
        metrics.set_gauge("global_test_gauge", 100.0);
        
        let all_metrics = metrics.get_all_metrics();
        assert!(!all_metrics.is_empty());
    }
}
