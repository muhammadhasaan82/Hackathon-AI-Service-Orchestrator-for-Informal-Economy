// ═══════════════════════════════════════════════════════════════
// CAG KV-Cache — Rust implementation via PyO3
//
// Thread-safe, memory-efficient key-value cache for preloading
// static "Golden Knowledge" into the LLM's extended context window.
// ═══════════════════════════════════════════════════════════════

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::RwLock;

/// Thread-safe KV-Cache for golden knowledge storage.
/// Uses RwLock for concurrent reads with exclusive writes.
#[pyclass]
pub struct KVCache {
    cache: RwLock<HashMap<String, String>>,
}

#[pymethods]
impl KVCache {
    #[new]
    fn new() -> Self {
        KVCache {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Store a key-value pair in the cache.
    fn set(&self, key: String, value: String) -> PyResult<()> {
        let mut cache = self.cache.write().unwrap();
        cache.insert(key, value);
        Ok(())
    }

    /// Retrieve a value by key. Returns None if not found.
    fn get(&self, key: String) -> PyResult<Option<String>> {
        let cache = self.cache.read().unwrap();
        Ok(cache.get(&key).cloned())
    }

    /// Batch insert multiple key-value pairs.
    fn batch_set(&self, entries: Vec<(String, String)>) -> PyResult<usize> {
        let mut cache = self.cache.write().unwrap();
        let count = entries.len();
        for (key, value) in entries {
            cache.insert(key, value);
        }
        Ok(count)
    }

    /// Batch retrieve multiple keys. Returns list of (key, value) for found keys.
    fn batch_get(&self, keys: Vec<String>) -> PyResult<Vec<(String, String)>> {
        let cache = self.cache.read().unwrap();
        let results: Vec<(String, String)> = keys
            .iter()
            .filter_map(|k| cache.get(k).map(|v| (k.clone(), v.clone())))
            .collect();
        Ok(results)
    }

    /// Get all entries as a list of (key, value) tuples.
    fn get_all(&self) -> PyResult<Vec<(String, String)>> {
        let cache = self.cache.read().unwrap();
        Ok(cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }

    /// Remove a key from the cache.
    fn evict(&self, key: String) -> PyResult<bool> {
        let mut cache = self.cache.write().unwrap();
        Ok(cache.remove(&key).is_some())
    }

    /// Clear the entire cache.
    fn clear(&self) -> PyResult<()> {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
        Ok(())
    }

    /// Number of entries in the cache.
    fn len(&self) -> PyResult<usize> {
        let cache = self.cache.read().unwrap();
        Ok(cache.len())
    }

    /// Build a context string from all cached entries.
    /// This is injected into the LLM's context window.
    fn build_context(&self, separator: String) -> PyResult<String> {
        let cache = self.cache.read().unwrap();
        let parts: Vec<String> = cache
            .iter()
            .map(|(k, v)| format!("[{}]: {}", k, v))
            .collect();
        Ok(parts.join(&separator))
    }
}

/// Python module definition
#[pymodule]
fn cag_cache(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KVCache>()?;
    Ok(())
}
