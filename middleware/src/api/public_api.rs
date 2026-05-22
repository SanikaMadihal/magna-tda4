// =============================================================================
// Magna Middleware — Public API
// =============================================================================
//! Thread-safe, state-aware public API for the middleware.
//!
//! All methods are guarded by a mutex and check the lifecycle state before
//! performing any operation.  This is the **only** entry point that external
//! code should use.

use crate::inference::engine_manager::EngineManager;
use crate::inference::traits::{EngineInfo, TensorBuffer};
use crate::lifecycle::state_manager::{State, StateManager};
#[cfg(feature = "metrics")]
use crate::metrics::{MetricsCollector, MetricsSnapshot};
use crate::postprocess::classifier::{Classifier, InferenceResult};
use crate::preprocess::imagenet;
#[cfg(any(test, feature = "metrics"))]
use crate::utils::errors::Precision;
use crate::utils::errors::{MiddlewareConfig, MiddlewareError, MiddlewareResult};
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use tracing::{error, info};
// `debug!` only used in metrics-gated timing logs, so conditionally imported
#[cfg(feature = "metrics")]
use tracing::debug;

// ---------------------------------------------------------------------------
// Middleware (inner state)
// ---------------------------------------------------------------------------

struct Inner {
    state: StateManager,
    config: MiddlewareConfig,
    engine_manager: Option<EngineManager>,
    classifier: Classifier,
    #[cfg(feature = "metrics")]
    metrics: Mutex<MetricsCollector>,
    last_prediction: Mutex<Option<InferenceResult>>,
}

// ---------------------------------------------------------------------------
// Middleware (public handle)
// ---------------------------------------------------------------------------

/// Thread-safe handle to the Magna middleware.
///
/// Clone this handle freely — all clones share the same state via `Arc<Mutex>`.
#[derive(Clone)]
pub struct Middleware {
    inner: Arc<RwLock<Inner>>,
}

impl Middleware {
    // == Lifecycle ===========================================================

    /// Create a new middleware instance (starts in `Uninitialized`).
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                state: StateManager::new(),
                config: MiddlewareConfig::default(),
                engine_manager: None,
                classifier: Classifier::new(false),
                #[cfg(feature = "metrics")]
                metrics: Mutex::new(MetricsCollector::new(Precision::FP32)),
                last_prediction: Mutex::new(None),
            })),
        }
    }

    /// Initialize the middleware with the given configuration.
    pub fn initialize(&self, config: MiddlewareConfig) -> MiddlewareResult<()> {
        let mut inner = self.inner.write();

        inner.state.transition_to(State::Initialized)?;

        if config.debug {
            info!(debug = true, "Debug logging enabled");
        }

        // Create engine manager using compile-time backend.
        let engine_manager = EngineManager::new(&config.backend).inspect_err(|_e| {
            inner.state.set_error();
        })?;

        // Load labels if provided.
        let classifier = if let Some(ref labels_path) = config.labels_path {
            Classifier::load_labels_from_file(labels_path, true)?
        } else {
            Classifier::new(true)
        };

        // Update internal state.
        #[cfg(feature = "metrics")]
        {
            inner.metrics = Mutex::new(MetricsCollector::new(config.precision));
        }
        inner.engine_manager = Some(engine_manager);
        inner.classifier = classifier;
        inner.config = config;

        info!(backend = %inner.config.backend, "Middleware initialized");
        Ok(())
    }

    /// Load a precompiled engine artifact.
    pub fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        let mut inner = self.inner.write();

        inner.state.require_at_least(State::Initialized)?;

        let em = inner
            .engine_manager
            .as_mut()
            .ok_or_else(|| MiddlewareError::Internal("Engine manager not initialized".into()))?;

        let info = em.load_engine(path).map_err(|e| {
            error!(engine_path = %path, error = %e, "Engine load failed");
            inner.state.set_error();
            e
        })?;

        // Update metrics with model memory usage
        #[cfg(feature = "metrics")]
        inner.metrics.lock().set_memory_usage(info.memory_bytes);

        // Transition through EngineLoaded → Ready.
        if inner.state.current() == State::Initialized || inner.state.current() == State::Ready {
            inner.state.transition_to(State::EngineLoaded)?;
        }
        inner.state.transition_to(State::Ready)?;

        info!(
            engine  = %info.name,
            inputs  = info.inputs.len(),
            outputs = info.outputs.len(),
            "Engine loaded"
        );
        Ok(info)
    }

    // == Inference ===========================================================

    /// Run inference on generic input tensors.
    pub fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        let inner = self.inner.read();
        inner.state.require(State::Ready)?;

        #[cfg(feature = "metrics")]
        inner.metrics.lock().start_timing();

        let em = inner
            .engine_manager
            .as_ref()
            .ok_or(MiddlewareError::EngineNotLoaded)?;
        let output_res = em.infer(inputs);

        match output_res {
            Ok(output) => {
                #[cfg(feature = "metrics")]
                {
                    let elapsed = inner.metrics.lock().stop_timing();
                    debug!(
                        latency_ms = elapsed.as_secs_f64() * 1000.0,
                        "Inference complete"
                    );
                }
                Ok(output)
            }
            Err(e) => {
                error!(error = %e, "Inference failed");
                #[cfg(feature = "metrics")]
                inner.metrics.lock().stop_timing();
                Err(e)
            }
        }
    }

    /// End-to-end: load image → preprocess → infer → postprocess.
    pub fn infer_from_image(&self, image_path: &str) -> MiddlewareResult<InferenceResult> {
        let precision = {
            let inner = self.inner.read();
            inner.config.precision
        };

        let mut input_tensor = imagenet::preprocess_image_file(image_path, precision)?;
        input_tensor.name = "input".into();
        let output = self.infer(&[input_tensor])?;

        let inner = self.inner.read();
        let result = InferenceResult::from_outputs(&output, Some(&inner.classifier))?;

        info!(result = %result, "Classification complete");
        *inner.last_prediction.lock() = Some(result.clone());

        Ok(result)
    }

    /// Run inference and return a generic result (raw tensor + optional classification).
    pub fn infer_generic(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<InferenceResult> {
        let output = self.infer(inputs)?;

        let inner = self.inner.read();
        let result = InferenceResult::from_outputs(&output, Some(&inner.classifier))?;
        *inner.last_prediction.lock() = Some(result.clone());

        Ok(result)
    }

    // == Queries =============================================================

    /// Get the result of the last inference run.
    pub fn get_last_prediction(&self) -> MiddlewareResult<InferenceResult> {
        let inner = self.inner.read();
        let x = inner.last_prediction.lock().clone();
        x.ok_or(MiddlewareError::Internal("No prediction available".into()))
    }

    /// Get a snapshot of collected metrics.
    ///
    /// Returns an empty (all-zero) snapshot when the `metrics` feature is
    /// disabled.
    #[cfg(feature = "metrics")]
    pub fn get_metrics(&self) -> MetricsSnapshot {
        let inner = self.inner.read();
        let snapshot = inner.metrics.lock().snapshot();
        snapshot
    }

    /// Stub returned when `metrics` feature is disabled.
    #[cfg(not(feature = "metrics"))]
    pub fn get_metrics(&self) -> crate::metrics::MetricsSnapshot {
        crate::metrics::MetricsSnapshot::default()
    }

    /// Get the current lifecycle state.
    pub fn state(&self) -> State {
        let inner = self.inner.read();
        inner.state.current()
    }

    /// Get info about the currently loaded engine, if any.
    pub fn engine_info(&self) -> Option<EngineInfo> {
        let inner = self.inner.read();
        inner
            .engine_manager
            .as_ref()
            .and_then(|em| em.engine_info_owned())
    }

    /// Get the current configuration.
    pub fn config(&self) -> MiddlewareConfig {
        let inner = self.inner.read();
        inner.config.clone()
    }

    /// Get the backend name.
    pub fn backend_name(&self) -> String {
        let inner = self.inner.read();
        inner
            .engine_manager
            .as_ref()
            .map(|em| em.backend_name().to_string())
            .unwrap_or_else(|| "none".into())
    }

    // == Shutdown ============================================================

    /// Gracefully shut down the middleware, releasing all resources.
    pub fn shutdown(&self) -> MiddlewareResult<()> {
        let mut inner = self.inner.write();

        if let Some(ref mut em) = inner.engine_manager {
            em.release().ok(); // Best-effort release.
        }
        inner.engine_manager = None;
        *inner.last_prediction.lock() = None;
        #[cfg(feature = "metrics")]
        inner.metrics.lock().reset();

        inner.state.transition_to(State::Uninitialized)?;
        info!("Middleware shut down");
        Ok(())
    }
}

impl Default for Middleware {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Middleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("Middleware")
            .field("state", &inner.state.current())
            .field("backend", &inner.config.backend)
            .field("precision", &inner.config.precision)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_middleware_is_uninitialized() {
        let mw = Middleware::new();
        assert_eq!(mw.state(), State::Uninitialized);
    }

    #[test]
    fn initialize_with_simulated_backend() {
        let mw = Middleware::new();
        let config = MiddlewareConfig {
            backend: "simulated".into(),
            precision: Precision::FP32,
            ..Default::default()
        };
        mw.initialize(config).unwrap();
        assert_eq!(mw.state(), State::Initialized);
    }

    #[test]
    fn full_lifecycle() {
        use std::io::Write;

        let mw = Middleware::new();
        mw.initialize(MiddlewareConfig {
            backend: "simulated".into(),
            precision: Precision::FP32,
            ..Default::default()
        })
        .unwrap();

        let mut tmp = tempfile::Builder::new()
            .suffix(".engine")
            .tempfile()
            .unwrap();
        tmp.write_all(b"FAKE_ENGINE").unwrap();
        let path = tmp.path().to_str().unwrap();

        mw.load_engine(path).unwrap();
        assert_eq!(mw.state(), State::Ready);

        let input = TensorBuffer {
            name: "input".to_string(),
            data: vec![0u8; 3 * 224 * 224 * 4],
            shape: vec![1, 3, 224, 224],
            precision: Precision::FP32,
        };
        let output = mw.infer(&[input]).unwrap();
        assert_eq!(output[0].shape, vec![1, 1000]);

        #[cfg(feature = "metrics")]
        {
            let metrics = mw.get_metrics();
            assert_eq!(metrics.inference_count, 1);
        }

        mw.shutdown().unwrap();
        assert_eq!(mw.state(), State::Uninitialized);
    }

    #[test]
    fn infer_before_ready_fails() {
        let mw = Middleware::new();
        mw.initialize(MiddlewareConfig {
            backend: "simulated".into(),
            ..Default::default()
        })
        .unwrap();

        let input = TensorBuffer::from_f32("input", &[0.0; 6], vec![1, 2, 3]);
        assert!(mw.infer(&[input]).is_err());
    }

    #[test]
    fn thread_safe_clone() {
        let mw = Middleware::new();
        let mw2 = mw.clone();

        let handle = std::thread::spawn(move || {
            mw2.initialize(MiddlewareConfig {
                backend: "simulated".into(),
                ..Default::default()
            })
            .unwrap();
        });

        handle.join().unwrap();
        assert_eq!(mw.state(), State::Initialized);
    }
}
