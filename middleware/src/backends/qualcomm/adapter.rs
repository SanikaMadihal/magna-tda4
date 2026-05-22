// =============================================================================
// Magna Middleware — Qualcomm Backend Adapter
// =============================================================================
//! Implements [`InferenceBackend`] for Qualcomm Snapdragon hardware using
//! SNPE (Snapdragon Neural Processing Engine) or QNN (Qualcomm AI Engine).
//!
//! When the `qualcomm` feature is enabled and the SNPE/QNN SDK is available,
//! this adapter uses the actual hardware DSP/HTP.  Otherwise it runs in
//! simulation mode.

use tracing::{debug, info};

use crate::inference::traits::{EngineInfo, InferenceBackend, TensorBuffer, TensorSpec};
use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};

// ---------------------------------------------------------------------------
// FFI declarations for QNN (linked when `qualcomm` feature is enabled)
// ---------------------------------------------------------------------------

#[cfg(feature = "qualcomm")]
extern "C" {
    /// Initialize the QNN backend for a given device (0 = HTP, 1 = CPU).
    fn qnn_backend_init(device: i32) -> *mut std::ffi::c_void;

    /// Load a precompiled model context (from .bin or serialized graph).
    fn qnn_load_context(
        backend: *mut std::ffi::c_void,
        path: *const u8,
        path_len: usize,
    ) -> *mut std::ffi::c_void;

    /// Execute inference on the loaded context.
    fn qnn_execute_graph(
        ctx: *mut std::ffi::c_void,
        input: *const u8,
        input_bytes: usize,
        output: *mut u8,
        output_bytes: usize,
    ) -> i32;

    /// Get the number of input elements.
    fn qnn_get_input_elems(ctx: *mut std::ffi::c_void) -> i32;

    /// Get the number of output elements.
    fn qnn_get_output_elems(ctx: *mut std::ffi::c_void) -> i32;

    /// Release the context and backend resources.
    fn qnn_destroy(ctx: *mut std::ffi::c_void);
    fn qnn_backend_destroy(backend: *mut std::ffi::c_void);
}

// ---------------------------------------------------------------------------
// SNPE FFI declarations (alternative to QNN)
// ---------------------------------------------------------------------------

#[cfg(feature = "qualcomm")]
extern "C" {
    fn snpe_load_dlc(path: *const u8, path_len: usize) -> *mut std::ffi::c_void;
    fn snpe_execute(
        handle: *mut std::ffi::c_void,
        input: *const u8,
        input_bytes: usize,
        output: *mut u8,
        output_bytes: usize,
    ) -> i32;
    fn snpe_destroy(handle: *mut std::ffi::c_void);
}

// ---------------------------------------------------------------------------
// QualcommAdapter
// ---------------------------------------------------------------------------

/// Qualcomm backend adapter supporting SNPE and QNN.
pub struct QualcommAdapter {
    state: parking_lot::RwLock<QualcommState>,
}

struct QualcommState {
    engine_info: Option<EngineInfo>,
    buffers_allocated: bool,
    /// Tracks which runtime is being used.
    runtime: QualcommRuntime,
    #[cfg(feature = "qualcomm")]
    backend_handle: Option<*mut std::ffi::c_void>,
    #[cfg(feature = "qualcomm")]
    context_handle: Option<*mut std::ffi::c_void>,
}

// SAFETY: QualcommState is exclusively accessed via RwLock by the adapter.
// While it holds raw FFI pointers, access to these is safely synchronized.
// QNN/SNPE engines can be safely moved between threads and shared securely.
unsafe impl Send for QualcommState {}
// SAFETY: See Send impl. RwLock ensures exclusive access to the raw pointers.
unsafe impl Sync for QualcommState {}

#[derive(Debug, Clone, Copy, PartialEq)]
enum QualcommRuntime {
    None,
    Qnn,
    Snpe,
    Simulated,
}

impl QualcommAdapter {
    pub fn new() -> Self {
        Self {
            state: parking_lot::RwLock::new(QualcommState {
                engine_info: None,
                buffers_allocated: false,
                runtime: QualcommRuntime::None,
                #[cfg(feature = "qualcomm")]
                backend_handle: None,
                #[cfg(feature = "qualcomm")]
                context_handle: None,
            }),
        }
    }

    /// Detect which Qualcomm runtime is available (QNN preferred over SNPE).
    fn detect_runtime() -> QualcommRuntime {
        #[cfg(feature = "qualcomm")]
        {
            // Prefer QNN if available
            if std::env::var("QNN_SDK_ROOT").is_ok() {
                info!(backend = "qualcomm", runtime = "qnn", "Runtime selected");
                return QualcommRuntime::Qnn;
            }
            if std::env::var("SNPE_ROOT").is_ok() {
                info!(backend = "qualcomm", runtime = "snpe", "Runtime selected");
                return QualcommRuntime::Snpe;
            }
        }
        debug!(
            backend = "qualcomm",
            mode = "simulation",
            "No Qualcomm SDK found"
        );
        QualcommRuntime::Simulated
    }
}

impl Default for QualcommAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for QualcommAdapter {
    fn backend_name(&self) -> &str {
        "qualcomm"
    }

    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        let mut state = self.state.write();
        info!(backend = "qualcomm", engine_path = %path, "Loading model");

        if !std::path::Path::new(path).exists() {
            return Err(MiddlewareError::EngineLoadFailed(format!(
                "Model file not found: {}",
                path
            )));
        }

        state.runtime = Self::detect_runtime();

        #[cfg(feature = "qualcomm")]
        {
            match state.runtime {
                QualcommRuntime::Qnn => {
                    // SAFETY: `qnn_backend_init` is called with 0 for HTP device.
                    // The C library handles global state initialization internally.
                    let backend = unsafe { qnn_backend_init(0) }; // 0 = HTP
                    if backend.is_null() {
                        return Err(MiddlewareError::EngineLoadFailed(
                            "QNN backend initialization failed".into(),
                        ));
                    }
                    let path_bytes = path.as_bytes();
                    // SAFETY: `backend` is verified to be non-null. `path_bytes` is a valid
                    // byte slice representing the file path. The C API expects a byte array and length.
                    let ctx =
                        unsafe { qnn_load_context(backend, path_bytes.as_ptr(), path_bytes.len()) };
                    if ctx.is_null() {
                        // SAFETY: `backend` is a valid handle returned by `qnn_backend_init`.
                        unsafe {
                            qnn_backend_destroy(backend);
                        }
                        return Err(MiddlewareError::EngineLoadFailed(
                            "QNN context load failed".into(),
                        ));
                    }
                    // SAFETY: `ctx` is explicitly checked to be non-null and valid.
                    // The C library safely queries the context properties without mutating.
                    let input_elems = unsafe { qnn_get_input_elems(ctx) } as usize;
                    // SAFETY: `ctx` is valid and non-null.
                    let output_elems = unsafe { qnn_get_output_elems(ctx) } as usize;

                    state.backend_handle = Some(backend);
                    state.context_handle = Some(ctx);

                    let info = EngineInfo {
                        name: std::path::Path::new(path)
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "qnn_model".into()),
                        precision: Precision::FP32,
                        inputs: vec![TensorSpec {
                            name: "input".to_string(),
                            shape: if input_elems > 0 {
                                vec![1, input_elems]
                            } else {
                                vec![1, 3, 224, 224]
                            },
                        }],
                        outputs: vec![TensorSpec {
                            name: "output".to_string(),
                            shape: if output_elems > 0 {
                                vec![1, output_elems]
                            } else {
                                vec![1, 1000]
                            },
                        }],
                        memory_bytes: 0,
                    };
                    state.engine_info = Some(info.clone());
                    return Ok(info);
                }
                QualcommRuntime::Snpe => {
                    let path_bytes = path.as_bytes();
                    // SAFETY: `path_bytes` is a valid byte slice representing the file path.
                    // The SNPE C API safely handles reading the DLC model.
                    let handle = unsafe { snpe_load_dlc(path_bytes.as_ptr(), path_bytes.len()) };
                    if handle.is_null() {
                        return Err(MiddlewareError::EngineLoadFailed(
                            "SNPE DLC load failed".into(),
                        ));
                    }
                    state.context_handle = Some(handle);
                    let info = EngineInfo {
                        name: std::path::Path::new(path)
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "snpe_model".into()),
                        precision: Precision::FP32,
                        inputs: vec![TensorSpec {
                            name: "input".to_string(),
                            shape: vec![1, 3, 224, 224],
                        }],
                        outputs: vec![TensorSpec {
                            name: "output".to_string(),
                            shape: vec![1, 1000],
                        }],
                        memory_bytes: 0,
                    };
                    state.engine_info = Some(info.clone());
                    return Ok(info);
                }
                _ => {}
            }
        }

        // Simulation path
        let info = EngineInfo {
            name: std::path::Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "qualcomm_sim".into()),
            precision: Precision::FP32,
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 3, 224, 224],
            }],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![1, 1000],
            }],
            memory_bytes: 0,
        };

        state.runtime = QualcommRuntime::Simulated;
        state.engine_info = Some(info.clone());
        info!(backend = "qualcomm", mode = "simulation", "Model loaded");
        Ok(info)
    }

    fn allocate_buffers(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        // QNN/SNPE manage their own device buffers internally.
        state.buffers_allocated = true;
        debug!(backend = "qualcomm", "Buffers allocated");
        Ok(())
    }

    fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        let state = self.state.read();
        let info = state
            .engine_info
            .as_ref()
            .ok_or(MiddlewareError::EngineNotLoaded)?;

        if !state.buffers_allocated {
            return Err(MiddlewareError::InferenceFailed(
                "Buffers not allocated".into(),
            ));
        }

        if inputs.is_empty() {
            return Err(MiddlewareError::InferenceFailed(
                "No inputs provided".into(),
            ));
        }
        let input = &inputs[0];
        let output_spec = &info.outputs[0];

        let output_elems: usize = output_spec.shape.iter().product();
        let output_bytes = output_elems * Precision::FP32.element_size();

        #[cfg(feature = "qualcomm")]
        {
            match state.runtime {
                QualcommRuntime::Qnn => {
                    if let Some(ctx) = state.context_handle {
                        let mut output_data = vec![0u8; output_bytes];
                        // SAFETY: `ctx` is a valid context handle. The `input.data` and `output_data`
                        // are valid allocated memory regions of exactly the required sizes. The RwLock
                        // read guard on `state` ensures exclusive access to `ctx` during this operation.
                        let result = unsafe {
                            qnn_execute_graph(
                                ctx,
                                input.data.as_ptr(),
                                input.data.len(),
                                output_data.as_mut_ptr(),
                                output_bytes,
                            )
                        };
                        if result != 0 {
                            return Err(MiddlewareError::InferenceFailed(format!(
                                "QNN graph execution failed (error {})",
                                result
                            )));
                        }
                        return Ok(vec![TensorBuffer {
                            name: output_spec.name.clone(),
                            data: output_data,
                            shape: output_spec.shape.clone(),
                            precision: Precision::FP32,
                        }]);
                    }
                }
                QualcommRuntime::Snpe => {
                    if let Some(handle) = state.context_handle {
                        let mut output_data = vec![0u8; output_bytes];
                        // SAFETY: `handle` is a valid SNPE context. Memory slices are safely
                        // passed with correct exact lengths. RwLock ensures exclusive access.
                        let result = unsafe {
                            snpe_execute(
                                handle,
                                input.data.as_ptr(),
                                input.data.len(),
                                output_data.as_mut_ptr(),
                                output_bytes,
                            )
                        };
                        if result != 0 {
                            return Err(MiddlewareError::InferenceFailed(format!(
                                "SNPE execution failed (error {})",
                                result
                            )));
                        }
                        return Ok(vec![TensorBuffer {
                            name: output_spec.name.clone(),
                            data: output_data,
                            shape: output_spec.shape.clone(),
                            precision: Precision::FP32,
                        }]);
                    }
                }
                _ => {}
            }
        }

        // Simulation: return uniform scores
        let value = 1.0f32 / output_elems as f32;
        let mut output_data = vec![0u8; output_bytes];
        for chunk in output_data.chunks_exact_mut(4) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }

        Ok(vec![TensorBuffer {
            name: output_spec.name.clone(),
            data: output_data,
            shape: output_spec.shape.clone(),
            precision: Precision::FP32,
        }])
    }

    fn release(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        #[cfg(feature = "qualcomm")]
        {
            if let Some(ctx) = state.context_handle.take() {
                match state.runtime {
                    QualcommRuntime::Qnn => {
                        // SAFETY: `ctx` is a valid QNN context being exclusively destroyed.
                        unsafe { qnn_destroy(ctx) }
                    }
                    QualcommRuntime::Snpe => {
                        // SAFETY: `ctx` is a valid SNPE context being exclusively destroyed.
                        unsafe { snpe_destroy(ctx) }
                    }
                    _ => {}
                }
            }
            if let Some(backend) = state.backend_handle.take() {
                // SAFETY: `backend` is a valid QNN backend handle being exclusively destroyed.
                unsafe {
                    qnn_backend_destroy(backend);
                }
            }
        }
        state.engine_info = None;
        state.buffers_allocated = false;
        state.runtime = QualcommRuntime::None;
        info!(backend = "qualcomm", "Backend released");
        Ok(())
    }

    fn engine_info(&self) -> Option<EngineInfo> {
        self.state.read().engine_info.clone()
    }

    fn is_ready(&self) -> bool {
        let state = self.state.read();
        state.engine_info.is_some() && state.buffers_allocated
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualcomm_adapter_basics() {
        let a = QualcommAdapter::new();
        assert_eq!(a.backend_name(), "qualcomm");
        assert!(!a.is_ready());
    }

    #[test]
    fn qualcomm_adapter_default() {
        let a = QualcommAdapter::default();
        assert!(a.engine_info().is_none());
    }

    #[test]
    fn qualcomm_load_nonexistent_fails() {
        let a = QualcommAdapter::new();
        assert!(a.load_engine("/does/not/exist.bin").is_err());
    }

    #[test]
    fn qualcomm_infer_without_load_fails() {
        let a = QualcommAdapter::new();
        let input = TensorBuffer::from_f32("test", &[0.0; 6], vec![1, 2, 3]);
        assert!(a.infer(&[input]).is_err());
    }
}
