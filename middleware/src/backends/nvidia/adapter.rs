// =============================================================================
// Magna Middleware — NVIDIA TensorRT Backend Adapter
// =============================================================================
//! Implements [`InferenceBackend`] for NVIDIA Orin / Thor hardware using
//! TensorRT.
//!
//! On non-NVIDIA platforms (no `nvidia` feature), this module still compiles
//! but uses a simulation path.  This allows tests and development to proceed
//! without actual CUDA hardware.

use std::ffi::c_void;
use tracing::{debug, info};

use crate::inference::traits::{EngineInfo, InferenceBackend, TensorBuffer, TensorSpec};
use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};

// ---------------------------------------------------------------------------
// FFI declarations (only linked when `nvidia` feature is enabled)
// ---------------------------------------------------------------------------

#[cfg(feature = "nvidia")]
extern "C" {
    fn trt_load_engine(path: *const u8, path_len: usize) -> *mut c_void;
    fn trt_destroy_engine(ctx: *mut c_void);
    fn trt_allocate_buffers(ctx: *mut c_void) -> i32;
    fn trt_infer(
        ctx: *mut c_void,
        input: *const u8,
        input_bytes: usize,
        output: *mut u8,
        output_bytes: usize,
    ) -> i32;
    fn trt_get_device_memory(ctx: *mut c_void) -> i64;
    fn trt_get_input_elems(ctx: *mut c_void) -> i32;
    fn trt_get_output_elems(ctx: *mut c_void) -> i32;
}

// ---------------------------------------------------------------------------
// TrtContext wrapper (Send but NOT Sync — TRT is not thread-safe)
// ---------------------------------------------------------------------------

struct TrtContext {
    ptr: *mut c_void,
}

// SAFETY: TensorRT contexts can be moved between threads.
unsafe impl Send for TrtContext {}
// SAFETY: The raw pointer is private and access is mediated through RwLock ensuring exclusive access.
unsafe impl Sync for TrtContext {}

impl Drop for TrtContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            #[cfg(feature = "nvidia")]
            // SAFETY: `self.ptr` is checked for null before being passed to `trt_destroy_engine`.
            // The pointer is exclusively owned by `TrtContext` and was validly initialized by `trt_load_engine`.
            unsafe {
                trt_destroy_engine(self.ptr);
            }
            info!(backend = "nvidia", "Engine resources released");
        }
    }
}

// ---------------------------------------------------------------------------
// NvidiaAdapter
// ---------------------------------------------------------------------------

/// NVIDIA hardware generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvidiaHardware {
    /// Orin (Ampere) — supports up to FP16 precision
    Orin,
    /// Thor (next-gen) — supports up to FP8 precision
    Thor,
}

/// NVIDIA TensorRT backend adapter.
pub struct NvidiaAdapter {
    hardware: NvidiaHardware,
    state: parking_lot::RwLock<NvidiaState>,
}

struct NvidiaState {
    context: Option<TrtContext>,
    engine_info: Option<EngineInfo>,
    buffers_allocated: bool,
}

impl NvidiaAdapter {
    pub fn new(hardware: NvidiaHardware) -> Self {
        Self {
            hardware,
            state: parking_lot::RwLock::new(NvidiaState {
                context: None,
                engine_info: None,
                buffers_allocated: false,
            }),
        }
    }

    /// Maximum precision supported by this adapter.
    pub fn max_precision(&self) -> Precision {
        match self.hardware {
            NvidiaHardware::Thor => Precision::FP8,
            NvidiaHardware::Orin => Precision::FP16,
        }
    }
}

impl InferenceBackend for NvidiaAdapter {
    fn backend_name(&self) -> &str {
        "nvidia"
    }

    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        info!(backend = "nvidia", engine_path = %path, "Loading engine");

        // Validate file exists
        if !std::path::Path::new(path).exists() {
            return Err(MiddlewareError::EngineLoadFailed(format!(
                "Engine file not found: {}",
                path
            )));
        }

        #[cfg(feature = "nvidia")]
        {
            let path_bytes = path.as_bytes();
            // SAFETY: `path_bytes` points to a valid byte slice representing the file path.
            // `trt_load_engine` safely reads the byte array up to the provided length and allocates
            // an opaque context pointer, returning null on failure.
            let ctx_ptr = unsafe { trt_load_engine(path_bytes.as_ptr(), path_bytes.len()) };

            if ctx_ptr.is_null() {
                return Err(MiddlewareError::EngineLoadFailed(
                    "TensorRT returned null context — engine deserialization failed".into(),
                ));
            }

            // SAFETY: `ctx_ptr` was successfully allocated by `trt_load_engine` and explicitly checked
            // to be non-null. The C library guarantees these accessors safely read engine properties
            // without mutating the context.
            let input_elems = unsafe { trt_get_input_elems(ctx_ptr) } as usize;
            // SAFETY: `ctx_ptr` is valid and non-null as verified above.
            let output_elems = unsafe { trt_get_output_elems(ctx_ptr) } as usize;
            // SAFETY: `ctx_ptr` is valid and non-null as verified above.
            let device_mem = unsafe { trt_get_device_memory(ctx_ptr) } as u64;

            let mut state = self.state.write();
            state.context = Some(TrtContext { ptr: ctx_ptr });

            // Dynamically determine shape from the engine
            // (the C++ side should report actual shapes; these defaults are for
            // engines that don't report them)
            let info = EngineInfo {
                name: std::path::Path::new(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "engine".into()),
                precision: match self.hardware {
                    NvidiaHardware::Thor => Precision::FP8,
                    NvidiaHardware::Orin => Precision::FP16,
                },
                inputs: vec![TensorSpec {
                    name: "input".to_string(),
                    shape: if input_elems > 0 {
                        // Try to infer shape from element count
                        infer_shape_from_elements(input_elems)
                    } else {
                        vec![1, 3, 224, 224] // fallback
                    },
                }],
                outputs: vec![TensorSpec {
                    name: "output".to_string(),
                    shape: if output_elems > 0 {
                        vec![1, output_elems]
                    } else {
                        vec![1, 1000] // fallback
                    },
                }],
                memory_bytes: device_mem,
            };

            info!(
                backend      = "nvidia",
                engine       = %info.name,
                memory_bytes = device_mem,
                "Engine loaded"
            );
            state.engine_info = Some(info.clone());
            Ok(info)
        }

        #[cfg(not(feature = "nvidia"))]
        {
            // Simulation path for development / CI
            debug!(
                backend = "nvidia",
                mode = "simulation",
                "No real TensorRT available"
            );
            let info = EngineInfo {
                name: std::path::Path::new(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "simulated_engine".into()),
                precision: Precision::FP32,
                inputs: vec![TensorSpec {
                    name: "input".to_string(),
                    shape: vec![1, 3, 224, 224],
                }],
                outputs: vec![TensorSpec {
                    name: "output".to_string(),
                    shape: vec![1, 1000],
                }],
                memory_bytes: 50 * 1024 * 1024,
            };
            let mut state = self.state.write();
            state.engine_info = Some(info.clone());
            Ok(info)
        }
    }

    fn allocate_buffers(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        #[cfg(feature = "nvidia")]
        {
            if let Some(ref ctx) = state.context {
                // SAFETY: `ctx.ptr` is a valid initialized TensorRT context managed by `TrtContext`.
                // `trt_allocate_buffers` handles CUDA memory allocation internally and safely binds
                // it to the context.
                let result = unsafe { trt_allocate_buffers(ctx.ptr) };
                if result != 0 {
                    return Err(MiddlewareError::BufferAllocationFailed(format!(
                        "CUDA buffer allocation failed (error code {})",
                        result
                    )));
                }
            } else {
                return Err(MiddlewareError::EngineNotLoaded);
            }
        }

        state.buffers_allocated = true;
        debug!(backend = "nvidia", "Buffers allocated");
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
                "Buffers not allocated — call allocate_buffers() first".into(),
            ));
        }

        if inputs.is_empty() {
            return Err(MiddlewareError::InferenceFailed(
                "No inputs provided".into(),
            ));
        }
        let input = &inputs[0];
        let input_spec = &info.inputs[0];
        let output_spec = &info.outputs[0];

        // Validate input tensor size
        let expected_input_bytes =
            input_spec.shape.iter().product::<usize>() * info.precision.element_size();
        if input.data.len() != expected_input_bytes {
            // Allow FP32 input even if engine is FP16 (auto-conversion)
            let fp32_expected = input_spec.shape.iter().product::<usize>() * 4;
            if input.data.len() != fp32_expected {
                return Err(MiddlewareError::InferenceFailed(format!(
                    "Input tensor size mismatch: got {} bytes, expected {} or {} bytes",
                    input.data.len(),
                    expected_input_bytes,
                    fp32_expected
                )));
            }
        }

        let output_elems: usize = output_spec.shape.iter().product();
        let output_bytes = output_elems * Precision::FP32.element_size();

        #[cfg(feature = "nvidia")]
        {
            let ctx = state
                .context
                .as_ref()
                .ok_or(MiddlewareError::EngineNotLoaded)?;
            let mut output_data = vec![0u8; output_bytes];
            // SAFETY: `ctx.ptr` is a valid initialized TensorRT engine context.
            // The `input.data` and `output_data` slices are valid memory allocations, passed with their
            // exact lengths. The `RwLock` read guard on `state` ensures that `ctx` is not modified or
            // dropped during this inference call.
            let result = unsafe {
                trt_infer(
                    ctx.ptr,
                    input.data.as_ptr(),
                    input.data.len(),
                    output_data.as_mut_ptr(),
                    output_bytes,
                )
            };
            if result != 0 {
                return Err(MiddlewareError::InferenceFailed(format!(
                    "TensorRT inference returned error code {}",
                    result
                )));
            }
            Ok(vec![TensorBuffer {
                name: output_spec.name.clone(),
                data: output_data,
                shape: output_spec.shape.clone(),
                precision: Precision::FP32,
            }])
        }

        #[cfg(not(feature = "nvidia"))]
        {
            // Simulation: return uniform scores
            let mut output_data = vec![0u8; output_bytes];
            let value = 1.0f32 / output_elems as f32;
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
    }

    fn release(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        state.context = None; // TrtContext::drop will call trt_destroy_engine
        state.engine_info = None;
        state.buffers_allocated = false;
        info!(backend = "nvidia", "Backend released");
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

impl Default for NvidiaAdapter {
    fn default() -> Self {
        Self::new(NvidiaHardware::Orin)
    }
}

// ---------------------------------------------------------------------------
// Helper: try to guess a shape from element count
// ---------------------------------------------------------------------------

fn infer_shape_from_elements(elems: usize) -> Vec<usize> {
    // Common ImageNet-like shapes
    if elems == 3 * 224 * 224 {
        return vec![1, 3, 224, 224];
    }
    if elems == 3 * 256 * 256 {
        return vec![1, 3, 256, 256];
    }
    if elems == 3 * 299 * 299 {
        return vec![1, 3, 299, 299];
    }
    if elems == 3 * 384 * 384 {
        return vec![1, 3, 384, 384];
    }
    if elems == 3 * 416 * 416 {
        return vec![1, 3, 416, 416];
    }
    if elems == 3 * 480 * 480 {
        return vec![1, 3, 480, 480];
    }
    if elems == 3 * 640 * 640 {
        return vec![1, 3, 640, 640];
    }
    // Generic: assume batch=1 and flatten
    vec![1, elems]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_adapter_create() {
        let a = NvidiaAdapter::new(NvidiaHardware::Orin);
        assert_eq!(a.backend_name(), "nvidia");
        assert!(!a.is_ready());
        assert_eq!(a.max_precision(), Precision::FP16);
    }

    #[test]
    fn nvidia_adapter_thor() {
        let a = NvidiaAdapter::new(NvidiaHardware::Thor);
        assert_eq!(a.max_precision(), Precision::FP8);
    }

    #[test]
    fn nvidia_adapter_default() {
        let a = NvidiaAdapter::default();
        assert_eq!(a.hardware, NvidiaHardware::Orin);
    }

    #[test]
    fn load_nonexistent_engine_fails() {
        let a = NvidiaAdapter::new(NvidiaHardware::Orin);
        assert!(a.load_engine("/does/not/exist.engine").is_err());
    }

    #[test]
    fn infer_without_load_fails() {
        let a = NvidiaAdapter::new(NvidiaHardware::Orin);
        let input = TensorBuffer::from_f32("test", &[0.0; 6], vec![1, 2, 3]);
        assert!(a.infer(&[input]).is_err());
    }
}
