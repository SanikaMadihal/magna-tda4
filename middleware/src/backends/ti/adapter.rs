// =============================================================================
// Magna Middleware — Texas Instruments TDA4 Backend Adapter
// =============================================================================
use std::ffi::c_void;
use tracing::{debug, info};

use crate::inference::traits::{EngineInfo, InferenceBackend, TensorBuffer, TensorSpec};
use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};

#[cfg(feature = "ti")]
extern "C" {
    fn tidl_rt_init() -> *mut c_void;
    fn tidl_rt_load_model(handle: *mut c_void, path: *const u8, path_len: usize) -> i32;
    fn tidl_rt_alloc_tensors(handle: *mut c_void) -> i32;
    fn tidl_rt_process(
        handle: *mut c_void,
        input: *const u8,
        input_bytes: usize,
        output: *mut u8,
        output_bytes: usize,
    ) -> i32;
    fn tidl_rt_get_input_elems(handle: *mut c_void) -> i32;
    fn tidl_rt_get_output_elems(handle: *mut c_void) -> i32;
    fn tidl_rt_destroy(handle: *mut c_void);
}

pub struct TiAdapter {
    state: parking_lot::RwLock<TiState>,
}

struct TiState {
    engine_info: Option<EngineInfo>,
    buffers_allocated: bool,
    is_simulated: bool,
    #[cfg(feature = "ti")]
    tidl_handle: Option<*mut c_void>,
}

unsafe impl Send for TiState {}
unsafe impl Sync for TiState {}

impl TiAdapter {
    pub fn new() -> Self {
        Self {
            state: parking_lot::RwLock::new(TiState {
                engine_info: None,
                buffers_allocated: false,
                is_simulated: true,
                #[cfg(feature = "ti")]
                tidl_handle: None,
            }),
        }
    }
}

impl Default for TiAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for TiAdapter {
    fn backend_name(&self) -> &str {
        "ti"
    }

    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        let mut state = self.state.write();
        info!(backend = "ti", engine_path = %path, "Loading model");

        #[cfg(feature = "ti")]
        {
            let handle = unsafe { tidl_rt_init() };
            if handle.is_null() {
                return Err(MiddlewareError::EngineLoadFailed(
                    "DLR runtime init failed".into(),
                ));
            }

            let path_bytes = path.as_bytes();
            let ret = unsafe {
                tidl_rt_load_model(handle, path_bytes.as_ptr(), path_bytes.len())
            };
            if ret != 0 {
                unsafe { tidl_rt_destroy(handle); }
                return Err(MiddlewareError::EngineLoadFailed(
                    format!("DLR model load failed (error {})", ret),
                ));
            }

            let input_elems  = unsafe { tidl_rt_get_input_elems(handle) } as usize;
            let output_elems = unsafe { tidl_rt_get_output_elems(handle) } as usize;

            state.tidl_handle  = Some(handle);
            state.is_simulated = false;

            let info = EngineInfo {
                name: std::path::Path::new(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "dlr_model".into()),
                precision: Precision::FP32,
                inputs: vec![TensorSpec {
                    name: "input.1Net_IN".into(),
                    shape: vec![1, 3, 224, 224],  // gRPC client sends NCHW float32
                }],
                outputs: vec![TensorSpec {
                    name: "output".into(),
                    shape: vec![1, 1000],
                }],
                memory_bytes: 0,
            };
            state.engine_info = Some(info.clone());
            info!(backend = "ti", engine = %info.name, "DLR model loaded");
            return Ok(info);
        }

        // Simulation fallback (non-ti builds)
        #[allow(unreachable_code)]
        {
            let info = EngineInfo {
                name: "sim".into(),
                precision: Precision::FP32,
                inputs: vec![TensorSpec { name: "input".into(), shape: vec![1, 3, 224, 224] }],
                outputs: vec![TensorSpec { name: "output".into(), shape: vec![1, 1000] }],
                memory_bytes: 0,
            };
            state.is_simulated = true;
            state.engine_info  = Some(info.clone());
            Ok(info)
        }
    }

    fn allocate_buffers(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        #[cfg(feature = "ti")]
        {
            if !state.is_simulated {
                if let Some(handle) = state.tidl_handle {
                    let ret = unsafe { tidl_rt_alloc_tensors(handle) };
                    if ret != 0 {
                        return Err(MiddlewareError::BufferAllocationFailed(
                            format!("CMEM alloc failed (error {})", ret),
                        ));
                    }
                }
            }
        }
        state.buffers_allocated = true;
        debug!(backend = "ti", "Buffers allocated");
        Ok(())
    }

    fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        let state = self.state.read();
        let info = state.engine_info.as_ref().ok_or(MiddlewareError::EngineNotLoaded)?;

        if !state.buffers_allocated {
            return Err(MiddlewareError::InferenceFailed("Buffers not allocated".into()));
        }
        if inputs.is_empty() {
            return Err(MiddlewareError::InferenceFailed("No inputs provided".into()));
        }

        let input        = &inputs[0];
        let output_spec  = &info.outputs[0];
        let output_elems: usize = output_spec.shape.iter().product();
        let output_bytes = output_elems * Precision::FP32.element_size();

        #[cfg(feature = "ti")]
        {
            if !state.is_simulated {
                if let Some(handle) = state.tidl_handle {
                    // Model expects uint8 NCHW [1,3,224,224]
                    // Pass input bytes directly — benchmark sends uint8 NCHW
                    let mut output_data = vec![0u8; output_bytes];
                    let ret = unsafe {
                        tidl_rt_process(
                            handle,
                            input.data.as_ptr(),
                            input.data.len(),
                            output_data.as_mut_ptr(),
                            output_bytes,
                        )
                    };
                    if ret != 0 {
                        return Err(MiddlewareError::InferenceFailed(
                            format!("DLR inference failed (error {})", ret),
                        ));
                    }
                    return Ok(vec![TensorBuffer {
                        name:      output_spec.name.clone(),
                        data:      output_data,
                        shape:     output_spec.shape.clone(),
                        precision: Precision::FP32,
                    }]);
                }
            }
        }

        // Simulation: uniform scores
        let value = 1.0f32 / output_elems as f32;
        let mut output_data = vec![0u8; output_bytes];
        for chunk in output_data.chunks_exact_mut(4) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        Ok(vec![TensorBuffer {
            name:      output_spec.name.clone(),
            data:      output_data,
            shape:     output_spec.shape.clone(),
            precision: Precision::FP32,
        }])
    }

    fn release(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        #[cfg(feature = "ti")]
        {
            if let Some(handle) = state.tidl_handle.take() {
                unsafe { tidl_rt_destroy(handle); }
            }
        }
        state.engine_info      = None;
        state.buffers_allocated = false;
        state.is_simulated     = true;
        info!(backend = "ti", "Backend released");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ti_adapter_basics() {
        let a = TiAdapter::new();
        assert_eq!(a.backend_name(), "ti");
        assert!(!a.is_ready());
    }

    #[test]
    fn ti_adapter_default() {
        let a = TiAdapter::default();
        assert!(a.engine_info().is_none());
    }

    #[test]
    fn ti_infer_without_load_fails() {
        let a = TiAdapter::new();
        let input = TensorBuffer::from_f32("test", &[0.0; 6], vec![1, 2, 3]);
        assert!(a.infer(&[input]).is_err());
    }
}
