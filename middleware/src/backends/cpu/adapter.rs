// =============================================================================
// Magna Middleware — CPU Fallback Backend
// =============================================================================
//! A simple CPU-based inference backend that serves as the universal fallback.
//!
//! In production, this would use ONNX Runtime CPU EP.  For now, the simulation
//! path returns uniform scores so the pipeline can be tested end-to-end.

use tracing::{debug, info};

use crate::inference::traits::{EngineInfo, InferenceBackend, TensorBuffer, TensorSpec};
use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};

pub struct CpuAdapter {
    state: parking_lot::RwLock<CpuState>,
}

struct CpuState {
    engine_info: Option<EngineInfo>,
    buffers_allocated: bool,
}

impl CpuAdapter {
    pub fn new() -> Self {
        Self {
            state: parking_lot::RwLock::new(CpuState {
                engine_info: None,
                buffers_allocated: false,
            }),
        }
    }
}

impl Default for CpuAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for CpuAdapter {
    fn backend_name(&self) -> &str {
        "cpu"
    }

    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        info!(backend = "cpu", engine_path = %path, "Loading model");

        if !std::path::Path::new(path).exists() {
            return Err(MiddlewareError::EngineLoadFailed(format!(
                "Model file not found: {}",
                path
            )));
        }

        // TODO: Use ONNX Runtime to load the model and extract actual shapes.
        // For now, we create a generic info based on the file.
        let info = EngineInfo {
            name: std::path::Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "cpu_model".into()),
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

        let mut state = self.state.write();
        state.engine_info = Some(info.clone());
        info!(backend = "cpu", engine = %info.name, "Model loaded");
        Ok(info)
    }

    fn allocate_buffers(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        state.buffers_allocated = true;
        debug!(backend = "cpu", "Buffers allocated (no-op for CPU)");
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

        // Validate input
        input
            .validate()
            .map_err(|e| MiddlewareError::InferenceFailed(format!("Input validation: {}", e)))?;

        let output_spec = &info.outputs[0];
        let output_elems: usize = output_spec.shape.iter().product();
        let output_bytes = output_elems * Precision::FP32.element_size();

        // TODO: Replace with actual ONNX Runtime inference.
        // For now, return uniform distribution.
        let value = 1.0f32 / output_elems as f32;
        let mut output_data = vec![0u8; output_bytes];
        for chunk in output_data.chunks_exact_mut(4) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }

        debug!(
            backend = "cpu",
            output_elements = output_elems,
            "Inference complete"
        );
        Ok(vec![TensorBuffer {
            name: output_spec.name.clone(),
            data: output_data,
            shape: output_spec.shape.clone(),
            precision: Precision::FP32,
        }])
    }

    fn release(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        state.engine_info = None;
        state.buffers_allocated = false;
        info!(backend = "cpu", "Backend released");
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
    fn cpu_adapter_basics() {
        let a = CpuAdapter::new();
        assert_eq!(a.backend_name(), "cpu");
        assert!(!a.is_ready());
    }

    #[test]
    fn cpu_adapter_default() {
        let a = CpuAdapter::default();
        assert!(a.engine_info().is_none());
    }

    #[test]
    fn cpu_load_nonexistent_fails() {
        let a = CpuAdapter::new();
        assert!(a.load_engine("/nonexistent/model.onnx").is_err());
    }
}
