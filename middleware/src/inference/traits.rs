// =============================================================================
// Magna Middleware — Inference Backend Trait
// =============================================================================
//! Defines the core abstraction that every backend adapter must implement.
//!
//! The middleware never calls vendor-specific APIs directly.  All interaction
//! goes through [`InferenceBackend`].  This trait is object-safe so that the
//! engine manager can hold `Box<dyn InferenceBackend>`.

use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};

// ---------------------------------------------------------------------------
// Tensor descriptor
// ---------------------------------------------------------------------------

/// Lightweight descriptor for a contiguous, row-major tensor buffer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

/// Lightweight descriptor for a contiguous, row-major tensor buffer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorBuffer {
    pub name: String,
    /// Raw bytes of the tensor (owned).
    pub data: Vec<u8>,
    /// Shape — e.g. `[1, 3, 224, 224]` for a single RGB image.
    pub shape: Vec<usize>,
    /// Element precision.
    pub precision: Precision,
}

impl TensorBuffer {
    /// Number of elements implied by shape.
    pub fn num_elements(&self) -> usize {
        if self.shape.is_empty() {
            return 0;
        }
        self.shape.iter().product()
    }

    /// Size of one element in bytes for the current precision.
    pub fn element_size(&self) -> usize {
        self.precision.element_size()
    }

    /// Expected total byte count.
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.element_size()
    }

    /// Create a new tensor buffer from f32 data.
    pub fn from_f32(name: &str, data: &[f32], shape: Vec<usize>) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            name: name.to_string(),
            data: bytes,
            shape,
            precision: Precision::FP32,
        }
    }

    /// Interpret the raw bytes as `&[f32]`.
    ///
    /// Returns an error if precision is not FP32 or byte alignment is wrong.
    /// Uses `bytemuck` for a safe, zero-cost cast — no undefined behavior.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.precision, Precision::FP32, "Tensor is not FP32");
        assert_eq!(
            self.data.len() % 4,
            0,
            "Tensor byte length {} is not a multiple of 4",
            self.data.len()
        );
        // Safe cast using bytemuck — handles alignment correctly.
        // If the Vec<u8> is not aligned to 4 bytes (unlikely but possible),
        // bytemuck will panic with a clear message rather than causing UB.
        bytemuck::cast_slice(&self.data)
    }

    /// Try to interpret as f32 slice, returning Result instead of panicking.
    pub fn try_as_f32_slice(&self) -> MiddlewareResult<&[f32]> {
        if self.precision != Precision::FP32 {
            return Err(MiddlewareError::PostprocessingFailed(format!(
                "Expected FP32 tensor, got {:?}",
                self.precision
            )));
        }
        if !self.data.len().is_multiple_of(4) {
            return Err(MiddlewareError::PostprocessingFailed(format!(
                "Byte length {} is not a multiple of 4",
                self.data.len()
            )));
        }
        Ok(bytemuck::cast_slice(&self.data))
    }

    /// Validate that this tensor's byte count matches shape × element_size.
    pub fn validate(&self) -> MiddlewareResult<()> {
        let expected = self.byte_size();
        if self.data.len() != expected {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Tensor size mismatch: data has {} bytes but shape {:?} × {:?} expects {} bytes",
                self.data.len(),
                self.shape,
                self.precision,
                expected,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Engine metadata
// ---------------------------------------------------------------------------

/// Metadata describing a loaded engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineInfo {
    /// Human-readable engine name or file stem.
    pub name: String,
    /// Precision the engine was compiled for.
    pub precision: Precision,
    /// Input descriptors.
    pub inputs: Vec<TensorSpec>,
    /// Output descriptors.
    pub outputs: Vec<TensorSpec>,
    /// Estimated GPU/accelerator memory usage in bytes.
    pub memory_bytes: u64,
}

// ---------------------------------------------------------------------------
// InferenceBackend trait
// ---------------------------------------------------------------------------

/// Core trait that every backend adapter implements.
///
/// # Lifecycle
/// 1. `load_engine` — load the precompiled artifact from disk.
/// 2. `allocate_buffers` — pre-allocate device-side I/O buffers.
/// 3. `infer` — run a single forward pass (may be called many times).
/// 4. `release` — tear down all resources.
pub trait InferenceBackend: Send + Sync {
    /// Human-readable backend identifier (e.g. `"nvidia"`, `"qualcomm"`).
    fn backend_name(&self) -> &str;

    /// Load a precompiled engine file from the given path.
    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo>;

    /// Pre-allocate input/output buffers on the device.
    fn allocate_buffers(&self) -> MiddlewareResult<()>;

    /// Execute a single inference pass.
    ///
    /// The implementation copies `inputs` to device memory, runs the engine,
    /// and copies the `outputs` back.
    fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>>;

    /// Release all resources (engine handle, device memory, etc.).
    fn release(&self) -> MiddlewareResult<()>;

    /// Return metadata about the currently loaded engine, or `None`.
    fn engine_info(&self) -> Option<EngineInfo>;

    /// Whether an engine is currently loaded and buffers are allocated.
    fn is_ready(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_buffer_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buf = TensorBuffer::from_f32("input", &data, vec![1, 2, 3]);
        assert_eq!(buf.num_elements(), 6);
        assert_eq!(buf.element_size(), 4);
        assert_eq!(buf.byte_size(), 24);

        let slice = buf.as_f32_slice();
        assert_eq!(slice.len(), 6);
        assert!((slice[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn tensor_shape_calculation() {
        let buf = TensorBuffer {
            name: "input".to_string(),
            data: vec![0u8; 3 * 224 * 224 * 4],
            shape: vec![1, 3, 224, 224],
            precision: Precision::FP32,
        };
        assert_eq!(buf.num_elements(), 150_528);
        assert_eq!(buf.byte_size(), 150_528 * 4);
    }

    #[test]
    fn tensor_validation() {
        let valid = TensorBuffer {
            name: "test".to_string(),
            data: vec![0u8; 24],
            shape: vec![1, 2, 3],
            precision: Precision::FP32,
        };
        assert!(valid.validate().is_ok());

        let invalid = TensorBuffer {
            name: "test".to_string(),
            data: vec![0u8; 10], // wrong size
            shape: vec![1, 2, 3],
            precision: Precision::FP32,
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn try_as_f32_non_fp32_fails() {
        let buf = TensorBuffer {
            name: "test".to_string(),
            data: vec![0u8; 4],
            shape: vec![1, 4],
            precision: Precision::INT8,
        };
        assert!(buf.try_as_f32_slice().is_err());
    }

    #[test]
    fn empty_shape_is_zero_elements() {
        let buf = TensorBuffer {
            name: "test".to_string(),
            data: vec![],
            shape: vec![],
            precision: Precision::FP32,
        };
        assert_eq!(buf.num_elements(), 0);
        assert_eq!(buf.byte_size(), 0);
    }
}
