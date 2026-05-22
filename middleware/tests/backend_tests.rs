// =============================================================================
// Integration Test — Backend Adapters
// =============================================================================
//! Tests that each backend adapter correctly implements the InferenceBackend
//! trait lifecycle.

use std::io::Write;

use magna_middleware::backends::cpu::adapter::CpuAdapter;
#[cfg(feature = "nvidia")]
use magna_middleware::backends::nvidia::adapter::{NvidiaAdapter, NvidiaHardware};
#[cfg(feature = "qualcomm")]
use magna_middleware::backends::qualcomm::adapter::QualcommAdapter;
#[cfg(feature = "ti")]
use magna_middleware::backends::ti::adapter::TiAdapter;
use magna_middleware::inference::traits::{InferenceBackend, TensorBuffer};
use magna_middleware::utils::errors::Precision;

fn make_dummy_file(ext: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(&format!(".{}", ext))
        .tempfile()
        .unwrap();
    f.write_all(b"DUMMY_ENGINE_DATA").unwrap();
    f
}

fn dummy_input() -> TensorBuffer {
    TensorBuffer {
        name: "input".into(),
        data: vec![0u8; 3 * 224 * 224 * 4],
        shape: vec![1, 3, 224, 224],
        precision: Precision::FP32,
    }
}

/// Generic lifecycle test for any backend.
fn test_backend_lifecycle(adapter: &mut dyn InferenceBackend, engine_file: &str) {
    // Not ready initially.
    assert!(!adapter.is_ready());

    // Load engine.
    let info = adapter.load_engine(engine_file).unwrap();
    assert!(!info.name.is_empty());
    assert_eq!(info.inputs[0].shape, vec![1, 3, 224, 224]);
    assert_eq!(info.outputs[0].shape, vec![1, 1000]);

    // Allocate buffers.
    adapter.allocate_buffers().unwrap();
    assert!(adapter.is_ready());

    // Run inference.
    let output = adapter.infer(&[dummy_input()]).unwrap();
    assert_eq!(output[0].shape, vec![1, 1000]);
    assert_eq!(output[0].precision, Precision::FP32);
    assert!(!output[0].data.is_empty());

    // Verify the output is interpretable as f32.
    let scores = output[0].as_f32_slice();
    assert_eq!(scores.len(), 1000);

    // Check engine info.
    assert!(adapter.engine_info().is_some());

    // Release.
    adapter.release().unwrap();
    assert!(!adapter.is_ready());
    assert!(adapter.engine_info().is_none());
}

#[test]
#[cfg(feature = "nvidia")]
fn nvidia_adapter_lifecycle() {
    let file = make_dummy_file("engine");
    let mut adapter = NvidiaAdapter::new(NvidiaHardware::Orin);
    test_backend_lifecycle(&mut adapter, file.path().to_str().unwrap());
}

#[test]
#[cfg(feature = "nvidia")]
fn nvidia_adapter_with_thor() {
    let file = make_dummy_file("engine");
    let mut adapter = NvidiaAdapter::new(NvidiaHardware::Thor);
    assert_eq!(adapter.max_precision(), Precision::FP8);
    test_backend_lifecycle(&mut adapter, file.path().to_str().unwrap());
}

#[test]
#[cfg(feature = "qualcomm")]
fn qualcomm_adapter_lifecycle() {
    let file = make_dummy_file("bin");
    let mut adapter = QualcommAdapter::new();
    test_backend_lifecycle(&mut adapter, file.path().to_str().unwrap());
}

#[test]
#[cfg(feature = "ti")]
fn ti_adapter_lifecycle() {
    let file = make_dummy_file("tidl");
    let mut adapter = TiAdapter::new();
    test_backend_lifecycle(&mut adapter, file.path().to_str().unwrap());
}

#[test]
fn cpu_adapter_lifecycle() {
    let file = make_dummy_file("onnx");
    let mut adapter = CpuAdapter::new();
    test_backend_lifecycle(&mut adapter, file.path().to_str().unwrap());
}

#[test]
fn backend_names() {
    #[cfg(feature = "nvidia")]
    assert_eq!(
        NvidiaAdapter::new(NvidiaHardware::Orin).backend_name(),
        "nvidia"
    );
    #[cfg(feature = "qualcomm")]
    assert_eq!(QualcommAdapter::new().backend_name(), "qualcomm");
    #[cfg(feature = "ti")]
    assert_eq!(TiAdapter::new().backend_name(), "ti");
    assert_eq!(CpuAdapter::new().backend_name(), "cpu");
}

#[cfg(all(feature = "nvidia", feature = "qualcomm", feature = "ti"))]
#[test]
fn load_nonexistent_fails_all_adapters() {
    #[cfg(feature = "nvidia")]
    let nv = NvidiaAdapter::new(NvidiaHardware::Orin);
    #[cfg(feature = "qualcomm")]
    let qc = QualcommAdapter::new();
    #[cfg(feature = "ti")]
    let ti = TiAdapter::new();
    let cpu = CpuAdapter::new();

    #[cfg(feature = "nvidia")]
    assert!(nv.load_engine("/does/not/exist.engine").is_err());
    #[cfg(feature = "qualcomm")]
    assert!(qc.load_engine("/does/not/exist.bin").is_err());
    #[cfg(feature = "ti")]
    assert!(ti.load_engine("/does/not/exist.tidl").is_err());
    assert!(cpu.load_engine("/does/not/exist.onnx").is_err());
}

#[cfg(all(feature = "nvidia", feature = "qualcomm", feature = "ti"))]
#[test]
fn infer_without_load_fails() {
    #[cfg(feature = "nvidia")]
    let nv = NvidiaAdapter::new(NvidiaHardware::Orin);
    #[cfg(feature = "qualcomm")]
    let qc = QualcommAdapter::new();
    #[cfg(feature = "ti")]
    let ti = TiAdapter::new();
    let cpu = CpuAdapter::new();
    let input = dummy_input();

    #[cfg(feature = "nvidia")]
    assert!(nv.infer(std::slice::from_ref(&input)).is_err());
    #[cfg(feature = "qualcomm")]
    assert!(qc.infer(std::slice::from_ref(&input)).is_err());
    #[cfg(feature = "ti")]
    assert!(ti.infer(std::slice::from_ref(&input)).is_err());
    assert!(cpu.infer(std::slice::from_ref(&input)).is_err());
}

#[test]
fn default_trait_implementations() {
    #[cfg(feature = "nvidia")]
    let _nv = NvidiaAdapter::default();
    #[cfg(feature = "qualcomm")]
    let _qc = QualcommAdapter::default();
    #[cfg(feature = "ti")]
    let _ti = TiAdapter::default();
    let _cpu = CpuAdapter::default();
}
