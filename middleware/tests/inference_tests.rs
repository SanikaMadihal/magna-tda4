// =============================================================================
// Integration Test — Inference Pipeline
// =============================================================================
//! End-to-end test covering:
//! 1. Initialize middleware
//! 2. Load engine
//! 3. Run inference
//! 4. Classify output
//! 5. Check metrics
//! 6. Shutdown

use std::io::Write;

use magna_middleware::api::public_api::Middleware;
use magna_middleware::inference::traits::TensorBuffer;
use magna_middleware::lifecycle::state_manager::State;
use magna_middleware::utils::errors::{MiddlewareConfig, Precision};
use magna_middleware::utils::logging::init_test_logger;

fn create_dummy_engine() -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(".engine")
        .tempfile()
        .unwrap();
    f.write_all(b"FAKE_TRT_ENGINE_DATA").unwrap();
    f
}

#[test]
fn full_inference_pipeline() {
    init_test_logger();

    let mw = Middleware::new();
    assert_eq!(mw.state(), State::Uninitialized);

    // Initialize.
    let config = MiddlewareConfig {
        backend: "simulated".into(),
        precision: Precision::FP32,
        warmup_runs: 0,
        debug: true,
        labels_path: None,
        model_path: None,
        grpc_address: String::new(),
    };
    mw.initialize(config).unwrap();
    assert_eq!(mw.state(), State::Initialized);

    // Load engine.
    let engine_file = create_dummy_engine();
    let path = engine_file.path().to_str().unwrap();
    let info = mw.load_engine(path).unwrap();
    assert_eq!(mw.state(), State::Ready);
    assert_eq!(info.inputs[0].shape, vec![1, 3, 224, 224]);
    assert_eq!(info.outputs[0].shape, vec![1, 1000]);

    // Run inference with a raw tensor.
    let input = TensorBuffer {
        name: "input".into(),
        data: vec![0u8; 3 * 224 * 224 * 4],
        shape: vec![1, 3, 224, 224],
        precision: Precision::FP32,
    };
    let output = mw.infer(&[input]).unwrap();
    assert_eq!(output[0].shape, vec![1, 1000]);
    assert_eq!(output[0].precision, Precision::FP32);

    // Check that metrics recorded the inference (requires --features metrics).
    #[cfg(feature = "metrics")]
    {
        let metrics = mw.get_metrics();
        assert_eq!(metrics.inference_count, 1);
        assert!(metrics.last_latency_ms >= 0.0); // may be very fast
    }

    // Shutdown.
    mw.shutdown().unwrap();
    assert_eq!(mw.state(), State::Uninitialized);
}

#[test]
fn multiple_inferences() {
    init_test_logger();

    let mw = Middleware::new();
    mw.initialize(MiddlewareConfig {
        backend: "simulated".into(),
        precision: Precision::FP32,
        ..Default::default()
    })
    .unwrap();

    let engine_file = create_dummy_engine();
    mw.load_engine(engine_file.path().to_str().unwrap())
        .unwrap();

    let input = TensorBuffer {
        name: "input".to_string(),
        data: vec![0u8; 3 * 224 * 224 * 4],
        shape: vec![1, 3, 224, 224],
        precision: Precision::FP32,
    };

    // Run 10 inferences.
    for _ in 0..10 {
        mw.infer(std::slice::from_ref(&input)).unwrap();
    }

    #[cfg(feature = "metrics")]
    {
        let metrics = mw.get_metrics();
        assert_eq!(metrics.inference_count, 10);
        assert!(metrics.avg_latency_ms >= 0.0);
    }

    mw.shutdown().unwrap();
}

#[test]
fn reload_engine_while_ready() {
    init_test_logger();

    let mw = Middleware::new();
    mw.initialize(MiddlewareConfig {
        backend: "simulated".into(),
        ..Default::default()
    })
    .unwrap();

    let engine1 = create_dummy_engine();
    let engine2 = create_dummy_engine();

    mw.load_engine(engine1.path().to_str().unwrap()).unwrap();
    assert_eq!(mw.state(), State::Ready);

    // Reload with a different engine.
    mw.load_engine(engine2.path().to_str().unwrap()).unwrap();
    assert_eq!(mw.state(), State::Ready);

    mw.shutdown().unwrap();
}

#[test]
fn thread_safety() {
    init_test_logger();

    let mw = Middleware::new();
    mw.initialize(MiddlewareConfig {
        backend: "simulated".into(),
        ..Default::default()
    })
    .unwrap();

    let engine_file = create_dummy_engine();
    mw.load_engine(engine_file.path().to_str().unwrap())
        .unwrap();

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let mw_clone = mw.clone();
            std::thread::spawn(move || {
                let input = TensorBuffer {
                    name: "input".to_string(),
                    data: vec![0u8; 3 * 224 * 224 * 4],
                    shape: vec![1, 3, 224, 224],
                    precision: Precision::FP32,
                };
                for _ in 0..5 {
                    mw_clone.infer(std::slice::from_ref(&input)).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    #[cfg(feature = "metrics")]
    {
        let metrics = mw.get_metrics();
        assert_eq!(metrics.inference_count, 20); // 4 threads × 5 inferences
    }

    mw.shutdown().unwrap();
}
