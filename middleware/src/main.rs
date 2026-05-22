// =============================================================================
// Magna Middleware ΓÇö CLI Entry Point
// =============================================================================
//! Binary entry point for direct model inference from the command line.
//!
//! Usage:
//! ```bash
//! magna --model model.engine --image cat.jpg
//! magna --model model.engine --image cat.jpg --labels labels.txt
//! magna --model model.engine --image cat.jpg --backend simulated
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

use clap::Parser;
use std::time::Instant;
use tracing::{error, info};

use magna_middleware::api::public_api::Middleware;
use magna_middleware::utils::errors::{MiddlewareConfig, Precision};
use magna_middleware::utils::logging::{init_logger, LogConfig, LogFormat};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "magna", about = "Magna Edge AI Middleware ΓÇö Inference CLI")]
struct Cli {
    /// Path to the compiled engine / model file.
    #[arg(short, long)]
    model: String,

    /// Path to an image file for inference.
    #[arg(short, long)]
    image: String,

    /// Backend selection: auto, nvidia, qualcomm, ti, cpu, simulated.
    #[arg(short, long, default_value = "auto")]
    backend: String,

    /// Precision: fp32, fp16, int8, fp8.
    #[arg(short, long, default_value = "fp32")]
    precision: String,

    /// Path to labels file (one class per line).
    #[arg(short, long)]
    labels: Option<String>,

    /// Number of warmup inference runs.
    #[arg(short, long, default_value = "3")]
    warmup: usize,

    /// Number of benchmark iterations (0 = single inference, no benchmark).
    #[arg(long, default_value = "0")]
    benchmark_iters: usize,

    /// Enable debug logging.
    #[arg(short, long)]
    debug: bool,

    /// Log output format: pretty, json, compact.
    #[arg(long, default_value = "pretty")]
    log_format: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let format = match cli.log_format.as_str() {
        "json" => LogFormat::Json,
        "compact" => LogFormat::Compact,
        _ => LogFormat::Pretty,
    };
    let default_level = if cli.debug { "debug" } else { "info" };
    init_logger(LogConfig {
        format,
        default_level: default_level.into(),
    });

    let precision: Precision = match cli.precision.parse() {
        Ok(p) => p,
        Err(e) => {
            error!(precision = %cli.precision, error = %e, "Invalid precision");
            std::process::exit(1);
        }
    };

    let mw = Middleware::new();

    // Initialize
    let config = MiddlewareConfig {
        backend: cli.backend.clone(),
        precision,
        warmup_runs: cli.warmup,
        debug: cli.debug,
        labels_path: cli.labels.clone(),
        model_path: Some(cli.model.clone()),
        grpc_address: String::new(),
    };

    if let Err(e) = mw.initialize(config) {
        error!(error = %e, "Failed to initialize middleware");
        std::process::exit(1);
    }

    // Load engine
    match mw.load_engine(&cli.model) {
        Ok(info) => {
            info!(
                engine       = %info.name,
                inputs       = info.inputs.len(),
                outputs      = info.outputs.len(),
                memory_bytes = info.memory_bytes,
                "Engine loaded"
            );
        }
        Err(e) => {
            error!(engine_path = %cli.model, error = %e, "Failed to load engine");
            std::process::exit(1);
        }
    }

    // Warmup
    if cli.warmup > 0 {
        info!(warmup_iters = cli.warmup, "Running warmup");
        for _ in 0..cli.warmup {
            let _ = mw.infer_from_image(&cli.image);
        }
    }

    // Inference
    let start = Instant::now();
    let iters = if cli.benchmark_iters > 0 {
        cli.benchmark_iters
    } else {
        1
    };

    for i in 0..iters {
        match mw.infer_from_image(&cli.image) {
            Ok(result) => {
                if iters == 1 || i == iters - 1 {
                    println!("\n=== Inference Result ===");
                    println!("{}", result);

                    if let Some(cls) = &result.classification {
                        println!("Top-1: {} ({:.4})", cls.top1_label, cls.top1_score);
                        for (label, score, idx) in &cls.top5 {
                            println!("  {} ({:.4}) [{}]", label, score, idx);
                        }
                    } else {
                        println!("Output tensors: {}", result.outputs.len());
                        if let Some(first) = result.outputs.first() {
                            println!("Output shape: {:?}", first.shape);
                            println!("Output elements: {}", first.num_elements());
                        }
                    }
                }
            }
            Err(e) => {
                error!(error = %e, "Inference failed");
                std::process::exit(1);
            }
        }
    }

    let elapsed = start.elapsed();
    if cli.benchmark_iters > 0 {
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        let fps = iters as f64 / elapsed.as_secs_f64();
        println!("\n=== Benchmark ===");
        println!("  Iterations: {}", iters);
        println!("  Total time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Avg latency: {:.3}ms", avg_ms);
        println!("  Throughput:  {:.1} FPS", fps);
    } else {
        println!("\nInference time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    }

    // Metrics (only available when compiled with --features metrics)
    #[cfg(feature = "metrics")]
    {
        let metrics = mw.get_metrics();
        println!("\n=== Metrics ===");
        println!("  Total inferences: {}", metrics.inference_count);
        println!("  Avg latency:      {:.3}ms", metrics.avg_latency_ms);
        println!("  Min latency:      {:.3}ms", metrics.min_latency_ms);
        println!("  Max latency:      {:.3}ms", metrics.max_latency_ms);
    }

    mw.shutdown().ok();
}
