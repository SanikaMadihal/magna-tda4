// =============================================================================
// Magna Middleware — gRPC Client (Camera Edge Process)
// =============================================================================
use clap::Parser;
use std::time::Duration;
#[cfg(feature = "camera")]
use std::time::Instant;
use tonic::Request;
use tracing::{error, info, warn};

use magna_middleware::api::grpc::magna_grpc::inference_service_client::InferenceServiceClient;
use magna_middleware::api::grpc::magna_grpc::{InferInputTensor, InferenceRequest};
use magna_middleware::preprocess::imagenet;
use magna_middleware::utils::errors::Precision;

pub mod input;

#[derive(Parser, Debug)]
#[command(name = "magna_client", about = "Edge Camera gRPC Client for Magna Inference")]
struct Cli {
    #[arg(short = 'c', long, default_value = "/dev/video0")]
    camera: String,

    #[arg(short, long, default_value = "http://127.0.0.1:50051")]
    server: String,

    /// Path to labels file (one class per line)
    #[arg(short, long)]
    labels: Option<String>,

    #[arg(long, default_value = "0")]
    max_reconnects: usize,

    #[arg(long, default_value = "3")]
    reconnect_delay: u64,

    #[arg(short, long)]
    debug: bool,
}

// ── Top-5 decoder ─────────────────────────────────────────────────────────────
fn top5_from_f32_bytes(raw: &[u8]) -> Vec<(usize, f32)> {
    let mut floats: Vec<(usize, f32)> = raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .enumerate()
        .collect();
    floats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    floats.into_iter().take(5).collect()
}

fn load_labels(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| l.trim().to_string())
        .collect()
}

fn print_top5(raw: &[u8], labels: &[String]) {
    let top5 = top5_from_f32_bytes(raw);
    println!("\n┌─────────────────────────────────────────┐");
    println!("│           TOP-5 PREDICTIONS             │");
    println!("├─────────────────────────────────────────┤");
    for (rank, (idx, score)) in top5.iter().enumerate() {
        let label = labels.get(*idx)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        println!("│ #{} [{:>4}] {:<25} {:.4} │",
            rank + 1, idx, label, score);
    }
    println!("└─────────────────────────────────────────┘\n");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let default_level = if cli.debug { "debug" } else { "info" };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    tracing_subscriber::fmt().with_env_filter(filter).try_init().ok();

    // Load labels if provided
    let labels = cli.labels.as_deref()
        .map(load_labels)
        .unwrap_or_default();

    info!(server = %cli.server, camera = %cli.camera, "Magna Edge Camera Client starting");

    let mut reconnect_count = 0;
    loop {
        match run_inference_loop(&cli, &labels).await {
            Ok(()) => {
                info!("[Client] Inference loop completed normally");
                break;
            }
            Err(e) => {
                error!(error = %e, "Connection lost");
                reconnect_count += 1;
                if cli.max_reconnects > 0 && reconnect_count >= cli.max_reconnects {
                    error!("Max reconnection attempts reached — exiting");
                    return Err(e);
                }
                warn!(delay_secs = cli.reconnect_delay, attempt = reconnect_count, "Reconnecting...");
                tokio::time::sleep(Duration::from_secs(cli.reconnect_delay)).await;
            }
        }
    }
    Ok(())
}

async fn run_inference_loop(
    cli: &Cli,
    labels: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    info!("[Client] Connecting to {}...", cli.server);
    let mut client = InferenceServiceClient::connect(cli.server.clone()).await?;
    info!("[Client] Connected!");

    let is_file = std::path::Path::new(&cli.camera).extension().is_some();

    if is_file {
        info!("[Client] File mode: sending '{}'", cli.camera);

        let img = image::open(&cli.camera)?;
        let rgb = img.to_rgb8();
        let tensor = imagenet::preprocess_raw_rgb(
            rgb.as_raw(), rgb.width(), rgb.height(), Precision::FP32)?;

        let req = Request::new(InferenceRequest {
            inputs: vec![InferInputTensor {
                name: "input".to_string(),
                datatype: "FP32".to_string(),
                shape: tensor.shape.iter().map(|&s| s as i64).collect(),
                raw_data: tensor.data,
            }],
            model_name: String::new(),
        });

        let response = client.infer(req).await?;
        let inner = response.into_inner();

        info!("[Client] Received {} outputs in {:.1} ms",
            inner.outputs.len(), inner.inference_time_ms);

        for (i, out) in inner.outputs.iter().enumerate() {
            info!("   Output {}: '{}' ({}) shape={:?} elements={}",
                i, out.name, out.datatype, out.shape,
                out.raw_data.len() / 4);

            // Print top-5 classifications
            if !out.raw_data.is_empty() {
                print_top5(&out.raw_data, labels);
            }
        }
    } else {
        info!("[Client] Camera mode: streaming from '{}'", cli.camera);

        #[cfg(feature = "camera")]
        {
            use crate::input::camera::CameraInputProvider;
            use crate::input::provider::InputProvider;

            let mut camera = CameraInputProvider::new(&cli.camera)?;
            let mut frame_count = 0u64;
            let start_time = Instant::now();

            loop {
                match camera.next_frame() {
                    Ok(Some(frame)) => {
                        let tensor = imagenet::preprocess_raw_rgb(
                            &frame.rgb_data, frame.width, frame.height, Precision::FP32)?;

                        let req = Request::new(InferenceRequest {
                            inputs: vec![InferInputTensor {
                                name: "input".to_string(),
                                datatype: "FP32".to_string(),
                                shape: tensor.shape.iter().map(|&s| s as i64).collect(),
                                raw_data: tensor.data,
                            }],
                            model_name: String::new(),
                        });

                        match client.infer(req).await {
                            Ok(response) => {
                                let inner = response.into_inner();
                                info!("[Client] Frame {}: {:.1}ms",
                                    frame_count, inner.inference_time_ms);
                                if let Some(out) = inner.outputs.first() {
                                    print_top5(&out.raw_data, labels);
                                }
                            }
                            Err(e) => {
                                error!("[Client] gRPC error: {}", e);
                                return Err(e.into());
                            }
                        }

                        frame_count += 1;
                        if frame_count % 30 == 0 {
                            let fps = frame_count as f64 / start_time.elapsed().as_secs_f64();
                            info!("[Client] Throughput: {:.1} FPS ({} frames)", fps, frame_count);
                        }
                    }
                    Ok(None) => { info!("[Client] Camera stream ended"); break; }
                    Err(e) => { error!("[Client] Camera error: {}", e); return Err(e.into()); }
                }
            }
        }

        #[cfg(not(feature = "camera"))]
        {
            error!("[Client] Camera support requires --features camera");
            return Err("Camera feature not enabled".into());
        }
    }
    Ok(())
}
