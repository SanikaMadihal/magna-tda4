// =============================================================================
// Magna Middleware ΓÇö gRPC Inference Server
// =============================================================================
//! Production-grade gRPC server that wraps the Middleware public API.
//!
//! Features:
//! - Handles RGB, BGR, and JPEG image formats
//! - Returns raw output tensor + optional classification
//! - Measures and reports inference latency
//! - Health check endpoint
//! - Graceful shutdown on SIGTERM/Ctrl+C
//! - Configurable max request size

use clap::Parser;
use std::net::SocketAddr;
use std::time::Instant;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{debug, error, info};

use magna_middleware::api::grpc::magna_grpc::inference_service_server::{
    InferenceService, InferenceServiceServer,
};
use magna_middleware::api::grpc::magna_grpc::{
    HealthRequest, HealthResponse, InferenceRequest, InferenceResponse,
};
use magna_middleware::api::public_api::Middleware;
use magna_middleware::inference::traits::TensorBuffer;
use magna_middleware::utils::errors::{MiddlewareConfig, Precision};
use magna_middleware::utils::logging::{init_logger, LogConfig, LogFormat};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "magna_server",
    about = "Magna Middleware gRPC Inference Server"
)]
struct Cli {
    /// Path to the compiled engine / model file.
    #[arg(short, long)]
    model: String,

    /// Backend selection: auto, nvidia, qualcomm, ti, cpu, simulated.
    #[arg(short, long, default_value = "auto")]
    backend: String,

    /// Precision: fp32, fp16, int8, fp8.
    #[arg(short, long, default_value = "fp32")]
    precision: String,

    /// gRPC listen address.
    #[arg(short, long, default_value = "127.0.0.1:50051")]
    address: String,

    /// Path to labels file (optional, enables classification output).
    #[arg(short, long)]
    labels: Option<String>,

    /// Maximum request size in bytes (default 10 MB).
    #[arg(long, default_value = "10485760")]
    max_request_size: usize,

    /// Enable debug logging.
    #[arg(short, long)]
    debug: bool,

    /// Log output format: pretty, json, compact.
    #[arg(long, default_value = "pretty")]
    log_format: String,
}

// ---------------------------------------------------------------------------
// Request correlation ID counter
// ---------------------------------------------------------------------------

/// Lock-free monotonic counter for per-request correlation IDs.
static REQUEST_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[inline]
fn next_request_id() -> u64 {
    REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Service implementation
// ---------------------------------------------------------------------------

struct MagnaInferenceService {
    middleware: Middleware,
}

#[tonic::async_trait]
impl InferenceService for MagnaInferenceService {
    async fn infer(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let req = request.into_inner();

        let req_id = next_request_id();
        debug!(
            request_id = req_id,
            input_count = req.inputs.len(),
            "Inference request received"
        );

        if req.inputs.is_empty() {
            return Err(Status::invalid_argument("inputs is empty"));
        }

        let mut buffers = Vec::new();
        for grpc_in in &req.inputs {
            let precision = match grpc_in.datatype.as_str() {
                "FP32" => Precision::FP32,
                "FP16" => Precision::FP16,
                "INT8" => Precision::INT8,
                "FP8" => Precision::FP8,
                _ => Precision::FP32,
            };

            buffers.push(TensorBuffer {
                name: grpc_in.name.clone(),
                data: grpc_in.raw_data.clone(),
                shape: grpc_in.shape.iter().map(|&s| s as usize).collect(),
                precision,
            });
        }

        // Run inference
        let start = Instant::now();
        let result = self.middleware.infer_generic(&buffers).map_err(|e| {
            error!(request_id = req_id, error = %e, "Inference failed");
            Status::internal(format!("Inference failed: {}", e))
        })?;
        let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;

        // Build response
        let mut resp_outputs = Vec::new();
        for tb in &result.outputs {
            let datatype = match tb.precision {
                Precision::FP32 => "FP32",
                Precision::FP16 => "FP16",
                Precision::INT8 => "INT8",
                Precision::FP8 => "FP8",
            }
            .to_string();

            resp_outputs.push(magna_middleware::api::grpc::magna_grpc::InferOutputTensor {
                name: tb.name.clone(),
                datatype,
                shape: tb.shape.iter().map(|&s| s as i64).collect(),
                raw_data: tb.data.clone(),
            });
        }

        let resp = InferenceResponse {
            outputs: resp_outputs,
            inference_time_ms: elapsed_ms,
        };

        debug!(
            request_id = req_id,
            output_count = resp.outputs.len(),
            latency_ms = elapsed_ms,
            "Inference complete"
        );
        Ok(Response::new(resp))
    }

    async fn health_check(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let state = self.middleware.state();
        let engine = self.middleware.engine_info();

        // inference_count is 0 when metrics feature is disabled
        #[cfg(feature = "metrics")]
        let inference_count = self.middleware.get_metrics().inference_count;
        #[cfg(not(feature = "metrics"))]
        let inference_count: u64 = 0;

        Ok(Response::new(HealthResponse {
            ready: state == magna_middleware::lifecycle::state_manager::State::Ready,
            state: format!("{:?}", state),
            backend: self.middleware.backend_name(),
            model_name: engine.map(|e| e.name).unwrap_or_default(),
            inference_count,
        }))
    }
}

// (Removed decode_image_data entirely due to new Generic grpc interface)

// ---------------------------------------------------------------------------
// Server startup
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    info!(
        backend   = %cli.backend,
        precision = %cli.precision,
        model     = %cli.model,
        address   = %cli.address,
        "Magna gRPC Inference Server starting"
    );

    let precision: Precision = cli
        .precision
        .parse()
        .map_err(|e| format!("Invalid precision: {}", e))?;

    let mw = Middleware::new();
    let config = MiddlewareConfig {
        backend: cli.backend,
        precision,
        warmup_runs: 1,
        debug: cli.debug,
        labels_path: cli.labels.clone(),
        model_path: Some(cli.model.clone()),
        grpc_address: cli.address.clone(),
    };

    mw.initialize(config)?;
    let engine_info = mw.load_engine(&cli.model)?;
    info!(
        engine  = %engine_info.name,
        inputs  = engine_info.inputs.len(),
        outputs = engine_info.outputs.len(),
        "Engine loaded"
    );

    let addr: SocketAddr = cli.address.parse()?;
    let service = MagnaInferenceService {
        middleware: mw.clone(),
    };

    // Configure server with request size limit
    let server = Server::builder().add_service(
        InferenceServiceServer::new(service).max_decoding_message_size(cli.max_request_size),
    );

    info!(address = %addr, "Listening for gRPC connections");

    // Graceful shutdown on Ctrl+C / SIGTERM
    let shutdown_mw = mw.clone();
    server
        .serve_with_shutdown(addr, async move {
            tokio::signal::ctrl_c().await.ok();
            info!("Shutdown signal received — cleaning up");
            if let Err(e) = shutdown_mw.shutdown() {
                error!(error = %e, "Shutdown error");
            }
            info!("Shutdown complete");
        })
        .await?;

    Ok(())
}
