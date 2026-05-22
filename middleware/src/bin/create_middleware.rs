// =============================================================================
// Magna Middleware — AOT Edge AI Compiler / Builder
// =============================================================================
//! CLI tool that takes a generic ONNX model, converts it for a target hardware
//! platform, and packages a ready-to-deploy inference server.
//!
//! Usage:
//! ```bash
//! create_middleware \
//!     -m model.onnx \
//!     --hw ORIN \
//!     -o middleware_server \
//!     --om model_orin.engine
//! ```
//!
//! The tool:
//! 1. Detects / validates the target hardware
//! 2. Checks dependencies
//! 3. Converts the ONNX model to hardware-optimized format
//! 4. Cross-compiles the middleware server binary
//! 5. Outputs the optimized model + server ready for deployment

use clap::Parser;
use std::process::Command;
use std::time::Instant;
use tracing::{error, info, warn};

use magna_middleware::hardware::dependency;
use magna_middleware::hardware::profile::Vendor;
use magna_middleware::utils::errors::Precision;
use magna_middleware::utils::logging::{init_logger, LogConfig, LogFormat};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "create_middleware",
    about = "Magna AOT Edge AI Compiler — Build hardware-optimized inference middleware"
)]
struct Cli {
    /// Path to the input ONNX model.
    #[arg(short = 'm', long)]
    model: String,

    /// Target hardware: ORIN, THOR, QUALCOMM, SNAPDRAGON, TDA4, CPU.
    #[arg(long = "hw")]
    hardware: String,

    /// Output path for the middleware server binary.
    #[arg(short = 'o', long)]
    output_bin: String,

    /// Output path for the hardware-optimized model.
    #[arg(long = "om")]
    output_model: String,

    /// Precision for model conversion: fp32, fp16, int8.
    #[arg(short, long, default_value = "fp16")]
    precision: String,

    /// Skip dependency checks.
    #[arg(long)]
    skip_deps: bool,

    /// Cargo build target triple for cross-compilation
    /// (e.g. aarch64-unknown-linux-gnu).
    #[arg(long)]
    target: Option<String>,

    /// Enable verbose compiler logging.
    #[arg(short, long)]
    verbose: bool,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let start = Instant::now();
    let cli = Cli::parse();

    init_logger(LogConfig {
        format: if cli.verbose {
            LogFormat::Pretty
        } else {
            LogFormat::Compact
        },
        default_level: if cli.verbose { "debug" } else { "info" }.into(),
    });

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          Magna — AOT Edge AI Middleware Compiler            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Validate inputs
    info!(step = "1/5", "Validating inputs");
    if !std::path::Path::new(&cli.model).exists() {
        error!(model = %cli.model, "Input model not found");
        error!("Please provide a valid ONNX model file.");
        std::process::exit(1);
    }
    info!(model = %cli.model, hardware = %cli.hardware, precision = %cli.precision, output_bin = %cli.output_bin, output_model = %cli.output_model, "Build configuration");

    let precision: Precision = match cli.precision.parse() {
        Ok(p) => p,
        Err(e) => {
            error!(error = %e, "Invalid precision");
            std::process::exit(1);
        }
    };

    // Step 2: Detect hardware / validate target
    info!(step = "2/5", "Detecting hardware");
    let vendor = match cli.hardware.to_uppercase().as_str() {
        "ORIN" | "NVIDIA" | "THOR" => {
            info!(step = "2/5", target = %cli.hardware.to_uppercase(), "Target platform: NVIDIA");
            Vendor::Nvidia
        }
        "QUALCOMM" | "SNAPDRAGON" | "QNN" | "SNPE" => {
            info!(
                step = "2/5",
                target = "qualcomm",
                "Target platform: Qualcomm Snapdragon"
            );
            Vendor::Qualcomm
        }
        "TDA4" | "TI" | "JACINTO" | "TIDL" => {
            info!(
                step = "2/5",
                target = "ti",
                "Target platform: Texas Instruments TDA4"
            );
            Vendor::TexasInstruments
        }
        "CPU" => {
            info!(step = "2/5", target = "cpu", "Target platform: Generic CPU");
            Vendor::Cpu
        }
        _ => {
            error!(hardware = %cli.hardware, "Unsupported hardware target");
            error!("Supported: ORIN, THOR, QUALCOMM, TDA4, CPU");
            std::process::exit(1);
        }
    };

    // Step 3: Check dependencies
    if !cli.skip_deps {
        info!(step = "3/5", "Checking dependencies");
        match dependency::check_dependencies(vendor) {
            Ok(statuses) => {
                let found = statuses.iter().filter(|s| s.found).count();
                let total = statuses.len();
                info!(step = "3/5", found, total, "Dependency check complete");
            }
            Err(e) => {
                warn!(step = "3/5", error = %e, "Dependency check warning — some features may not be available");
            }
        }
    } else {
        info!(step = "3/5", "Skipping dependency check (--skip-deps)");
    }

    // Step 4: Convert model to hardware-optimized format
    info!(step = "4/5", target = %cli.hardware.to_uppercase(), "Converting model");
    let conversion_ok = match vendor {
        Vendor::Nvidia => convert_nvidia(&cli.model, &cli.output_model, precision, cli.verbose),
        Vendor::Qualcomm => convert_qualcomm(&cli.model, &cli.output_model, precision, cli.verbose),
        Vendor::TexasInstruments => {
            convert_ti(&cli.model, &cli.output_model, precision, cli.verbose)
        }
        Vendor::Cpu => convert_cpu(&cli.model, &cli.output_model, cli.verbose),
        _ => {
            error!("Unknown vendor for conversion");
            false
        }
    };

    if !conversion_ok {
        error!("Model conversion failed!");
        error!("The optimized model was NOT generated.");
        error!("Please check that the appropriate SDK tools are installed.");
        std::process::exit(1);
    }

    info!(step = "4/5", output = %cli.output_model, "Optimized model written");

    // Step 5: Build the middleware server binary
    info!(step = "5/5", "Building middleware server binary");
    let build_ok = build_server(&cli.output_bin, vendor, cli.target.as_deref(), cli.verbose);

    if !build_ok {
        error!("Server binary build failed!");
        std::process::exit(1);
    }

    let elapsed = start.elapsed();
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Build complete in {:.1}s", elapsed.as_secs_f64());
    println!("║");
    println!("║  Optimized model: {}", cli.output_model);
    println!("║  Server binary:   {}", cli.output_bin);
    println!("║");
    println!(
        "║  To deploy, copy both files to the {} device and run:",
        cli.hardware.to_uppercase()
    );
    println!(
        "║    ./{} --model {} --backend {}",
        cli.output_bin,
        cli.output_model,
        match vendor {
            Vendor::Nvidia => "nvidia",
            Vendor::Qualcomm => "qualcomm",
            Vendor::TexasInstruments => "ti",
            _ => "cpu",
        }
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

// ---------------------------------------------------------------------------
// Model conversion per hardware
// ---------------------------------------------------------------------------

fn convert_nvidia(model: &str, output: &str, precision: Precision, verbose: bool) -> bool {
    info!("  Running TensorRT engine compilation via trtexec...");

    let precision_flag = match precision {
        Precision::FP16 => "--fp16",
        Precision::INT8 => "--int8",
        Precision::FP8 => "--fp8",
        _ => "", // FP32 is default
    };

    let mut cmd = Command::new("trtexec");
    cmd.arg(format!("--onnx={}", model))
        .arg(format!("--saveEngine={}", output));

    if !precision_flag.is_empty() {
        cmd.arg(precision_flag);
    }

    if verbose {
        cmd.arg("--verbose");
    }

    info!(
        "  Command: trtexec --onnx={} --saveEngine={} {}",
        model, output, precision_flag
    );

    match cmd.status() {
        Ok(status) => {
            if status.success() {
                info!("  TensorRT compilation succeeded");
                true
            } else {
                error!(status = ?status, "  trtexec exited with error");
                false
            }
        }
        Err(e) => {
            error!(error = %e, "  Failed to run trtexec");
            error!("  Ensure TensorRT is installed and trtexec is in PATH.");
            false
        }
    }
}

fn convert_qualcomm(model: &str, output: &str, precision: Precision, _verbose: bool) -> bool {
    info!("  Converting ONNX model for Qualcomm QNN / SNPE...");

    let arch_str = if cfg!(target_arch = "aarch64") {
        "aarch64-linux-clang"
    } else {
        "x86_64-linux-clang"
    };

    // Try QNN first
    if let Ok(qnn_root) = std::env::var("QNN_SDK_ROOT") {
        let converter = format!("{}/bin/{}/qnn-onnx-converter", qnn_root, arch_str);
        info!(
            "  Command: {} --input_network {} --output_path {}",
            converter, model, output
        );

        let mut cmd = Command::new(&converter);
        cmd.arg("--input_network")
            .arg(model)
            .arg("--output_path")
            .arg(output);

        if precision == Precision::FP16 {
            cmd.arg("--float_bitwidth").arg("16");
        }

        match cmd.status() {
            Ok(status) if status.success() => {
                info!("  QNN model conversion succeeded");
                return true;
            }
            Ok(status) => error!(status = ?status, "  QNN converter exited with error"),
            Err(e) => error!(error = %e, "  Failed to run QNN converter"),
        }
    }

    // Try SNPE
    if let Ok(snpe_root) = std::env::var("SNPE_ROOT") {
        let converter = format!("{}/bin/{}/snpe-onnx-to-dlc", snpe_root, arch_str);
        info!(converter = %converter, model, output, "  Command");

        let mut cmd = Command::new(&converter);
        cmd.arg("-d").arg(model).arg("-o").arg(output);

        match cmd.status() {
            Ok(status) if status.success() => {
                info!("  SNPE model conversion succeeded");
                return true;
            }
            Ok(status) => error!(status = ?status, "  SNPE converter exited with error"),
            Err(e) => error!(error = %e, "  Failed to run SNPE converter"),
        }
    }

    error!("  No Qualcomm SDK found (set QNN_SDK_ROOT or SNPE_ROOT)");
    false
}

fn convert_ti(model: &str, output: &str, _precision: Precision, _verbose: bool) -> bool {
    info!("  Converting ONNX model for TI TIDL...");

    if let Ok(tidl_path) = std::env::var("TIDL_TOOLS_PATH") {
        let compiler = format!("{}/tidl_model_import.out", tidl_path);
        info!(compiler = %compiler, model, output, "  Command");

        let mut cmd = Command::new(&compiler);
        cmd.arg(model).arg(output);

        match cmd.status() {
            Ok(status) if status.success() => {
                info!("  TIDL model compilation succeeded");
                return true;
            }
            Ok(status) => error!(status = ?status, "  TIDL compiler exited with error"),
            Err(e) => error!(error = %e, "  Failed to run TIDL compiler"),
        }
    }

    error!("  TIDL_TOOLS_PATH not set. Cannot convert model for TDA4.");
    error!("  Install TI Edge AI SDK and set TIDL_TOOLS_PATH.");
    false
}

fn convert_cpu(model: &str, output: &str, _verbose: bool) -> bool {
    info!("  CPU mode — copying ONNX model as-is (no conversion needed)");

    match std::fs::copy(model, output) {
        Ok(bytes) => {
            info!(bytes, output, "  Copied model");
            true
        }
        Err(e) => {
            error!(error = %e, "  Failed to copy model");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Build server binary
// ---------------------------------------------------------------------------

fn build_server(output: &str, vendor: Vendor, target: Option<&str>, _verbose: bool) -> bool {
    info!("  Building magna_server binary...");

    let feature = match vendor {
        Vendor::Nvidia => "nvidia",
        Vendor::Qualcomm => "qualcomm",
        Vendor::TexasInstruments => "ti",
        _ => "",
    };

    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .arg("--release")
        .arg("--bin")
        .arg("magna_server");

    if !feature.is_empty() {
        cmd.arg("--features").arg(feature);
    }

    if let Some(t) = target {
        cmd.arg("--target").arg(t);
        info!(target = t, "  Cross-compiling");
    }

    info!(
        "  Command: cargo build --release --bin magna_server{}{}",
        if !feature.is_empty() {
            format!(" --features {}", feature)
        } else {
            String::new()
        },
        target
            .map(|t| format!(" --target {}", t))
            .unwrap_or_default()
    );

    match cmd.status() {
        Ok(status) if status.success() => {
            // Copy the binary to the output path
            let source = if let Some(t) = target {
                format!("target/{}/release/magna_server", t)
            } else {
                "target/release/magna_server".into()
            };

            // Add .exe on Windows
            #[cfg(target_os = "windows")]
            let source = format!("{}.exe", source);
            #[cfg(target_os = "windows")]
            let output = format!("{}.exe", output);

            match std::fs::copy(&source, output) {
                Ok(bytes) => {
                    info!(output, bytes, "  Server binary built");
                    true
                }
                Err(e) => {
                    error!(error = %e, source = %source, "  Failed to copy binary");
                    false
                }
            }
        }
        Ok(status) => {
            error!(status = ?status, "  Cargo build failed");
            false
        }
        Err(e) => {
            error!(error = %e, "  Failed to run cargo");
            false
        }
    }
}
