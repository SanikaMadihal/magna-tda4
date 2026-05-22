//! Full ImageNet benchmark — DSP
//! Preprocessing matches Python benchmark_tda4vm.py exactly (uint8, resize 256, crop 224)

use clap::Parser;
use magna_middleware::inference::traits::TensorBuffer;
use magna_middleware::api::public_api::Middleware;
use magna_middleware::utils::errors::{MiddlewareConfig, Precision};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "benchmark", about = "Full ImageNet benchmark for TI TDA4 DSP")]
struct Cli {
    /// Path to TVM model artifacts directory
    #[arg(short, long)]
    model: String,

    /// Path to ImageNet validation dataset root
    #[arg(short, long)]
    dataset: String,

    /// Maximum images to benchmark (0 = all)
    #[arg(long, default_value = "0")]
    max_images: usize,

    /// Number of warmup iterations
    #[arg(short, long, default_value = "10")]
    warmup: usize,

    /// Backend to use
    #[arg(short, long, default_value = "ti")]
    backend: String,

    /// Report progress every N images
    #[arg(long, default_value = "1000")]
    report_every: usize,
}

fn preprocess_uint8(img_path: &Path) -> Option<Vec<u8>> {
    let img = image::open(img_path).ok()?;
    let img = img.resize_exact(256, 256, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let mut out = vec![0u8; 3 * 224 * 224];
    for y in 0..224usize {
        for x in 0..224usize {
            let px = img.get_pixel((x + 16) as u32, (y + 16) as u32);
            out[0 * 224 * 224 + y * 224 + x] = px[0];
            out[1 * 224 * 224 + y * 224 + x] = px[1];
            out[2 * 224 * 224 + y * 224 + x] = px[2];
        }
    }
    Some(out)
}

fn build_synset_map(dataset_root: &Path) -> std::collections::HashMap<String, usize> {
    let mut synsets: Vec<String> = std::fs::read_dir(dataset_root)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    synsets.sort();
    synsets.into_iter().enumerate().map(|(i, s)| (s, i)).collect()
}

fn load_records(
    dataset_root: &Path,
    synset_map: &std::collections::HashMap<String, usize>,
    max_images: usize,
) -> Vec<(PathBuf, usize)> {
    let mut records: Vec<(PathBuf, usize)> = Vec::new();
    let mut synsets: Vec<_> = synset_map.keys().cloned().collect();
    synsets.sort();
    for syn in synsets {
        let syn_dir = dataset_root.join(&syn);
        if !syn_dir.is_dir() { continue; }
        let gt = synset_map[&syn];
        let mut files: Vec<_> = std::fs::read_dir(&syn_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let n = e.file_name().to_string_lossy().to_uppercase();
                n.ends_with(".JPEG") || n.ends_with(".JPG")
            })
            .map(|e| e.path())
            .collect();
        files.sort();
        for f in files { records.push((f, gt)); }
    }
    if max_images > 0 && max_images < records.len() {
        let step = records.len() as f64 / max_images as f64;
        records = (0..max_images)
            .map(|i| records[(i as f64 * step) as usize].clone())
            .collect();
    }
    records
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((p / 100.0) * sorted.len() as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn dir_size(path: &Path) -> u64 {
    if path.is_file() {
        return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    std::fs::read_dir(path).map(|rd| {
        rd.filter_map(|e| e.ok()).map(|e| dir_size(&e.path())).sum()
    }).unwrap_or(0)
}

fn top5_from_output(raw: &[u8]) -> Vec<usize> {
    let mut floats: Vec<(usize, f32)> = raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .enumerate()
        .collect();
    floats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    floats.iter().take(5).map(|(i, _)| *i).collect()
}

fn print_report(
    label: &str,
    total: usize,
    top1: usize,
    top5: usize,
    mut latencies: Vec<f64>,
    total_secs: f64,
    model_path: &str,
    precision: &str,
) {
    if latencies.is_empty() {
        println!("No results for {}", label);
        return;
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean       = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let median     = percentile(&latencies, 50.0);
    let p95        = percentile(&latencies, 95.0);
    let p99        = percentile(&latencies, 99.0);
    let fps        = 1000.0 / mean;
    let throughput = total as f64 / total_secs;
    let model_mb   = dir_size(Path::new(model_path)) as f64 / 1_048_576.0;

    println!("\n{}", "=".repeat(62));
    println!("📊 BENCHMARK — {}", label);
    println!("{}", "=".repeat(62));
    println!("Model         : {}", model_path);
    println!("Model Size    : {:.2} MB", model_mb);
    println!("Precision     : {}", precision);
    println!("Images        : {}", total);
    println!("Top-1 Accuracy: {:.2}%", top1 as f64 / total as f64 * 100.0);
    println!("Top-5 Accuracy: {:.2}%", top5 as f64 / total as f64 * 100.0);
    println!("\nLatency (ms):");
    println!("  Mean        : {:.3}", mean);
    println!("  Median      : {:.3}", median);
    println!("  P95         : {:.3}", p95);
    println!("  P99         : {:.3}", p99);
    println!("  Min         : {:.3}", latencies.first().unwrap());
    println!("  Max         : {:.3}", latencies.last().unwrap());
    println!("\nPerformance:");
    println!("  FPS         : {:.2}", fps);
    println!("  Throughput  : {:.2} img/s", throughput);
    println!("{}\n", "=".repeat(62));
}

fn main() {
    let cli = Cli::parse();

    println!("🚀 TDA4 MobileNetV2 Benchmark");
    println!("   Model   : {}", cli.model);
    println!("   Dataset : {}", cli.dataset);
    println!("   Backend : {}", cli.backend);
    println!("   Warmup  : {}", cli.warmup);
    println!("   Max imgs: {}", if cli.max_images == 0 { "all".to_string() } else { cli.max_images.to_string() });

    println!("\nBuilding synset map...");
    let synset_map = build_synset_map(Path::new(&cli.dataset));
    println!("Loading records...");
    let records = load_records(Path::new(&cli.dataset), &synset_map, cli.max_images);
    println!("Total images : {}", records.len());

    println!("\n{}", "=".repeat(62));
    println!("🔧 Starting {} benchmark...", cli.backend);

    let mw = Middleware::new();
    let config = MiddlewareConfig {
        backend: cli.backend.clone(),
        precision: Precision::FP32,
        warmup_runs: cli.warmup,
        debug: false,
        labels_path: None,
        model_path: None,
        grpc_address: "127.0.0.1:50051".to_string(),
    };

    if let Err(e) = mw.initialize(config) {
        println!("❌ Init failed: {}", e);
        return;
    }
    if let Err(e) = mw.load_engine(&cli.model) {
        println!("❌ Engine load failed: {}", e);
        return;
    }
    println!("✅ Backend ready");

    println!("Warming up ({} iters)...", cli.warmup);
    let dummy = TensorBuffer {
        name: "input".to_string(),
        data: vec![128u8; 3 * 224 * 224],
        shape: vec![1, 3, 224, 224],
        precision: Precision::FP32,
    };
    for _ in 0..cli.warmup {
        let _ = mw.infer(&[dummy.clone()]);
    }

    println!("Running {} images...\n", records.len());
    let mut top1 = 0usize;
    let mut top5_count = 0usize;
    let mut latencies = Vec::with_capacity(records.len());
    let total_start = Instant::now();

    for (i, (img_path, gt)) in records.iter().enumerate() {
        let input_data = match preprocess_uint8(img_path) {
            Some(v) => v,
            None => continue,
        };

        let tensor = TensorBuffer {
            name: "input".to_string(),
            data: input_data,
            shape: vec![1, 3, 224, 224],
            precision: Precision::FP32,
        };

        let t0 = Instant::now();
        let output = match mw.infer(&[tensor]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("Infer error at {}: {}", img_path.display(), e);
                continue;
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        latencies.push(ms);

        let topk = top5_from_output(&output[0].data);
        if topk[0] == *gt { top1 += 1; }
        if topk.contains(gt) { top5_count += 1; }

        if (i + 1) % cli.report_every == 0 {
            println!("  {}/{} done  Top-1 so far: {:.2}%",
                i + 1, records.len(),
                top1 as f64 / (i + 1) as f64 * 100.0);
        }
    }

    let total_secs = total_start.elapsed().as_secs_f64();
    let dsp_mean = if !latencies.is_empty() {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    } else { 0.0 };

    mw.shutdown().ok();

    print_report(
        &format!("{} backend", cli.backend.to_uppercase()),
        latencies.len(), top1, top5_count,
        latencies, total_secs,
        &cli.model, "UINT8 (TVM quantized)",
    );

    // ARM CPU preprocessing cost — dynamic, based on actual images
    println!("Measuring ARM CPU preprocessing cost (1000 images)...");
    let mut pre_lats = Vec::with_capacity(1000);
    for (img_path, _) in records.iter().take(1000) {
        let t0 = Instant::now();
        let _ = preprocess_uint8(img_path);
        pre_lats.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    pre_lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pre_mean = pre_lats.iter().sum::<f64>() / pre_lats.len() as f64;

    println!("\n{}", "=".repeat(62));
    println!("📊 ARM CPU — Preprocessing Only (1000 images)");
    println!("{}", "=".repeat(62));
    println!("Mean preprocess latency : {:.3} ms", pre_mean);
    println!("DSP inference latency   : {:.3} ms", dsp_mean);
    println!("Preprocess % of total   : {:.1}%",
        pre_mean / (pre_mean + dsp_mean) * 100.0);
    println!("{}\n", "=".repeat(62));
}
