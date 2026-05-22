fn main() {
    // =========================
    // gRPC Proto Compilation (REQUIRED)
    // =========================
    tonic_build::configure()
        .build_server(true)
        .compile(
            &["proto/magna.proto"],   // path to your proto file
            &["proto"],               // include directory
        )
        .expect("Failed to compile protos");

    // =========================
    // CPU backend (no special linking)
    // =========================
    #[cfg(feature = "cpu")]
    {
        println!("cargo:warning=Building CPU backend");
    }

    // =========================
    // NVIDIA backend (keep placeholder)
    // =========================
    #[cfg(feature = "nvidia")]
    {
        println!("cargo:warning=Building NVIDIA backend");
        // Keep your existing TensorRT linking here if present
    }

    // =========================
    // TI backend (DLR + C shim)
    // =========================
    #[cfg(feature = "ti")]
    {
        println!("cargo:warning=Building TI backend (DLR)");

        // -------------------------
        // Sysroot detection
        // -------------------------
        let sysroot = std::env::var("TDA4_SYSROOT")
            .unwrap_or_else(|_| format!("{}/tda4-sysroot", std::env::var("HOME").unwrap()));

        println!("cargo:warning=Using sysroot: {}", sysroot);

        let dlr_header = format!("{}/usr/include/dlr.h", sysroot);

        // -------------------------
        // Only build if DLR exists
        // -------------------------
        if std::path::Path::new(&dlr_header).exists() {

            println!("cargo:warning=DLR headers found. Enabling full TI build.");

            // Compile C shim
            cc::Build::new()
                .file("dlr_c_api.c")
                .include(format!("{}/usr/include", sysroot))
                .include(format!("{}/edgeai/include", sysroot))
                .flag("-O2")
                .compile("dlr_c_api");

            // Link DLR + system libs
            println!("cargo:rustc-link-search={}/edgeai/lib", sysroot);
            println!("cargo:rustc-link-search={}/usr/lib", sysroot);

            println!("cargo:rustc-link-lib=dlr");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");

        } else {
            println!("cargo:warning=DLR not found. Skipping TI C backend build (expected without board)");
        }
    }
}
