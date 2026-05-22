# Magna Middleware — TI TDA4 MobileNetV2 Inference

## Benchmark Results

| Metric | Value |
|---|---|
| Model | MobileNetV2 (TVM + TIDL quantized) |
| Hardware | C7x DSP + MMA |
| Images | 49,633 |
| Top-1 Accuracy | 70.21% |
| Top-5 Accuracy | 89.57% |
| Mean Latency | 2.68 ms |
| FPS | 372.61 |
| Model Size | 20.70 MB |
| Precision | UINT8 (TVM quantized) |

---

## Step 1 — Setup WSL2 (Ubuntu 22.04)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl git pkg-config \
    libssl-dev cmake protobuf-compiler \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    gcc-12-aarch64-linux-gnu g++-12-aarch64-linux-gnu \
    libclang-dev

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup target add aarch64-unknown-linux-gnu
```

---

## Step 2 — Copy Board Libraries to WSL2 Sysroot

```bash
mkdir -p ~/tda4-sysroot/usr/lib
mkdir -p ~/tda4-sysroot/usr/include

BOARD=root@10.9.2.16

scp ${BOARD}:/usr/lib/libdlr.so        ~/tda4-sysroot/usr/lib/
scp ${BOARD}:/usr/lib/libstdc++.so.6   ~/tda4-sysroot/usr/lib/
scp ${BOARD}:/usr/lib/libc.so.6        ~/tda4-sysroot/usr/lib/
scp ${BOARD}:/usr/lib/libm.so.6        ~/tda4-sysroot/usr/lib/
scp ${BOARD}:/usr/lib/libgcc_s.so.1    ~/tda4-sysroot/usr/lib/
scp ${BOARD}:/usr/include/dlr.h        ~/tda4-sysroot/usr/include/
scp ${BOARD}:/usr/include/dlr_common.h ~/tda4-sysroot/usr/include/
```

---

## Step 3 — Clone Repository

```bash
cd ~
git clone <your-repo-url> mi-isal-main
cd mi-isal-main
```

---

## Step 4 — Set Build Environment Variables

```bash
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc-12
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc-12
export TDA4_SYSROOT=$HOME/tda4-sysroot
export RUSTFLAGS="-L ${TDA4_SYSROOT}/usr/lib \
    -C link-arg=-Wl,-rpath-link,${TDA4_SYSROOT}/usr/lib \
    -C link-arg=-Wl,--allow-shlib-undefined"
```

---

## Step 5 — Build

```bash
cd ~/mi-isal-main

cargo build --release --features ti --target aarch64-unknown-linux-gnu

cargo build --release --features ti --target aarch64-unknown-linux-gnu --bin magna-client
```

---

## Step 6 — Deploy to Board

```bash
BOARD=root@10.9.2.16

scp target/aarch64-unknown-linux-gnu/release/magna_server  ${BOARD}:/usr/local/bin/
scp target/aarch64-unknown-linux-gnu/release/magna         ${BOARD}:/usr/local/bin/
scp target/aarch64-unknown-linux-gnu/release/benchmark     ${BOARD}:/usr/local/bin/
scp target/aarch64-unknown-linux-gnu/release/magna-client  ${BOARD}:/usr/local/bin/
```

---

## Step 7 — Run on Board

**Single image:**
```bash
ssh root@10.9.2.16

magna \
    --model /opt/model_zoo/TVM-CL-3090-mobileNetV2-tv/artifacts \
    --image /path/to/image.jpg \
    --backend ti \
    --labels /opt/model_zoo/TVM-CL-3090-mobileNetV2-tv/imagenet_classes.txt
```

**gRPC server (Terminal 1):**
```bash
magna_server \
    --model /opt/model_zoo/TVM-CL-3090-mobileNetV2-tv/artifacts \
    --backend ti \
    --address 0.0.0.0:50051 \
    --labels /opt/model_zoo/TVM-CL-3090-mobileNetV2-tv/imagenet_classes.txt
```

**gRPC client (Terminal 2):**
```bash
magna-client \
    --camera /path/to/image.jpg \
    --server http://127.0.0.1:50051 \
    --labels /opt/model_zoo/TVM-CL-3090-mobileNetV2-tv/imagenet_classes.txt
```

**Full ImageNet benchmark:**
```bash
benchmark \
    --model /opt/model_zoo/TVM-CL-3090-mobileNetV2-tv/artifacts \
    --dataset /opt/model_final/imagenet-val \
    --backend ti \
    --warmup 10
```

---

## Step 8 — Push to GitHub (New Branch, Isolated)

```bash
cd ~/mi-isal-main

git config user.name "your-github-username"
git config user.email "your-email@example.com"

git checkout -b ti-tda4-backend

git add .
git commit -m "feat: TI TDA4 backend — MobileNetV2 DLR inference via C7x DSP"

git push --set-upstream origin ti-tda4-backend
```

> When prompted for password, use a GitHub Personal Access Token:  
> GitHub → Settings → Developer settings → Personal access tokens → Generate new token → tick `repo`
