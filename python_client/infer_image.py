import grpc
import sys
import json
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/sanika/mi-isal-main/python_client')
import magna_pb2
import magna_pb2_grpc

SERVER      = "10.9.2.16:50051"
IMAGE_PATH  = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
LABELS_PATH = "/home/sanika/mi-isal-main/imagenet_class_index.json"

# ── Preprocess ────────────────────────────────────────────────────────────────
img  = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224))
arr  = np.array(img).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
arr  = (arr - mean) / std
arr  = arr.transpose(2, 0, 1)                          # HWC → CHW
arr  = np.ascontiguousarray(arr[np.newaxis], dtype=np.float32)  # → NCHW

# ── Build request (matches proto exactly) ─────────────────────────────────────
tensor = magna_pb2.InferInputTensor(
    name="input",
    datatype="FP32",
    shape=list(arr.shape),   # [1, 3, 224, 224]
    raw_data=arr.tobytes()
)

request = magna_pb2.InferenceRequest(
    inputs=[tensor],
    model_name="mobilenetv2"
)

# ── Send ──────────────────────────────────────────────────────────────────────
print(f"Connecting to {SERVER} ...")
channel = grpc.insecure_channel(SERVER)
stub    = magna_pb2_grpc.InferenceServiceStub(channel)

# Health check first
health = stub.HealthCheck(magna_pb2.HealthRequest())
print(f"Server ready={health.ready}  state={health.state}  backend={health.backend}")

# Infer
response = stub.Infer(request)
print(f"Inference time reported by server: {response.inference_time_ms:.2f} ms")

# ── Parse output ──────────────────────────────────────────────────────────────
out_tensor = response.outputs[0]
out = np.frombuffer(out_tensor.raw_data, dtype=np.float32)
top5_idx = np.argsort(out)[::-1][:5]

# Load labels
with open(LABELS_PATH) as f:
    labels = json.load(f)

print(f"\nImage : {IMAGE_PATH}")
print(f"{'Rank':<6} {'ClassID':<10} {'Label':<35} {'Score':<10}")
print("-" * 63)
for rank, idx in enumerate(top5_idx):
    label = labels.get(str(idx), ["?", "unknown"])[1]
    print(f"{rank+1:<6} {idx:<10} {label:<35} {out[idx]:.4f}")
