// =============================================================================
// Magna Middleware — TensorRT C API Wrapper
// =============================================================================
// Thin C-compatible API around TensorRT for FFI from Rust.
//
// IMPORTANT: This is ONLY compiled when the `nvidia` feature is enabled.
// The build.rs feature-gates this compilation.

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>

using namespace nvinfer1;

// ---------------------------------------------------------------------------
// CUDA Error Checking Macro (H10 fix)
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "[TRT-C++] CUDA error in " << #call << ": "       \
                      << cudaGetErrorString(err) << std::endl;              \
            return -1;                                                      \
        }                                                                   \
    } while (0)

#define CUDA_CHECK_PTR(call)                                                \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "[TRT-C++] CUDA error in " << #call << ": "       \
                      << cudaGetErrorString(err) << std::endl;              \
            return nullptr;                                                 \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// TRT Logger
// ---------------------------------------------------------------------------

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT-C++] " << msg << std::endl;
        }
    }
} gLogger;

// ---------------------------------------------------------------------------
// TrtContext — holds all TRT and CUDA state
// ---------------------------------------------------------------------------

struct TrtContext {
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    void* d_input;
    void* d_output;
    int input_size;   // number of input elements
    int output_size;  // number of output elements
    int in_elem_size; // bytes per input element
    int out_elem_size; // bytes per output element
    cudaStream_t stream;
    int in_idx;
    int out_idx;
};

extern "C" {

// ---------------------------------------------------------------------------
// Load a serialized TRT engine from a file path (with length)
// ---------------------------------------------------------------------------

void* trt_load_engine(const uint8_t* path_ptr, size_t path_len) {
    // Construct a null-terminated string from the Rust &[u8]
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);

    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[TRT-C++] Engine file not found: " << path << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> trtModelStream(size);
    file.read(trtModelStream.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "[TRT-C++] Failed to create InferRuntime" << std::endl;
        return nullptr;
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
    if (!engine) {
        std::cerr << "[TRT-C++] Failed to deserialize engine" << std::endl;
        delete runtime;
        return nullptr;
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "[TRT-C++] Failed to create execution context" << std::endl;
        delete engine;
        delete runtime;
        return nullptr;
    }

    TrtContext* ctx = new TrtContext();
    ctx->runtime = runtime;
    ctx->engine = engine;
    ctx->context = context;

    // Discover I/O tensor indices and shapes dynamically
    ctx->in_idx = -1;
    ctx->out_idx = -1;
    const char* in_name = nullptr;
    const char* out_name = nullptr;

    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            ctx->in_idx = i;
            in_name = name;
        } else if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
            ctx->out_idx = i;
            out_name = name;
        }
    }

    if (ctx->in_idx < 0 || ctx->out_idx < 0) {
        std::cerr << "[TRT-C++] Engine has no input/output tensors" << std::endl;
        delete context;
        delete engine;
        delete runtime;
        delete ctx;
        return nullptr;
    }

    // Read shapes dynamically from the engine (no hardcoded 224×224)
    auto in_dims = engine->getTensorShape(in_name);
    auto out_dims = engine->getTensorShape(out_name);

    ctx->input_size = 1;
    for (int i = 0; i < in_dims.nbDims; i++) {
        if (in_dims.d[i] > 0) ctx->input_size *= in_dims.d[i];
    }

    ctx->output_size = 1;
    for (int i = 0; i < out_dims.nbDims; i++) {
        if (out_dims.d[i] > 0) ctx->output_size *= out_dims.d[i];
    }

    // Determine element sizes based on data type
    ctx->in_elem_size = 4; // default FP32
    DataType in_type = engine->getTensorDataType(in_name);
    if (in_type == DataType::kHALF) ctx->in_elem_size = 2;
    else if (in_type == DataType::kINT8) ctx->in_elem_size = 1;

    ctx->out_elem_size = 4; // default FP32
    DataType out_type = engine->getTensorDataType(out_name);
    if (out_type == DataType::kHALF) ctx->out_elem_size = 2;
    else if (out_type == DataType::kINT8) ctx->out_elem_size = 1;

    // Allocate device memory with error checking (H10 fix)
    CUDA_CHECK_PTR(cudaMalloc(&ctx->d_input, ctx->input_size * ctx->in_elem_size));
    CUDA_CHECK_PTR(cudaMalloc(&ctx->d_output, ctx->output_size * ctx->out_elem_size));
    CUDA_CHECK_PTR(cudaStreamCreate(&ctx->stream));

    std::cout << "[TRT-C++] Engine loaded: input=" << ctx->input_size
              << " output=" << ctx->output_size
              << " in_elem=" << ctx->in_elem_size
              << " out_elem=" << ctx->out_elem_size << std::endl;

    return (void*)ctx;
}

// ---------------------------------------------------------------------------
// Allocate buffers (called separately from Rust)
// ---------------------------------------------------------------------------

int trt_allocate_buffers(void* handle) {
    // Buffers are already allocated in trt_load_engine.
    // This exists for API symmetry with the Rust InferenceBackend trait.
    if (!handle) return -1;
    return 0;
}

// ---------------------------------------------------------------------------
// Destroy engine and free all resources
// ---------------------------------------------------------------------------

void trt_destroy_engine(void* handle) {
    if (!handle) return;
    TrtContext* ctx = (TrtContext*)handle;

    if (ctx->stream) {
        cudaStreamSynchronize(ctx->stream);
        cudaFree(ctx->d_input);
        cudaFree(ctx->d_output);
        cudaStreamDestroy(ctx->stream);
    }

    if (ctx->context) delete ctx->context;
    if (ctx->engine) delete ctx->engine;
    if (ctx->runtime) delete ctx->runtime;

    delete ctx;
    std::cout << "[TRT-C++] Engine destroyed" << std::endl;
}

// ---------------------------------------------------------------------------
// Query tensor sizes
// ---------------------------------------------------------------------------

int trt_get_input_elems(void* handle) {
    if (!handle) return 0;
    return ((TrtContext*)handle)->input_size;
}

int trt_get_output_elems(void* handle) {
    if (!handle) return 0;
    return ((TrtContext*)handle)->output_size;
}

// ---------------------------------------------------------------------------
// Run inference with CUDA error checking
// ---------------------------------------------------------------------------

int trt_infer(void* handle,
              const uint8_t* input, size_t input_bytes,
              uint8_t* output, size_t output_bytes) {
    if (!handle) return -1;
    TrtContext* ctx = (TrtContext*)handle;

    // Validate sizes
    size_t expected_in = (size_t)ctx->input_size * ctx->in_elem_size;
    if (input_bytes != expected_in && input_bytes != (size_t)ctx->input_size * 4) {
        std::cerr << "[TRT-C++] Input size mismatch: got " << input_bytes
                  << " expected " << expected_in << std::endl;
        return -3;
    }

    // Copy input to device
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_input, input, input_bytes,
                               cudaMemcpyHostToDevice, ctx->stream));

    // Set tensor addresses
    const char* in_name = ctx->engine->getIOTensorName(ctx->in_idx);
    const char* out_name = ctx->engine->getIOTensorName(ctx->out_idx);
    ctx->context->setTensorAddress(in_name, ctx->d_input);
    ctx->context->setTensorAddress(out_name, ctx->d_output);

    // If the input shape is dynamic, we must set it explicitly before enqueue
    auto in_dims = ctx->engine->getTensorShape(in_name);
    bool dynamic = false;
    for (int i = 0; i < in_dims.nbDims; i++) {
        if (in_dims.d[i] == -1) {
            in_dims.d[i] = 1; // Fallback to batch 1
            dynamic = true;
        }
    }
    if (dynamic) {
        ctx->context->setInputShape(in_name, in_dims);
    }

    // Execute
    bool status = ctx->context->enqueueV3(ctx->stream);
    if (!status) {
        std::cerr << "[TRT-C++] enqueueV3 failed" << std::endl;
        return -2;
    }

    // Copy output back to host
    size_t copy_bytes = std::min(output_bytes,
                                (size_t)ctx->output_size * ctx->out_elem_size);
    CUDA_CHECK(cudaMemcpyAsync(output, ctx->d_output, copy_bytes,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    return 0;
}

// ---------------------------------------------------------------------------
// Get device memory usage (using actual element sizes, not just sizeof(float))
// ---------------------------------------------------------------------------

long long trt_get_device_memory(void* handle) {
    if (!handle) return 0;
    TrtContext* ctx = (TrtContext*)handle;

    long long total = (long long)ctx->engine->getDeviceMemorySize();
    // Use actual element sizes, not sizeof(float) (M8 fix)
    total += (long long)ctx->input_size * ctx->in_elem_size;
    total += (long long)ctx->output_size * ctx->out_elem_size;
    return total;
}

} // extern "C"
