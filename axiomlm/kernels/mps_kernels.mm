#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <iostream>

static id<MTLLibrary> load_metal_library(id<MTLDevice> device, NSString* srcPath) {
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:srcPath encoding:NSUTF8StringEncoding error:&error];
    if (!source) {
        std::cerr << "Failed to read Metal source: " << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        std::cerr << "Failed to compile Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
    }
    return library;
}

static id<MTLComputePipelineState> get_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* functionName) {
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (!function) {
        std::cerr << "Function " << [functionName UTF8String] << " not found in library" << std::endl;
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pso) {
        std::cerr << "Failed to create pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
    }
    return pso;
}

static id<MTLDevice> mtl_device = nil;
static id<MTLLibrary> mtl_library = nil;
static id<MTLComputePipelineState> rmsnorm_fwd_pso = nil;
static id<MTLComputePipelineState> rmsnorm_bwd_pso = nil;
static id<MTLComputePipelineState> rmsnorm_bwd_weight_pso = nil;
static id<MTLComputePipelineState> swiglu_fwd_pso = nil;
static id<MTLComputePipelineState> swiglu_bwd_pso = nil;
static id<MTLComputePipelineState> rope_pso = nil;
static id<MTLComputePipelineState> ce_fwd_pso = nil;
static id<MTLComputePipelineState> ce_bwd_pso = nil;
static bool is_initialized = false;

void init_mps(std::string metal_path) {
    if (!is_initialized) {
        mtl_device = MTLCreateSystemDefaultDevice();
        NSString *ns_path = [NSString stringWithUTF8String:metal_path.c_str()];
        mtl_library = load_metal_library(mtl_device, ns_path);
        if (mtl_library) {
            rmsnorm_fwd_pso = get_pipeline(mtl_device, mtl_library, @"rmsnorm_forward_kernel");
            rmsnorm_bwd_pso = get_pipeline(mtl_device, mtl_library, @"rmsnorm_backward_kernel");
            rmsnorm_bwd_weight_pso = get_pipeline(mtl_device, mtl_library, @"rmsnorm_backward_weight_kernel");
            swiglu_fwd_pso = get_pipeline(mtl_device, mtl_library, @"swiglu_forward_kernel");
            swiglu_bwd_pso = get_pipeline(mtl_device, mtl_library, @"swiglu_backward_kernel");
            rope_pso = get_pipeline(mtl_device, mtl_library, @"rope_kernel");
            ce_fwd_pso = get_pipeline(mtl_device, mtl_library, @"cross_entropy_forward_kernel");
            ce_bwd_pso = get_pipeline(mtl_device, mtl_library, @"cross_entropy_backward_kernel");
        }
        is_initialized = true;
    }
}

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_backward_mps(torch::Tensor grad_y, torch::Tensor x, torch::Tensor weight, torch::Tensor rsqrt_cache) {
    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(weight);
    
    int64_t D = x.size(-1);
    int64_t num_rows = x.numel() / D;

    id<MTLBuffer> gy_buf = getMTLBufferStorage(grad_y);
    id<MTLBuffer> x_buf = getMTLBufferStorage(x);
    id<MTLBuffer> w_buf = getMTLBufferStorage(weight);
    id<MTLBuffer> rsqrt_buf = getMTLBufferStorage(rsqrt_cache);
    id<MTLBuffer> gx_buf = getMTLBufferStorage(grad_x);
    id<MTLBuffer> gw_buf = getMTLBufferStorage(grad_w);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    // 1. grad_x computation
    [encoder setComputePipelineState:rmsnorm_bwd_pso];
    [encoder setBuffer:gy_buf offset:grad_y.storage_offset() * grad_y.element_size() atIndex:0];
    [encoder setBuffer:x_buf offset:x.storage_offset() * x.element_size() atIndex:1];
    [encoder setBuffer:w_buf offset:weight.storage_offset() * weight.element_size() atIndex:2];
    [encoder setBuffer:rsqrt_buf offset:rsqrt_cache.storage_offset() * rsqrt_cache.element_size() atIndex:3];
    [encoder setBuffer:gx_buf offset:grad_x.storage_offset() * grad_x.element_size() atIndex:4];
    uint32_t D_uint = D;
    [encoder setBytes:&D_uint length:sizeof(uint32_t) atIndex:5];

    MTLSize gx_gridSize = MTLSizeMake(num_rows, 1, 1);
    NSUInteger gx_threadGroupSize = 256; 
    MTLSize gx_threadsPerThreadgroup = MTLSizeMake(gx_threadGroupSize, 1, 1);
    [encoder dispatchThreadgroups:gx_gridSize threadsPerThreadgroup:gx_threadsPerThreadgroup];

    // 2. grad_w computation
    [encoder setComputePipelineState:rmsnorm_bwd_weight_pso];
    [encoder setBuffer:gy_buf offset:grad_y.storage_offset() * grad_y.element_size() atIndex:0];
    [encoder setBuffer:x_buf offset:x.storage_offset() * x.element_size() atIndex:1];
    [encoder setBuffer:rsqrt_buf offset:rsqrt_cache.storage_offset() * rsqrt_cache.element_size() atIndex:2];
    [encoder setBuffer:gw_buf offset:grad_w.storage_offset() * grad_w.element_size() atIndex:3];
    [encoder setBytes:&D_uint length:sizeof(uint32_t) atIndex:4];
    uint32_t num_rows_uint = num_rows;
    [encoder setBytes:&num_rows_uint length:sizeof(uint32_t) atIndex:5];

    NSUInteger gw_tg_size = rmsnorm_bwd_weight_pso.maxTotalThreadsPerThreadgroup;
    MTLSize gw_threads = MTLSizeMake(gw_tg_size, 1, 1);
    MTLSize gw_grid = MTLSizeMake((D + gw_tg_size - 1) / gw_tg_size, 1, 1);
    [encoder dispatchThreadgroups:gw_grid threadsPerThreadgroup:gw_threads];

    return std::make_tuple(grad_x, grad_w);
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_forward_mps(torch::Tensor x, torch::Tensor weight, float eps) {
    auto out = torch::empty_like(x);
    int64_t D = x.size(-1);
    int64_t num_rows = x.numel() / D;
    auto rsqrt_cache = torch::empty({num_rows}, x.options());

    id<MTLBuffer> x_buf = getMTLBufferStorage(x);
    id<MTLBuffer> w_buf = getMTLBufferStorage(weight);
    id<MTLBuffer> y_buf = getMTLBufferStorage(out);
    id<MTLBuffer> rsqrt_buf = getMTLBufferStorage(rsqrt_cache);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    [encoder setComputePipelineState:rmsnorm_fwd_pso];
    [encoder setBuffer:x_buf offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:w_buf offset:weight.storage_offset() * weight.element_size() atIndex:1];
    [encoder setBuffer:y_buf offset:out.storage_offset() * out.element_size() atIndex:2];
    [encoder setBuffer:rsqrt_buf offset:rsqrt_cache.storage_offset() * rsqrt_cache.element_size() atIndex:3];
    uint32_t D_uint = D;
    [encoder setBytes:&D_uint length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&eps length:sizeof(float) atIndex:5];

    MTLSize gridSize = MTLSizeMake(num_rows, 1, 1);
    // MUST be a power of 2 for the reduction tree in metal_kernels.metal to work!
    NSUInteger threadGroupSize = 256; 
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
    
    // PyTorch manages ending the encoder, so do not call [encoder endEncoding]

    return std::make_tuple(out, rsqrt_cache);
}

torch::Tensor swiglu_forward_mps(torch::Tensor gate, torch::Tensor up) {
    auto out = torch::empty_like(gate);
    int64_t num_vec4 = gate.numel() / 4;

    id<MTLBuffer> gate_buf = getMTLBufferStorage(gate);
    id<MTLBuffer> up_buf = getMTLBufferStorage(up);
    id<MTLBuffer> y_buf = getMTLBufferStorage(out);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    [encoder setComputePipelineState:swiglu_fwd_pso];
    [encoder setBuffer:gate_buf offset:gate.storage_offset() * gate.element_size() atIndex:0];
    [encoder setBuffer:up_buf offset:up.storage_offset() * up.element_size() atIndex:1];
    [encoder setBuffer:y_buf offset:out.storage_offset() * out.element_size() atIndex:2];
    uint32_t n = num_vec4;
    [encoder setBytes:&n length:sizeof(uint32_t) atIndex:3];

    NSUInteger threadGroupSize = swiglu_fwd_pso.maxTotalThreadsPerThreadgroup;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize gridSize = MTLSizeMake((num_vec4 + threadGroupSize - 1) / threadGroupSize, 1, 1);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];

    return out;
}

std::tuple<torch::Tensor, torch::Tensor> swiglu_backward_mps(torch::Tensor grad_y, torch::Tensor gate, torch::Tensor up) {
    auto grad_gate = torch::empty_like(gate);
    auto grad_up = torch::empty_like(up);
    int64_t num_vec4 = gate.numel() / 4;

    id<MTLBuffer> gy_buf = getMTLBufferStorage(grad_y);
    id<MTLBuffer> gate_buf = getMTLBufferStorage(gate);
    id<MTLBuffer> up_buf = getMTLBufferStorage(up);
    id<MTLBuffer> gg_buf = getMTLBufferStorage(grad_gate);
    id<MTLBuffer> gu_buf = getMTLBufferStorage(grad_up);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    [encoder setComputePipelineState:swiglu_bwd_pso];
    [encoder setBuffer:gy_buf offset:grad_y.storage_offset() * grad_y.element_size() atIndex:0];
    [encoder setBuffer:gate_buf offset:gate.storage_offset() * gate.element_size() atIndex:1];
    [encoder setBuffer:up_buf offset:up.storage_offset() * up.element_size() atIndex:2];
    [encoder setBuffer:gg_buf offset:grad_gate.storage_offset() * grad_gate.element_size() atIndex:3];
    [encoder setBuffer:gu_buf offset:grad_up.storage_offset() * grad_up.element_size() atIndex:4];
    uint32_t n = num_vec4;
    [encoder setBytes:&n length:sizeof(uint32_t) atIndex:5];

    NSUInteger threadGroupSize = swiglu_bwd_pso.maxTotalThreadsPerThreadgroup;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize gridSize = MTLSizeMake((num_vec4 + threadGroupSize - 1) / threadGroupSize, 1, 1);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];

    return std::make_tuple(grad_gate, grad_up);
}



torch::Tensor apply_rope_mps(torch::Tensor x, torch::Tensor freqs_cis, bool is_forward) {
    auto out = torch::empty_like(x);
    
    // x shape: (B, n_head, T, head_dim)
    // freqs_cis shape: (T, head_dim // 2)
    // We treat x as a flat array of float2 (size: numel / 2)
    
    uint32_t T = x.size(2);
    uint32_t half_head_dim = x.size(3) / 2;
    uint32_t num_elements = x.numel() / 2; // number of float2 vectors
    uint32_t forward = is_forward ? 1 : 0;
    
    id<MTLBuffer> x_buf = getMTLBufferStorage(x);
    id<MTLBuffer> freqs_buf = getMTLBufferStorage(freqs_cis);
    id<MTLBuffer> out_buf = getMTLBufferStorage(out);
    
    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    
    [encoder setComputePipelineState:rope_pso];
    [encoder setBuffer:x_buf offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:freqs_buf offset:freqs_cis.storage_offset() * freqs_cis.element_size() atIndex:1];
    [encoder setBuffer:out_buf offset:out.storage_offset() * out.element_size() atIndex:2];
    [encoder setBytes:&T length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&forward length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&num_elements length:sizeof(uint32_t) atIndex:6];
    
    NSUInteger threadGroupSize = rope_pso.maxTotalThreadsPerThreadgroup;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize gridSize = MTLSizeMake((num_elements + threadGroupSize - 1) / threadGroupSize, 1, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
    
    return out;
}


torch::Tensor cross_entropy_forward_mps(torch::Tensor logits, torch::Tensor targets, int ignore_index) {
    uint32_t N = logits.size(0);
    uint32_t V = logits.size(1);
    
    auto losses = torch::empty({N}, logits.options());
    
    id<MTLBuffer> logits_buf = getMTLBufferStorage(logits);
    id<MTLBuffer> targets_buf = getMTLBufferStorage(targets);
    id<MTLBuffer> losses_buf = getMTLBufferStorage(losses);
    
    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    
    [encoder setComputePipelineState:ce_fwd_pso];
    [encoder setBuffer:logits_buf offset:logits.storage_offset() * logits.element_size() atIndex:0];
    [encoder setBuffer:targets_buf offset:targets.storage_offset() * targets.element_size() atIndex:1];
    [encoder setBuffer:losses_buf offset:losses.storage_offset() * losses.element_size() atIndex:2];
    [encoder setBytes:&V length:sizeof(uint32_t) atIndex:3];
    int32_t ign = ignore_index;
    [encoder setBytes:&ign length:sizeof(int32_t) atIndex:4];
    
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger threadGroupSize = 256; 
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
    
    return losses;
}

torch::Tensor cross_entropy_backward_mps(torch::Tensor logits, torch::Tensor targets, torch::Tensor grad_losses, int ignore_index) {
    uint32_t N = logits.size(0);
    uint32_t V = logits.size(1);
    
    auto grad_logits = torch::empty_like(logits);
    
    id<MTLBuffer> logits_buf = getMTLBufferStorage(logits);
    id<MTLBuffer> targets_buf = getMTLBufferStorage(targets);
    id<MTLBuffer> grad_losses_buf = getMTLBufferStorage(grad_losses);
    id<MTLBuffer> grad_logits_buf = getMTLBufferStorage(grad_logits);
    
    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    
    [encoder setComputePipelineState:ce_bwd_pso];
    [encoder setBuffer:logits_buf offset:logits.storage_offset() * logits.element_size() atIndex:0];
    [encoder setBuffer:targets_buf offset:targets.storage_offset() * targets.element_size() atIndex:1];
    [encoder setBuffer:grad_losses_buf offset:grad_losses.storage_offset() * grad_losses.element_size() atIndex:2];
    [encoder setBuffer:grad_logits_buf offset:grad_logits.storage_offset() * grad_logits.element_size() atIndex:3];
    [encoder setBytes:&V length:sizeof(uint32_t) atIndex:4];
    int32_t ign = ignore_index;
    [encoder setBytes:&ign length:sizeof(int32_t) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger threadGroupSize = 256; 
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
    
    return grad_logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_metal", &init_mps, "Initialize Metal with path");
  m.def("rmsnorm_forward_mps", &rmsnorm_forward_mps, "MPS RMSNorm Forward");
  m.def("rmsnorm_backward_mps", &rmsnorm_backward_mps, "MPS RMSNorm Backward");
  m.def("swiglu_forward_mps", &swiglu_forward_mps, "MPS SwiGLU Forward");
  m.def("swiglu_backward_mps", &swiglu_backward_mps, "MPS SwiGLU Backward");
  m.def("apply_rope_mps", &apply_rope_mps, "MPS Apply RoPE");
  m.def("cross_entropy_forward_mps", &cross_entropy_forward_mps, "MPS CE Fwd");
  m.def("cross_entropy_backward_mps", &cross_entropy_backward_mps, "MPS CE Bwd");
}
