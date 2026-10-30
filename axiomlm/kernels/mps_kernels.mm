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
static id<MTLComputePipelineState> rmsnorm_fwd_half_pso = nil;
static id<MTLComputePipelineState> add_rmsnorm_fwd_pso = nil;
static id<MTLComputePipelineState> add_rmsnorm_fwd_half_pso = nil;
static id<MTLComputePipelineState> embed_rmsnorm_fwd_pso = nil;
static id<MTLComputePipelineState> embed_rmsnorm_fwd_half_pso = nil;
static id<MTLComputePipelineState> argmax_half_pso = nil;
static id<MTLComputePipelineState> rmsnorm_bwd_pso = nil;
static id<MTLComputePipelineState> rmsnorm_bwd_weight_pso = nil;
static id<MTLComputePipelineState> swiglu_fwd_pso = nil;
static id<MTLComputePipelineState> swiglu_fwd_half_pso = nil;
static id<MTLComputePipelineState> swiglu_gate_up_pso = nil;
static id<MTLComputePipelineState> swiglu_gate_up_half_pso = nil;
static id<MTLComputePipelineState> swiglu_bwd_pso = nil;
static id<MTLComputePipelineState> rope_pso = nil;
static id<MTLComputePipelineState> rope_half_pso = nil;
static id<MTLComputePipelineState> rope_kv_write_pso = nil;
static id<MTLComputePipelineState> rope_kv_write_half_pso = nil;
static id<MTLComputePipelineState> ce_fwd_pso = nil;
static id<MTLComputePipelineState> ce_bwd_pso = nil;
static id<MTLComputePipelineState> fa_fwd_pso = nil;
static id<MTLComputePipelineState> qmv_q4_g64_half_pso = nil;
static id<MTLComputePipelineState> qmm_q4_g64_half_pso = nil;
static id<MTLComputePipelineState> qmv_q8_g64_half_pso = nil;
static id<MTLComputePipelineState> qmm_q8_g64_half_pso = nil;
static id<MTLComputePipelineState> gqa_split_kv_pso = nil;
static id<MTLComputePipelineState> gqa_reduce_pso = nil;
static bool is_initialized = false;

void init_mps(std::string metal_path) {
    if (!is_initialized) {
        mtl_device = MTLCreateSystemDefaultDevice();
        NSString *ns_path = [NSString stringWithUTF8String:metal_path.c_str()];
        mtl_library = load_metal_library(mtl_device, ns_path);
        if (mtl_library) {
            rmsnorm_fwd_pso = get_pipeline(mtl_device, mtl_library, @"rmsnorm_forward_kernel");
            rmsnorm_fwd_half_pso = get_pipeline(mtl_device, mtl_library, @"rmsnorm_forward_half_kernel");
            add_rmsnorm_fwd_pso = get_pipeline(mtl_device, mtl_library, @"add_rmsnorm_forward_kernel");
            add_rmsnorm_fwd_half_pso = get_pipeline(mtl_device, mtl_library, @"add_rmsnorm_forward_half_kernel");
            embed_rmsnorm_fwd_pso = get_pipeline(mtl_device, mtl_library, @"embed_rmsnorm_forward_kernel");
            embed_rmsnorm_fwd_half_pso = get_pipeline(mtl_device, mtl_library, @"embed_rmsnorm_forward_half_kernel");
            argmax_half_pso = get_pipeline(mtl_device, mtl_library, @"argmax_half_kernel");
            rmsnorm_bwd_pso = get_pipeline(mtl_device, mtl_library, @"rmsnorm_backward_kernel");
            rmsnorm_bwd_weight_pso = get_pipeline(mtl_device, mtl_library, @"rmsnorm_backward_weight_kernel");
            swiglu_fwd_pso = get_pipeline(mtl_device, mtl_library, @"swiglu_forward_kernel");
            swiglu_fwd_half_pso = get_pipeline(mtl_device, mtl_library, @"swiglu_forward_half_kernel");
            swiglu_gate_up_pso = get_pipeline(mtl_device, mtl_library, @"swiglu_gate_up_kernel");
            swiglu_gate_up_half_pso = get_pipeline(mtl_device, mtl_library, @"swiglu_gate_up_half_kernel");
            swiglu_bwd_pso = get_pipeline(mtl_device, mtl_library, @"swiglu_backward_kernel");
            rope_pso = get_pipeline(mtl_device, mtl_library, @"rope_kernel");
            rope_half_pso = get_pipeline(mtl_device, mtl_library, @"rope_half_kernel");
            rope_kv_write_pso = get_pipeline(mtl_device, mtl_library, @"rope_kv_cache_write_kernel");
            rope_kv_write_half_pso = get_pipeline(mtl_device, mtl_library, @"rope_kv_cache_write_half_kernel");
            ce_fwd_pso = get_pipeline(mtl_device, mtl_library, @"cross_entropy_forward_kernel");
            ce_bwd_pso = get_pipeline(mtl_device, mtl_library, @"cross_entropy_backward_kernel");
            fa_fwd_pso = get_pipeline(mtl_device, mtl_library, @"flash_attention_forward_kernel");
            qmv_q4_g64_half_pso = get_pipeline(mtl_device, mtl_library, @"qmv_q4_g64_half_kernel");
            qmm_q4_g64_half_pso = get_pipeline(mtl_device, mtl_library, @"qmm_q4_g64_half_kernel");
            qmv_q8_g64_half_pso = get_pipeline(mtl_device, mtl_library, @"qmv_q8_g64_half_kernel");
            qmm_q8_g64_half_pso = get_pipeline(mtl_device, mtl_library, @"qmm_q8_g64_half_kernel");
            gqa_split_kv_pso = get_pipeline(mtl_device, mtl_library, @"gqa_flash_decode_split_kv_half_kernel");
            gqa_reduce_pso = get_pipeline(mtl_device, mtl_library, @"gqa_flash_decode_reduce_half_kernel");
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
    auto rsqrt_cache = torch::empty({num_rows}, x.options().dtype(torch::kFloat32));

    id<MTLBuffer> x_buf = getMTLBufferStorage(x);
    id<MTLBuffer> w_buf = getMTLBufferStorage(weight);
    id<MTLBuffer> y_buf = getMTLBufferStorage(out);
    id<MTLBuffer> rsqrt_buf = getMTLBufferStorage(rsqrt_cache);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    id<MTLComputePipelineState> pso = (x.scalar_type() == at::kHalf) ? rmsnorm_fwd_half_pso : rmsnorm_fwd_pso;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:x_buf offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:w_buf offset:weight.storage_offset() * weight.element_size() atIndex:1];
    [encoder setBuffer:y_buf offset:out.storage_offset() * out.element_size() atIndex:2];
    [encoder setBuffer:rsqrt_buf offset:rsqrt_cache.storage_offset() * rsqrt_cache.element_size() atIndex:3];
    uint32_t D_uint = D;
    [encoder setBytes:&D_uint length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&eps length:sizeof(float) atIndex:5];

    MTLSize gridSize = MTLSizeMake(num_rows, 1, 1);
    NSUInteger threadGroupSize = 256;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];

    return std::make_tuple(out, rsqrt_cache);
}

static inline void dispatch_rmsnorm_infer_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    torch::Tensor& out,
    float eps) {
    int64_t D = x.size(-1);
    int64_t num_rows = x.numel() / D;
    id<MTLComputePipelineState> pso = (x.scalar_type() == at::kHalf) ? rmsnorm_fwd_half_pso : rmsnorm_fwd_pso;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(weight) offset:weight.storage_offset() * weight.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:2];
    [encoder setBuffer:nil offset:0 atIndex:3];
    uint32_t D_uint = D;
    [encoder setBytes:&D_uint length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&eps length:sizeof(float) atIndex:5];
    NSUInteger threadGroupSize = (num_rows <= 4) ? 64 : 256;
    [encoder dispatchThreadgroups:MTLSizeMake(num_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

static inline void dispatch_embed_rmsnorm_infer_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& idx,
    const torch::Tensor& wte_weight,
    const torch::Tensor& ln1_weight,
    torch::Tensor& x_out,
    torch::Tensor& norm_out,
    float eps) {
    int64_t D = wte_weight.size(-1);
    id<MTLComputePipelineState> pso = (wte_weight.scalar_type() == at::kHalf)
        ? embed_rmsnorm_fwd_half_pso
        : embed_rmsnorm_fwd_pso;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:getMTLBufferStorage(idx) offset:idx.storage_offset() * idx.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(wte_weight) offset:wte_weight.storage_offset() * wte_weight.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(ln1_weight) offset:ln1_weight.storage_offset() * ln1_weight.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(x_out) offset:x_out.storage_offset() * x_out.element_size() atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(norm_out) offset:norm_out.storage_offset() * norm_out.element_size() atIndex:4];
    uint32_t D_uint = D;
    [encoder setBytes:&D_uint length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&eps length:sizeof(float) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

torch::Tensor argmax_last_token_mps(torch::Tensor logits) {
    auto out = torch::empty({1, 1}, logits.options().dtype(torch::kLong));
    uint32_t V = logits.size(-1);
    NSUInteger row_offset = (logits.storage_offset() + (logits.numel() - V)) * logits.element_size();
    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:argmax_half_pso];
    [encoder setBuffer:getMTLBufferStorage(logits) offset:row_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(out) offset:0 atIndex:1];
    [encoder setBytes:&V length:sizeof(uint32_t) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return out;
}

torch::Tensor rmsnorm_infer_mps(torch::Tensor x, torch::Tensor weight, float eps) {
    auto out = torch::empty_like(x);
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_rmsnorm_infer_into(stream->commandEncoder(), x, weight, out, eps);
    return out;
}

static inline void dispatch_add_rmsnorm_infer_into(
    id<MTLComputeCommandEncoder> encoder,
    torch::Tensor& x_inout,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    torch::Tensor& out,
    float eps) {
    int64_t D = x_inout.size(-1);
    int64_t num_rows = x_inout.numel() / D;
    id<MTLComputePipelineState> pso = (x_inout.scalar_type() == at::kHalf) ? add_rmsnorm_fwd_half_pso : add_rmsnorm_fwd_pso;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:getMTLBufferStorage(x_inout) offset:x_inout.storage_offset() * x_inout.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(residual) offset:residual.storage_offset() * residual.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(weight) offset:weight.storage_offset() * weight.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
    uint32_t D_uint = D;
    [encoder setBytes:&D_uint length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&eps length:sizeof(float) atIndex:5];
    NSUInteger threadGroupSize = (num_rows <= 4) ? 64 : 256;
    [encoder dispatchThreadgroups:MTLSizeMake(num_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

torch::Tensor add_rmsnorm_infer_mps(torch::Tensor x_inout, torch::Tensor residual, torch::Tensor weight, float eps) {
    auto out = torch::empty_like(x_inout);
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_add_rmsnorm_infer_into(stream->commandEncoder(), x_inout, residual, weight, out, eps);
    return out;
}

static inline void dispatch_swiglu_gate_up_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& gate_up,
    torch::Tensor& out) {
    int64_t D = out.size(-1);
    uint32_t D_vec4 = D / 4;
    uint32_t total_vec4 = out.numel() / 4;
    id<MTLComputePipelineState> pso = (gate_up.scalar_type() == at::kHalf) ? swiglu_gate_up_half_pso : swiglu_gate_up_pso;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:getMTLBufferStorage(gate_up) offset:gate_up.storage_offset() * gate_up.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:1];
    [encoder setBytes:&D_vec4 length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&total_vec4 length:sizeof(uint32_t) atIndex:3];
    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_vec4 + threadGroupSize - 1) / threadGroupSize, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

torch::Tensor swiglu_gate_up_mps(torch::Tensor gate_up) {
    auto sizes = gate_up.sizes().vec();
    int64_t D = sizes.back() / 2;
    sizes.back() = D;
    auto out = torch::empty(sizes, gate_up.options());
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_swiglu_gate_up_into(stream->commandEncoder(), gate_up, out);
    return out;
}

torch::Tensor swiglu_forward_mps(torch::Tensor gate, torch::Tensor up) {
    auto out = torch::empty_like(gate);
    int64_t num_vec4 = gate.numel() / 4;

    id<MTLBuffer> gate_buf = getMTLBufferStorage(gate);
    id<MTLBuffer> up_buf = getMTLBufferStorage(up);
    id<MTLBuffer> y_buf = getMTLBufferStorage(out);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    id<MTLComputePipelineState> pso = (gate.scalar_type() == at::kHalf) ? swiglu_fwd_half_pso : swiglu_fwd_pso;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gate_buf offset:gate.storage_offset() * gate.element_size() atIndex:0];
    [encoder setBuffer:up_buf offset:up.storage_offset() * up.element_size() atIndex:1];
    [encoder setBuffer:y_buf offset:out.storage_offset() * out.element_size() atIndex:2];
    uint32_t n = num_vec4;
    [encoder setBytes:&n length:sizeof(uint32_t) atIndex:3];

    NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
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

    uint32_t T = x.size(2);
    uint32_t half_head_dim = x.size(3) / 2;
    uint32_t num_elements = x.numel() / 2;
    uint32_t forward = is_forward ? 1 : 0;

    id<MTLBuffer> x_buf = getMTLBufferStorage(x);
    id<MTLBuffer> freqs_buf = getMTLBufferStorage(freqs_cis);
    id<MTLBuffer> out_buf = getMTLBufferStorage(out);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    id<MTLComputePipelineState> pso = (x.scalar_type() == at::kHalf) ? rope_half_pso : rope_pso;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:x_buf offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:freqs_buf offset:freqs_cis.storage_offset() * freqs_cis.element_size() atIndex:1];
    [encoder setBuffer:out_buf offset:out.storage_offset() * out.element_size() atIndex:2];
    [encoder setBytes:&T length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&forward length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&num_elements length:sizeof(uint32_t) atIndex:6];

    NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize gridSize = MTLSizeMake((num_elements + threadGroupSize - 1) / threadGroupSize, 1, 1);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];

    return out;
}

torch::Tensor apply_rope_inplace_mps(torch::Tensor x, torch::Tensor freqs_cis, int64_t start_pos) {
    uint32_t T = x.size(2);
    uint32_t half_head_dim = x.size(3) / 2;
    uint32_t num_elements = x.numel() / 2;
    uint32_t forward = 1;

    id<MTLBuffer> x_buf = getMTLBufferStorage(x);
    id<MTLBuffer> freqs_buf = getMTLBufferStorage(freqs_cis);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    id<MTLComputePipelineState> pso = (x.scalar_type() == at::kHalf) ? rope_half_pso : rope_pso;
    [encoder setComputePipelineState:pso];
    NSUInteger x_offset = x.storage_offset() * x.element_size();
    NSUInteger freqs_offset = (freqs_cis.storage_offset() + start_pos * half_head_dim * 2) * freqs_cis.element_size();
    [encoder setBuffer:x_buf offset:x_offset atIndex:0];
    [encoder setBuffer:freqs_buf offset:freqs_offset atIndex:1];
    [encoder setBuffer:x_buf offset:x_offset atIndex:2];
    [encoder setBytes:&T length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&forward length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&num_elements length:sizeof(uint32_t) atIndex:6];

    NSUInteger threadGroupSize = 256;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize gridSize = MTLSizeMake((num_elements + threadGroupSize - 1) / threadGroupSize, 1, 1);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];

    return x;
}

static inline void dispatch_rope_kv_cache_write(
    id<MTLComputeCommandEncoder> encoder,
    torch::Tensor& qkv,
    const torch::Tensor& freqs_cis,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    uint32_t q_pairs,
    uint32_t kv_pairs,
    uint32_t half_head_dim,
    uint32_t max_seq_len,
    int64_t start_pos,
    uint32_t write_pos) {
    uint32_t total_pairs = q_pairs + (kv_pairs << 1);
    id<MTLComputePipelineState> pso = (qkv.scalar_type() == at::kHalf) ? rope_kv_write_half_pso : rope_kv_write_pso;
    [encoder setComputePipelineState:pso];

    NSUInteger qkv_offset = qkv.storage_offset() * qkv.element_size();
    NSUInteger freqs_offset = (freqs_cis.storage_offset() + start_pos * half_head_dim * 2) * freqs_cis.element_size();
    NSUInteger k_offset = k_cache.storage_offset() * k_cache.element_size();
    NSUInteger v_offset = v_cache.storage_offset() * v_cache.element_size();

    [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(freqs_cis) offset:freqs_offset atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(k_cache) offset:k_offset atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(v_cache) offset:v_offset atIndex:3];
    [encoder setBytes:&q_pairs length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&kv_pairs length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&write_pos length:sizeof(uint32_t) atIndex:8];

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_pairs + threadGroupSize - 1) / threadGroupSize, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

void rope_kv_cache_write_mps(
    torch::Tensor qkv,
    torch::Tensor freqs_cis,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t q_dim,
    int64_t kv_dim,
    int64_t head_dim,
    int64_t start_pos,
    int64_t write_pos) {
    uint32_t q_pairs = q_dim / 2;
    uint32_t kv_pairs = kv_dim / 2;
    uint32_t half_head_dim = head_dim / 2;
    uint32_t max_seq_len = k_cache.size(2);
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_rope_kv_cache_write(
        stream->commandEncoder(),
        qkv,
        freqs_cis,
        k_cache,
        v_cache,
        q_pairs,
        kv_pairs,
        half_head_dim,
        max_seq_len,
        start_pos,
        (uint32_t)write_pos);
}

torch::Tensor transformer_decode_step_mps(
    torch::Tensor idx,
    torch::Tensor wte_weight,
    torch::Tensor freqs_real,
    std::vector<torch::Tensor> ln1_weights,
    std::vector<torch::Tensor> c_attn_weights,
    std::vector<torch::Tensor> c_proj_weights,
    std::vector<torch::Tensor> ln2_weights,
    std::vector<torch::Tensor> gate_up_weights,
    std::vector<torch::Tensor> w_down_weights,
    torch::Tensor lnf_weight,
    std::vector<torch::Tensor> k_caches,
    std::vector<torch::Tensor> v_caches,
    torch::Tensor norm_buf,
    torch::Tensor qkv_buf,
    torch::Tensor attn_out_buf,
    torch::Tensor gate_up_buf,
    torch::Tensor swiglu_buf,
    torch::Tensor mlp_out_buf,
    torch::Tensor logits_buf,
    int64_t n_head,
    int64_t n_kv_head,
    int64_t head_dim,
    int64_t start_pos,
    int64_t write_pos,
    int64_t active_len,
    float eps) {
    auto x = at::embedding(wte_weight, idx);
    int64_t n_layer = ln1_weights.size();
    int64_t q_dim = n_head * head_dim;
    int64_t kv_dim = n_kv_head * head_dim;
    uint32_t q_pairs = q_dim / 2;
    uint32_t kv_pairs = kv_dim / 2;
    uint32_t half_head_dim = head_dim / 2;
    uint32_t max_seq_len = k_caches[0].size(2);
    bool enable_gqa = (n_head != n_kv_head);

    auto q_view = qkv_buf.slice(2, 0, q_dim).view({1, n_head, 1, head_dim});
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_rmsnorm_infer_into(stream->commandEncoder(), x, ln1_weights[0], norm_buf, eps);

    for (int64_t i = 0; i < n_layer; ++i) {
        at::linear_out(qkv_buf, norm_buf, c_attn_weights[i]);
        dispatch_rope_kv_cache_write(
            stream->commandEncoder(),
            qkv_buf,
            freqs_real,
            k_caches[i],
            v_caches[i],
            q_pairs,
            kv_pairs,
            half_head_dim,
            max_seq_len,
            start_pos,
            (uint32_t)write_pos);

        auto k_active = (active_len == max_seq_len)
            ? k_caches[i]
            : k_caches[i].slice(2, 0, active_len);
        auto v_active = (active_len == max_seq_len)
            ? v_caches[i]
            : v_caches[i].slice(2, 0, active_len);

        auto attn_y = at::scaled_dot_product_attention(
            q_view, k_active, v_active, {}, 0.0, false, c10::nullopt, enable_gqa);
        at::linear_out(attn_out_buf, attn_y.view({1, 1, q_dim}), c_proj_weights[i]);

        dispatch_add_rmsnorm_infer_into(
            stream->commandEncoder(), x, attn_out_buf, ln2_weights[i], norm_buf, eps);

        at::linear_out(gate_up_buf, norm_buf, gate_up_weights[i]);
        dispatch_swiglu_gate_up_into(stream->commandEncoder(), gate_up_buf, swiglu_buf);
        at::linear_out(mlp_out_buf, swiglu_buf, w_down_weights[i]);

        const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
        dispatch_add_rmsnorm_infer_into(
            stream->commandEncoder(), x, mlp_out_buf, next_ln_w, norm_buf, eps);
    }

    at::linear_out(logits_buf, norm_buf, wte_weight);
    return logits_buf;
}

static inline void dispatch_qmv_q4_g64_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales_biases,
    torch::Tensor& y) {
    uint32_t K = x.size(-1);
    uint32_t N = w_packed.size(0);

    [encoder setComputePipelineState:qmv_q4_g64_half_pso];
    [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() * w_packed.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales_biases) offset:scales_biases.storage_offset() * scales_biases.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:5];

    uint32_t num_tg = (N + 7) / 8;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static inline void dispatch_qmm_q4_g64_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales_biases,
    torch::Tensor& y) {
    uint32_t K = x.size(-1);
    uint32_t M = x.numel() / K;
    uint32_t N = w_packed.size(0);

    if (M == 1) {
        dispatch_qmv_q4_g64_into(encoder, x, w_packed, scales_biases, y);
        return;
    }

    [encoder setComputePipelineState:qmm_q4_g64_half_pso];
    [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() * w_packed.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales_biases) offset:scales_biases.storage_offset() * scales_biases.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
    [encoder setBytes:&M length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:6];

    uint32_t num_tg = (N + 3) / 4;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static inline void dispatch_gqa_flash_decode_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    torch::Tensor& out,
    torch::Tensor& partial_O,
    torch::Tensor& partial_m,
    torch::Tensor& partial_l,
    uint32_t max_seq_len,
    uint32_t active_len,
    uint32_t num_splits) {
    float scale = 0.125f; // 1.0f / sqrt(64)
    uint32_t n_kv_head = 4;

    [encoder setComputePipelineState:gqa_split_kv_pso];
    [encoder setBuffer:getMTLBufferStorage(q) offset:q.storage_offset() * q.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(k_cache) offset:k_cache.storage_offset() * k_cache.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(v_cache) offset:v_cache.storage_offset() * v_cache.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:6];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&active_len length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&scale length:sizeof(float) atIndex:10];

    MTLSize gridSize = MTLSizeMake(n_kv_head, num_splits, 1);
    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1) {
        [encoder setComputePipelineState:gqa_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];

        MTLSize reduceGrid = MTLSizeMake(12, 1, 1);
        [encoder dispatchThreadgroups:reduceGrid threadsPerThreadgroup:tgSize];
    }
}

torch::Tensor gqa_flash_decode_mps(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t active_len,
    int64_t num_splits) {
    auto out = torch::empty_like(q);
    uint32_t max_seq_len = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(2) : (uint32_t)k_cache.size(1);
    uint32_t n_splits = (num_splits > 0) ? (uint32_t)num_splits : ((active_len <= 64) ? 1 : ((active_len <= 256) ? 4 : 8));
    auto opts = q.options();
    auto partial_O = torch::empty({12, n_splits, 32, 2}, opts);
    auto partial_m = torch::empty({12, n_splits}, opts.dtype(torch::kFloat32));
    auto partial_l = torch::empty({12, n_splits}, opts.dtype(torch::kFloat32));

    auto stream = at::mps::getCurrentMPSStream();
    dispatch_gqa_flash_decode_into(
        stream->commandEncoder(),
        q,
        k_cache,
        v_cache,
        out,
        partial_O,
        partial_m,
        partial_l,
        max_seq_len,
        (uint32_t)active_len,
        n_splits);
    return out;
}

static inline void dispatch_qmv_q8_g64_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales_biases,
    torch::Tensor& y) {
    uint32_t K = x.size(-1);
    uint32_t N = w_packed.size(0);

    [encoder setComputePipelineState:qmv_q8_g64_half_pso];
    [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() * w_packed.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales_biases) offset:scales_biases.storage_offset() * scales_biases.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:5];

    uint32_t num_tg = (N + 7) / 8;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static inline void dispatch_qmm_q8_g64_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales_biases,
    torch::Tensor& y) {
    uint32_t K = x.size(-1);
    uint32_t M = x.numel() / K;
    uint32_t N = w_packed.size(0);

    if (M == 1) {
        dispatch_qmv_q8_g64_into(encoder, x, w_packed, scales_biases, y);
        return;
    }

    [encoder setComputePipelineState:qmm_q8_g64_half_pso];
    [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() * w_packed.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales_biases) offset:scales_biases.storage_offset() * scales_biases.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
    [encoder setBytes:&M length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:6];

    uint32_t num_tg = (N + 3) / 4;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

struct DecodeGraphState {
    torch::Tensor wte_weight;
    torch::Tensor freqs_real;
    std::vector<torch::Tensor> ln1_weights;
    std::vector<torch::Tensor> c_attn_weights;
    std::vector<torch::Tensor> c_proj_weights;
    std::vector<torch::Tensor> ln2_weights;
    std::vector<torch::Tensor> gate_up_weights;
    std::vector<torch::Tensor> w_down_weights;
    torch::Tensor lnf_weight;
    std::vector<torch::Tensor> k_caches;
    std::vector<torch::Tensor> v_caches;
    torch::Tensor x_buf;
    torch::Tensor x_buf_2d;
    torch::Tensor norm_buf;
    torch::Tensor qkv_buf;
    torch::Tensor q_view;
    torch::Tensor attn_out_buf;
    torch::Tensor attn_y_buf;
    torch::Tensor partial_O_buf;
    torch::Tensor partial_m_buf;
    torch::Tensor partial_l_buf;
    torch::Tensor gate_up_buf;
    torch::Tensor swiglu_buf;
    torch::Tensor mlp_out_buf;
    torch::Tensor logits_buf;
    torch::Tensor x_buf_multi;
    torch::Tensor norm_buf_multi;
    torch::Tensor qkv_buf_multi;
    torch::Tensor attn_y_buf_multi;
    torch::Tensor attn_out_buf_multi;
    torch::Tensor gate_up_buf_multi;
    torch::Tensor swiglu_buf_multi;
    torch::Tensor mlp_out_buf_multi;
    torch::Tensor logits_buf_multi;
    int64_t n_layer = 0;
    int64_t n_head = 0;
    int64_t n_kv_head = 0;
    int64_t head_dim = 0;
    int64_t q_dim = 0;
    int64_t kv_dim = 0;
    uint32_t q_pairs = 0;
    uint32_t kv_pairs = 0;
    uint32_t half_head_dim = 0;
    uint32_t max_seq_len = 0;
    bool enable_gqa = false;
    float eps = 1e-6f;
    bool initialized = false;

    DecodeGraphState(
        torch::Tensor wte_w,
        torch::Tensor freqs_r,
        std::vector<torch::Tensor> ln1_ws,
        std::vector<torch::Tensor> c_attn_ws,
        std::vector<torch::Tensor> c_proj_ws,
        std::vector<torch::Tensor> ln2_ws,
        std::vector<torch::Tensor> gate_up_ws,
        std::vector<torch::Tensor> w_down_ws,
        torch::Tensor lnf_w,
        std::vector<torch::Tensor> k_cs,
        std::vector<torch::Tensor> v_cs,
        int64_t n_h,
        int64_t n_kvh,
        int64_t hd,
        float epsilon)
        : wte_weight(wte_w.detach()),
          freqs_real(freqs_r),
          ln1_weights(std::move(ln1_ws)),
          c_attn_weights(std::move(c_attn_ws)),
          c_proj_weights(std::move(c_proj_ws)),
          ln2_weights(std::move(ln2_ws)),
          gate_up_weights(std::move(gate_up_ws)),
          w_down_weights(std::move(w_down_ws)),
          lnf_weight(lnf_w),
          k_caches(std::move(k_cs)),
          v_caches(std::move(v_cs)),
          n_layer((int64_t)this->ln1_weights.size()),
          n_head(n_h),
          n_kv_head(n_kvh),
          head_dim(hd),
          q_dim(n_h * hd),
          kv_dim(n_kvh * hd),
          q_pairs((uint32_t)((n_h * hd) / 2)),
          kv_pairs((uint32_t)((n_kvh * hd) / 2)),
          half_head_dim((uint32_t)(hd / 2)),
          max_seq_len((uint32_t)this->k_caches[0].size(2)),
          enable_gqa(n_h != n_kvh),
          eps(epsilon),
          initialized(true) {
        auto opts = wte_weight.options();
        int64_t d_model = q_dim;
        int64_t qkv_dim = q_dim + 2 * kv_dim;
        int64_t gu_dim = gate_up_weights[0].size(0);
        int64_t hidden_dim = gu_dim / 2;
        int64_t vocab_size = wte_weight.size(0);

        x_buf = torch::empty({1, 1, d_model}, opts);
        x_buf_2d = x_buf.view({1, d_model});
        norm_buf = torch::empty({1, 1, d_model}, opts);
        qkv_buf = torch::empty({1, 1, qkv_dim}, opts);
        q_view = qkv_buf.slice(2, 0, d_model).view({1, n_head, 1, head_dim});
        attn_out_buf = torch::empty({1, 1, d_model}, opts);
        attn_y_buf = torch::empty({1, 1, d_model}, opts);
        partial_O_buf = torch::empty({12, 16, 32, 2}, opts);
        partial_m_buf = torch::empty({12, 16}, opts.dtype(torch::kFloat32));
        partial_l_buf = torch::empty({12, 16}, opts.dtype(torch::kFloat32));
        gate_up_buf = torch::empty({1, 1, gu_dim}, opts);
        swiglu_buf = torch::empty({1, 1, hidden_dim}, opts);
        mlp_out_buf = torch::empty({1, 1, d_model}, opts);
        logits_buf = torch::empty({1, 1, vocab_size}, opts);
        x_buf_multi = torch::empty({1, 8, d_model}, opts);
        norm_buf_multi = torch::empty({1, 8, d_model}, opts);
        qkv_buf_multi = torch::empty({1, 8, qkv_dim}, opts);
        attn_y_buf_multi = torch::empty({1, 8, d_model}, opts);
        attn_out_buf_multi = torch::empty({1, 8, d_model}, opts);
        gate_up_buf_multi = torch::empty({1, 8, gu_dim}, opts);
        swiglu_buf_multi = torch::empty({1, 8, hidden_dim}, opts);
        mlp_out_buf_multi = torch::empty({1, 8, d_model}, opts);
        logits_buf_multi = torch::empty({1, 8, vocab_size}, opts);
    }

    torch::Tensor step(
        torch::Tensor idx,
        int64_t start_pos,
        int64_t write_pos,
        int64_t active_len) {
        at::NoGradGuard no_grad;
        at::index_select_out(x_buf_2d, wte_weight, 0, idx.view({-1}));
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        dispatch_rmsnorm_infer_into(encoder, x_buf, ln1_weights[0], norm_buf, eps);

        for (int64_t i = 0; i < n_layer; ++i) {
            at::linear_out(qkv_buf, norm_buf, c_attn_weights[i]);
            dispatch_rope_kv_cache_write(
                stream->commandEncoder(),
                qkv_buf,
                freqs_real,
                k_caches[i],
                v_caches[i],
                q_pairs,
                kv_pairs,
                half_head_dim,
                max_seq_len,
                start_pos,
                (uint32_t)write_pos);

            uint32_t n_splits = (active_len <= 64) ? 1 : ((active_len <= 256) ? 4 : 8);
            dispatch_gqa_flash_decode_into(
                stream->commandEncoder(),
                qkv_buf,
                k_caches[i],
                v_caches[i],
                attn_y_buf,
                partial_O_buf,
                partial_m_buf,
                partial_l_buf,
                max_seq_len,
                (uint32_t)active_len,
                n_splits);
            at::linear_out(attn_out_buf, attn_y_buf, c_proj_weights[i]);

            dispatch_add_rmsnorm_infer_into(
                stream->commandEncoder(), x_buf, attn_out_buf, ln2_weights[i], norm_buf, eps);

            at::linear_out(gate_up_buf, norm_buf, gate_up_weights[i]);
            dispatch_swiglu_gate_up_into(stream->commandEncoder(), gate_up_buf, swiglu_buf);
            at::linear_out(mlp_out_buf, swiglu_buf, w_down_weights[i]);

            const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
            dispatch_add_rmsnorm_infer_into(
                stream->commandEncoder(), x_buf, mlp_out_buf, next_ln_w, norm_buf, eps);
        }

        at::linear_out(logits_buf, norm_buf, wte_weight);
        return logits_buf;
    }

    torch::Tensor verify_step(
        torch::Tensor idx,
        int64_t start_pos,
        int64_t write_pos,
        int64_t active_len) {
        at::NoGradGuard no_grad;
        int64_t T = idx.numel();
        if (T == 1) {
            return step(idx, start_pos, write_pos, active_len);
        }
        TORCH_CHECK(T <= 8, "verify_step supports up to 8 tokens, got ", T);

        auto x_cur = x_buf_multi.narrow(1, 0, T);
        auto norm_cur = norm_buf_multi.narrow(1, 0, T);
        auto qkv_cur = qkv_buf_multi.narrow(1, 0, T);
        auto attn_y_cur = attn_y_buf_multi.narrow(1, 0, T);
        auto attn_out_cur = attn_out_buf_multi.narrow(1, 0, T);
        auto gate_up_cur = gate_up_buf_multi.narrow(1, 0, T);
        auto swiglu_cur = swiglu_buf_multi.narrow(1, 0, T);
        auto mlp_out_cur = mlp_out_buf_multi.narrow(1, 0, T);
        auto logits_cur = logits_buf_multi.narrow(1, 0, T);

        auto x_cur_2d = x_cur.view({T, q_dim});
        at::index_select_out(x_cur_2d, wte_weight, 0, idx.view({-1}));

        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        dispatch_rmsnorm_infer_into(encoder, x_cur, ln1_weights[0], norm_cur, eps);

        for (int64_t i = 0; i < n_layer; ++i) {
            at::linear_out(qkv_cur, norm_cur, c_attn_weights[i]);

            encoder = stream->commandEncoder();
            for (int64_t t = 0; t < T; ++t) {
                auto qkv_t = qkv_cur.narrow(1, t, 1);
                dispatch_rope_kv_cache_write(
                    encoder,
                    qkv_t,
                    freqs_real,
                    k_caches[i],
                    v_caches[i],
                    q_pairs,
                    kv_pairs,
                    half_head_dim,
                    max_seq_len,
                    start_pos + t,
                    (uint32_t)(write_pos + t));
            }

            for (int64_t t = 0; t < T; ++t) {
                auto qkv_t = qkv_cur.narrow(1, t, 1);
                auto attn_y_t = attn_y_cur.narrow(1, t, 1);
                uint32_t cur_active = (uint32_t)(start_pos + t + 1);
                if (cur_active > max_seq_len) cur_active = max_seq_len;
                uint32_t n_splits = (cur_active <= 64) ? 1 : ((cur_active <= 256) ? 4 : 8);

                dispatch_gqa_flash_decode_into(
                    encoder,
                    qkv_t,
                    k_caches[i],
                    v_caches[i],
                    attn_y_t,
                    partial_O_buf,
                    partial_m_buf,
                    partial_l_buf,
                    max_seq_len,
                    cur_active,
                    n_splits);
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            at::linear_out(attn_out_cur, attn_y_cur, c_proj_weights[i]);

            encoder = stream->commandEncoder();
            dispatch_add_rmsnorm_infer_into(
                encoder, x_cur, attn_out_cur, ln2_weights[i], norm_cur, eps);

            at::linear_out(gate_up_cur, norm_cur, gate_up_weights[i]);

            encoder = stream->commandEncoder();
            dispatch_swiglu_gate_up_into(encoder, gate_up_cur, swiglu_cur);

            at::linear_out(mlp_out_cur, swiglu_cur, w_down_weights[i]);

            const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
            encoder = stream->commandEncoder();
            dispatch_add_rmsnorm_infer_into(
                encoder, x_cur, mlp_out_cur, next_ln_w, norm_cur, eps);
        }

        at::linear_out(logits_cur, norm_cur, wte_weight);
        return logits_cur;
    }
};

struct QuantizedDecodeGraphState {
    torch::Tensor wte_embed;
    torch::Tensor lm_head_packed;
    torch::Tensor lm_head_sb;
    torch::Tensor freqs_real;
    std::vector<torch::Tensor> ln1_weights;
    std::vector<torch::Tensor> c_attn_packed;
    std::vector<torch::Tensor> c_attn_sb;
    std::vector<torch::Tensor> c_proj_packed;
    std::vector<torch::Tensor> c_proj_sb;
    std::vector<torch::Tensor> ln2_weights;
    std::vector<torch::Tensor> gate_up_packed;
    std::vector<torch::Tensor> gate_up_sb;
    std::vector<torch::Tensor> w_down_packed;
    std::vector<torch::Tensor> w_down_sb;
    torch::Tensor lnf_weight;
    std::vector<torch::Tensor> k_caches;
    std::vector<torch::Tensor> v_caches;
    torch::Tensor x_buf;
    torch::Tensor x_buf_2d;
    torch::Tensor norm_buf;
    torch::Tensor qkv_buf;
    torch::Tensor q_view;
    torch::Tensor attn_out_buf;
    torch::Tensor attn_y_buf;
    torch::Tensor partial_O_buf;
    torch::Tensor partial_m_buf;
    torch::Tensor partial_l_buf;
    torch::Tensor gate_up_buf;
    torch::Tensor swiglu_buf;
    torch::Tensor mlp_out_buf;
    torch::Tensor logits_buf;
    torch::Tensor x_buf_multi;
    torch::Tensor norm_buf_multi;
    torch::Tensor qkv_buf_multi;
    torch::Tensor attn_y_buf_multi;
    torch::Tensor attn_out_buf_multi;
    torch::Tensor gate_up_buf_multi;
    torch::Tensor swiglu_buf_multi;
    torch::Tensor mlp_out_buf_multi;
    torch::Tensor logits_buf_multi;
    int64_t n_layer = 0;
    int64_t n_head = 0;
    int64_t n_kv_head = 0;
    int64_t head_dim = 0;
    int64_t q_dim = 0;
    int64_t kv_dim = 0;
    uint32_t q_pairs = 0;
    uint32_t kv_pairs = 0;
    uint32_t half_head_dim = 0;
    uint32_t max_seq_len = 0;
    bool enable_gqa = false;
    bool is_q4 = true;
    float eps = 1e-6f;
    bool initialized = false;

    QuantizedDecodeGraphState(
        torch::Tensor wte_emb,
        torch::Tensor lmh_packed,
        torch::Tensor lmh_sb,
        torch::Tensor freqs_r,
        std::vector<torch::Tensor> ln1_ws,
        std::vector<torch::Tensor> c_attn_p,
        std::vector<torch::Tensor> c_attn_s,
        std::vector<torch::Tensor> c_proj_p,
        std::vector<torch::Tensor> c_proj_s,
        std::vector<torch::Tensor> ln2_ws,
        std::vector<torch::Tensor> gate_up_p,
        std::vector<torch::Tensor> gate_up_s,
        std::vector<torch::Tensor> w_down_p,
        std::vector<torch::Tensor> w_down_s,
        torch::Tensor lnf_w,
        std::vector<torch::Tensor> k_cs,
        std::vector<torch::Tensor> v_cs,
        int64_t n_h,
        int64_t n_kvh,
        int64_t hd,
        bool use_q4,
        float epsilon)
        : wte_embed(wte_emb.detach()),
          lm_head_packed(lmh_packed),
          lm_head_sb(lmh_sb),
          freqs_real(freqs_r),
          ln1_weights(std::move(ln1_ws)),
          c_attn_packed(std::move(c_attn_p)),
          c_attn_sb(std::move(c_attn_s)),
          c_proj_packed(std::move(c_proj_p)),
          c_proj_sb(std::move(c_proj_s)),
          ln2_weights(std::move(ln2_ws)),
          gate_up_packed(std::move(gate_up_p)),
          gate_up_sb(std::move(gate_up_s)),
          w_down_packed(std::move(w_down_p)),
          w_down_sb(std::move(w_down_s)),
          lnf_weight(lnf_w),
          k_caches(std::move(k_cs)),
          v_caches(std::move(v_cs)),
          n_layer((int64_t)this->ln1_weights.size()),
          n_head(n_h),
          n_kv_head(n_kvh),
          head_dim(hd),
          q_dim(n_h * hd),
          kv_dim(n_kvh * hd),
          q_pairs((uint32_t)((n_h * hd) / 2)),
          kv_pairs((uint32_t)((n_kvh * hd) / 2)),
          half_head_dim((uint32_t)(hd / 2)),
          max_seq_len((uint32_t)this->k_caches[0].size(2)),
          enable_gqa(n_h != n_kvh),
          is_q4(use_q4),
          eps(epsilon),
          initialized(true) {
        auto opts = wte_embed.options();
        int64_t d_model = q_dim;
        int64_t qkv_dim = q_dim + 2 * kv_dim;
        int64_t gu_dim = gate_up_packed[0].size(0);
        int64_t hidden_dim = gu_dim / 2;
        int64_t vocab_size = lm_head_packed.size(0);

        x_buf = torch::empty({1, 1, d_model}, opts);
        x_buf_2d = x_buf.view({1, d_model});
        norm_buf = torch::empty({1, 1, d_model}, opts);
        qkv_buf = torch::empty({1, 1, qkv_dim}, opts);
        q_view = qkv_buf.slice(2, 0, d_model).view({1, n_head, 1, head_dim});
        attn_out_buf = torch::empty({1, 1, d_model}, opts);
        attn_y_buf = torch::empty({1, 1, d_model}, opts);
        partial_O_buf = torch::empty({12, 16, 32, 2}, opts);
        partial_m_buf = torch::empty({12, 16}, opts.dtype(torch::kFloat32));
        partial_l_buf = torch::empty({12, 16}, opts.dtype(torch::kFloat32));
        gate_up_buf = torch::empty({1, 1, gu_dim}, opts);
        swiglu_buf = torch::empty({1, 1, hidden_dim}, opts);
        mlp_out_buf = torch::empty({1, 1, d_model}, opts);
        logits_buf = torch::empty({1, 1, vocab_size}, opts);
        x_buf_multi = torch::empty({1, 8, d_model}, opts);
        norm_buf_multi = torch::empty({1, 8, d_model}, opts);
        qkv_buf_multi = torch::empty({1, 8, qkv_dim}, opts);
        attn_y_buf_multi = torch::empty({1, 8, d_model}, opts);
        attn_out_buf_multi = torch::empty({1, 8, d_model}, opts);
        gate_up_buf_multi = torch::empty({1, 8, gu_dim}, opts);
        swiglu_buf_multi = torch::empty({1, 8, hidden_dim}, opts);
        mlp_out_buf_multi = torch::empty({1, 8, d_model}, opts);
        logits_buf_multi = torch::empty({1, 8, vocab_size}, opts);
    }

    torch::Tensor step(
        torch::Tensor idx,
        int64_t start_pos,
        int64_t write_pos,
        int64_t active_len) {
        at::NoGradGuard no_grad;
        at::index_select_out(x_buf_2d, wte_embed, 0, idx.view({-1}));
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        dispatch_rmsnorm_infer_into(encoder, x_buf, ln1_weights[0], norm_buf, eps);

        for (int64_t i = 0; i < n_layer; ++i) {
            if (is_q4) {
                dispatch_qmv_q4_g64_into(stream->commandEncoder(), norm_buf, c_attn_packed[i], c_attn_sb[i], qkv_buf);
            } else {
                dispatch_qmv_q8_g64_into(stream->commandEncoder(), norm_buf, c_attn_packed[i], c_attn_sb[i], qkv_buf);
            }

            dispatch_rope_kv_cache_write(
                stream->commandEncoder(),
                qkv_buf,
                freqs_real,
                k_caches[i],
                v_caches[i],
                q_pairs,
                kv_pairs,
                half_head_dim,
                max_seq_len,
                start_pos,
                (uint32_t)write_pos);

            uint32_t n_splits = (active_len <= 64) ? 1 : ((active_len <= 256) ? 4 : 8);
            dispatch_gqa_flash_decode_into(
                stream->commandEncoder(),
                qkv_buf,
                k_caches[i],
                v_caches[i],
                attn_y_buf,
                partial_O_buf,
                partial_m_buf,
                partial_l_buf,
                max_seq_len,
                (uint32_t)active_len,
                n_splits);

            if (is_q4) {
                dispatch_qmv_q4_g64_into(stream->commandEncoder(), attn_y_buf, c_proj_packed[i], c_proj_sb[i], attn_out_buf);
            } else {
                dispatch_qmv_q8_g64_into(stream->commandEncoder(), attn_y_buf, c_proj_packed[i], c_proj_sb[i], attn_out_buf);
            }

            dispatch_add_rmsnorm_infer_into(
                stream->commandEncoder(), x_buf, attn_out_buf, ln2_weights[i], norm_buf, eps);

            if (is_q4) {
                dispatch_qmv_q4_g64_into(stream->commandEncoder(), norm_buf, gate_up_packed[i], gate_up_sb[i], gate_up_buf);
            } else {
                dispatch_qmv_q8_g64_into(stream->commandEncoder(), norm_buf, gate_up_packed[i], gate_up_sb[i], gate_up_buf);
            }

            dispatch_swiglu_gate_up_into(stream->commandEncoder(), gate_up_buf, swiglu_buf);

            if (is_q4) {
                dispatch_qmv_q4_g64_into(stream->commandEncoder(), swiglu_buf, w_down_packed[i], w_down_sb[i], mlp_out_buf);
            } else {
                dispatch_qmv_q8_g64_into(stream->commandEncoder(), swiglu_buf, w_down_packed[i], w_down_sb[i], mlp_out_buf);
            }

            const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
            dispatch_add_rmsnorm_infer_into(
                stream->commandEncoder(), x_buf, mlp_out_buf, next_ln_w, norm_buf, eps);
        }

        if (is_q4) {
            dispatch_qmv_q4_g64_into(stream->commandEncoder(), norm_buf, lm_head_packed, lm_head_sb, logits_buf);
        } else {
            dispatch_qmv_q8_g64_into(stream->commandEncoder(), norm_buf, lm_head_packed, lm_head_sb, logits_buf);
        }

        return logits_buf;
    }

    torch::Tensor verify_step(
        torch::Tensor idx,
        int64_t start_pos,
        int64_t write_pos,
        int64_t active_len) {
        at::NoGradGuard no_grad;
        int64_t T = idx.numel();
        if (T == 1) {
            return step(idx, start_pos, write_pos, active_len);
        }
        TORCH_CHECK(T <= 8, "verify_step supports up to 8 tokens, got ", T);

        auto x_cur = x_buf_multi.narrow(1, 0, T);
        auto norm_cur = norm_buf_multi.narrow(1, 0, T);
        auto qkv_cur = qkv_buf_multi.narrow(1, 0, T);
        auto attn_y_cur = attn_y_buf_multi.narrow(1, 0, T);
        auto attn_out_cur = attn_out_buf_multi.narrow(1, 0, T);
        auto gate_up_cur = gate_up_buf_multi.narrow(1, 0, T);
        auto swiglu_cur = swiglu_buf_multi.narrow(1, 0, T);
        auto mlp_out_cur = mlp_out_buf_multi.narrow(1, 0, T);
        auto logits_cur = logits_buf_multi.narrow(1, 0, T);

        auto x_cur_2d = x_cur.view({T, q_dim});
        at::index_select_out(x_cur_2d, wte_embed, 0, idx.view({-1}));

        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        dispatch_rmsnorm_infer_into(encoder, x_cur, ln1_weights[0], norm_cur, eps);

        for (int64_t i = 0; i < n_layer; ++i) {
            if (is_q4) {
                dispatch_qmm_q4_g64_into(encoder, norm_cur, c_attn_packed[i], c_attn_sb[i], qkv_cur);
            } else {
                dispatch_qmm_q8_g64_into(encoder, norm_cur, c_attn_packed[i], c_attn_sb[i], qkv_cur);
            }

            for (int64_t t = 0; t < T; ++t) {
                auto qkv_t = qkv_cur.narrow(1, t, 1);
                dispatch_rope_kv_cache_write(
                    encoder,
                    qkv_t,
                    freqs_real,
                    k_caches[i],
                    v_caches[i],
                    q_pairs,
                    kv_pairs,
                    half_head_dim,
                    max_seq_len,
                    start_pos + t,
                    (uint32_t)(write_pos + t));
            }

            for (int64_t t = 0; t < T; ++t) {
                auto qkv_t = qkv_cur.narrow(1, t, 1);
                auto attn_y_t = attn_y_cur.narrow(1, t, 1);
                uint32_t cur_active = (uint32_t)(start_pos + t + 1);
                if (cur_active > max_seq_len) cur_active = max_seq_len;
                uint32_t n_splits = (cur_active <= 64) ? 1 : ((cur_active <= 256) ? 4 : 8);

                dispatch_gqa_flash_decode_into(
                    encoder,
                    qkv_t,
                    k_caches[i],
                    v_caches[i],
                    attn_y_t,
                    partial_O_buf,
                    partial_m_buf,
                    partial_l_buf,
                    max_seq_len,
                    cur_active,
                    n_splits);
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            if (is_q4) {
                dispatch_qmm_q4_g64_into(encoder, attn_y_cur, c_proj_packed[i], c_proj_sb[i], attn_out_cur);
            } else {
                dispatch_qmm_q8_g64_into(encoder, attn_y_cur, c_proj_packed[i], c_proj_sb[i], attn_out_cur);
            }

            dispatch_add_rmsnorm_infer_into(
                encoder, x_cur, attn_out_cur, ln2_weights[i], norm_cur, eps);

            if (is_q4) {
                dispatch_qmm_q4_g64_into(encoder, norm_cur, gate_up_packed[i], gate_up_sb[i], gate_up_cur);
            } else {
                dispatch_qmm_q8_g64_into(encoder, norm_cur, gate_up_packed[i], gate_up_sb[i], gate_up_cur);
            }

            dispatch_swiglu_gate_up_into(encoder, gate_up_cur, swiglu_cur);

            if (is_q4) {
                dispatch_qmm_q4_g64_into(encoder, swiglu_cur, w_down_packed[i], w_down_sb[i], mlp_out_cur);
            } else {
                dispatch_qmm_q8_g64_into(encoder, swiglu_cur, w_down_packed[i], w_down_sb[i], mlp_out_cur);
            }

            const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
            dispatch_add_rmsnorm_infer_into(
                encoder, x_cur, mlp_out_cur, next_ln_w, norm_cur, eps);
        }

        if (is_q4) {
            dispatch_qmm_q4_g64_into(encoder, norm_cur, lm_head_packed, lm_head_sb, logits_cur);
        } else {
            dispatch_qmm_q8_g64_into(encoder, norm_cur, lm_head_packed, lm_head_sb, logits_cur);
        }

        return logits_cur;
    }
};

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

torch::Tensor flash_attention_forward_mps(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal) {
    uint32_t B = q.size(0);
    uint32_t n_head = q.size(1);
    uint32_t seq_len_q = q.size(2);
    uint32_t head_dim = q.size(3);
    uint32_t seq_len_k = k.size(2);

    auto out = torch::empty_like(q);

    id<MTLBuffer> q_buf = getMTLBufferStorage(q);
    id<MTLBuffer> k_buf = getMTLBufferStorage(k);
    id<MTLBuffer> v_buf = getMTLBufferStorage(v);
    id<MTLBuffer> out_buf = getMTLBufferStorage(out);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    [encoder setComputePipelineState:fa_fwd_pso];
    [encoder setBuffer:q_buf offset:q.storage_offset() * q.element_size() atIndex:0];
    [encoder setBuffer:k_buf offset:k.storage_offset() * k.element_size() atIndex:1];
    [encoder setBuffer:v_buf offset:v.storage_offset() * v.element_size() atIndex:2];
    [encoder setBuffer:out_buf offset:out.storage_offset() * out.element_size() atIndex:3];

    [encoder setBytes:&seq_len_q length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&seq_len_k length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:6];
    uint32_t causal = is_causal ? 1 : 0;
    [encoder setBytes:&causal length:sizeof(uint32_t) atIndex:7];

    uint32_t BLOCK_Q = 32;
    uint32_t num_blocks_q = (seq_len_q + BLOCK_Q - 1) / BLOCK_Q;

    MTLSize gridSize = MTLSizeMake(B * n_head, num_blocks_q, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(BLOCK_Q, 1, 1);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];

    return out;
}


torch::Tensor qmv_q4_g64_mps(torch::Tensor x, torch::Tensor w_packed, torch::Tensor scales_biases) {
    int64_t N = w_packed.size(0);
    auto sizes = x.sizes().vec();
    sizes.back() = N;
    auto y = torch::empty(sizes, x.options());
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_qmv_q4_g64_into(stream->commandEncoder(), x, w_packed, scales_biases, y);
    return y;
}

torch::Tensor qmv_q8_g64_mps(torch::Tensor x, torch::Tensor w_packed, torch::Tensor scales_biases) {
    int64_t N = w_packed.size(0);
    auto sizes = x.sizes().vec();
    sizes.back() = N;
    auto y = torch::empty(sizes, x.options());
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_qmv_q8_g64_into(stream->commandEncoder(), x, w_packed, scales_biases, y);
    return y;
}

torch::Tensor qmm_q4_g64_mps(torch::Tensor x, torch::Tensor w_packed, torch::Tensor scales_biases) {
    int64_t N = w_packed.size(0);
    auto sizes = x.sizes().vec();
    sizes.back() = N;
    auto y = torch::empty(sizes, x.options());
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_qmm_q4_g64_into(stream->commandEncoder(), x, w_packed, scales_biases, y);
    return y;
}

torch::Tensor qmm_q8_g64_mps(torch::Tensor x, torch::Tensor w_packed, torch::Tensor scales_biases) {
    int64_t N = w_packed.size(0);
    auto sizes = x.sizes().vec();
    sizes.back() = N;
    auto y = torch::empty(sizes, x.options());
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_qmm_q8_g64_into(stream->commandEncoder(), x, w_packed, scales_biases, y);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_metal", &init_mps, "Initialize Metal with path");
  m.def("rmsnorm_forward_mps", &rmsnorm_forward_mps, "MPS RMSNorm Forward");
  m.def("rmsnorm_infer_mps", &rmsnorm_infer_mps, "MPS RMSNorm Inference (Zero-Alloc)");
  m.def("add_rmsnorm_infer_mps", &add_rmsnorm_infer_mps, "MPS Fused In-Place Add + RMSNorm Inference");
  m.def("argmax_last_token_mps", &argmax_last_token_mps, "MPS Fast Argmax over Vocabulary");
  m.def("rmsnorm_backward_mps", &rmsnorm_backward_mps, "MPS RMSNorm Backward");
  m.def("swiglu_forward_mps", &swiglu_forward_mps, "MPS SwiGLU Forward");
  m.def("swiglu_gate_up_mps", &swiglu_gate_up_mps, "MPS SwiGLU Gate+Up Single-Buffer Forward");
  m.def("swiglu_backward_mps", &swiglu_backward_mps, "MPS SwiGLU Backward");
  m.def("apply_rope_mps", &apply_rope_mps, "MPS Apply RoPE");
  m.def("apply_rope_inplace_mps", &apply_rope_inplace_mps, "MPS Apply RoPE In-Place");
  m.def("rope_kv_cache_write_mps", &rope_kv_cache_write_mps, "MPS Fused RoPE + KV Cache Direct Write");
  m.def("transformer_decode_step_mps", &transformer_decode_step_mps, "MPS C++ 12-Layer Whole-Model Decode Step");
  pybind11::class_<DecodeGraphState, std::shared_ptr<DecodeGraphState>>(m, "MPSDecodeRunner")
      .def(pybind11::init<
          torch::Tensor,
          torch::Tensor,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          torch::Tensor,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          int64_t,
          int64_t,
          int64_t,
          float>())
      .def("step", &DecodeGraphState::step)
      .def("verify_step", &DecodeGraphState::verify_step);
  pybind11::class_<QuantizedDecodeGraphState, std::shared_ptr<QuantizedDecodeGraphState>>(m, "MPSQuantizedDecodeRunner")
      .def(pybind11::init<
          torch::Tensor,
          torch::Tensor,
          torch::Tensor,
          torch::Tensor,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          torch::Tensor,
          std::vector<torch::Tensor>,
          std::vector<torch::Tensor>,
          int64_t,
          int64_t,
          int64_t,
          bool,
          float>())
      .def("step", &QuantizedDecodeGraphState::step)
      .def("verify_step", &QuantizedDecodeGraphState::verify_step);
  m.def("cross_entropy_forward_mps", &cross_entropy_forward_mps, "MPS CE Fwd");
  m.def("cross_entropy_backward_mps", &cross_entropy_backward_mps, "MPS CE Bwd");
  m.def("flash_attention_forward_mps", &flash_attention_forward_mps, "MPS FlashAttention Fwd");
  m.def("qmv_q4_g64_mps", &qmv_q4_g64_mps, "MPS Fast Q4_G64 Quantized GEMV");
  m.def("qmm_q4_g64_mps", &qmm_q4_g64_mps, "MPS Batched Q4_G64 Quantized GEMM");
  m.def("qmv_q8_g64_mps", &qmv_q8_g64_mps, "MPS Fast Q8_G64 Quantized GEMV");
  m.def("qmm_q8_g64_mps", &qmm_q8_g64_mps, "MPS Batched Q8_G64 Quantized GEMM");
  m.def("gqa_flash_decode_mps", &gqa_flash_decode_mps, "MPS Fast Split-KV Flash-Decode for GQA");
}
