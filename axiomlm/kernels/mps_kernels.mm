#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <tuple>

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
static id<MTLComputePipelineState> embed_multi_half_pso = nil;
static id<MTLComputePipelineState> argmax_multi_half_pso = nil;
static id<MTLComputePipelineState> qmv_q4_marlin_pso = nil;
static id<MTLComputePipelineState> qmm_q4_marlin_m8_pso = nil;
static id<MTLComputePipelineState> qmm_q4_marlin_pso = nil;
static id<MTLComputePipelineState> rope_tree_kv_write_half_pso = nil;
static id<MTLComputePipelineState> gqa_tree_verify_split_kv_pso = nil;
static id<MTLComputePipelineState> gqa_tree_verify_reduce_pso = nil;
static id<MTLComputePipelineState> compact_tree_kv_half_pso = nil;
static id<MTLComputePipelineState> embed_q8_g64_gather_half_pso = nil;
static id<MTLComputePipelineState> rope_kv_write_fp8_pso = nil;
static id<MTLComputePipelineState> rope_kv_write_int4_pso = nil;
static id<MTLComputePipelineState> rope_tree_kv_write_fp8_pso = nil;
static id<MTLComputePipelineState> rope_tree_kv_write_int4_pso = nil;
static id<MTLComputePipelineState> dequantize_kv_fp8_pso = nil;
static id<MTLComputePipelineState> dequantize_kv_int4_pso = nil;
static id<MTLComputePipelineState> gqa_split_kv_fp8_pso = nil;
static id<MTLComputePipelineState> gqa_split_kv_int4_pso = nil;
static id<MTLComputePipelineState> gqa_tree_verify_split_kv_fp8_pso = nil;
static id<MTLComputePipelineState> gqa_tree_verify_split_kv_int4_pso = nil;
static id<MTLComputePipelineState> compact_tree_kv_fp8_pso = nil;
static id<MTLComputePipelineState> compact_tree_kv_int4_pso = nil;
static id<MTLComputePipelineState> qmv_ternary_half_pso = nil;
static id<MTLComputePipelineState> qmm_ternary_half_pso = nil;
static id<MTLComputePipelineState> qmv_q2_k_half_pso = nil;
static id<MTLComputePipelineState> qmm_q2_k_half_pso = nil;
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
            embed_multi_half_pso = get_pipeline(mtl_device, mtl_library, @"embed_multi_half_kernel");
            argmax_multi_half_pso = get_pipeline(mtl_device, mtl_library, @"argmax_multi_half_kernel");
            qmv_q4_marlin_pso = get_pipeline(mtl_device, mtl_library, @"qmv_q4_marlin_apple_kernel");
            qmm_q4_marlin_m8_pso = get_pipeline(mtl_device, mtl_library, @"qmm_q4_marlin_m8_apple_kernel");
            qmm_q4_marlin_pso = get_pipeline(mtl_device, mtl_library, @"qmm_q4_marlin_apple_kernel");
            rope_tree_kv_write_half_pso = get_pipeline(mtl_device, mtl_library, @"rope_tree_kv_cache_write_half_kernel");
            gqa_tree_verify_split_kv_pso = get_pipeline(mtl_device, mtl_library, @"gqa_tree_verify_split_kv_half_kernel");
            gqa_tree_verify_reduce_pso = get_pipeline(mtl_device, mtl_library, @"gqa_tree_verify_reduce_half_kernel");
            compact_tree_kv_half_pso = get_pipeline(mtl_device, mtl_library, @"compact_tree_kv_half_kernel");
            embed_q8_g64_gather_half_pso = get_pipeline(mtl_device, mtl_library, @"embed_q8_g64_gather_half_kernel");
            rope_kv_write_fp8_pso = get_pipeline(mtl_device, mtl_library, @"rope_kv_cache_write_fp8_kernel");
            rope_kv_write_int4_pso = get_pipeline(mtl_device, mtl_library, @"rope_kv_cache_write_int4_kernel");
            rope_tree_kv_write_fp8_pso = get_pipeline(mtl_device, mtl_library, @"rope_tree_kv_cache_write_fp8_kernel");
            rope_tree_kv_write_int4_pso = get_pipeline(mtl_device, mtl_library, @"rope_tree_kv_cache_write_int4_kernel");
            dequantize_kv_fp8_pso = get_pipeline(mtl_device, mtl_library, @"dequantize_kv_cache_fp8_kernel");
            dequantize_kv_int4_pso = get_pipeline(mtl_device, mtl_library, @"dequantize_kv_cache_int4_kernel");
            gqa_split_kv_fp8_pso = get_pipeline(mtl_device, mtl_library, @"gqa_flash_decode_split_kv_fp8_kernel");
            gqa_split_kv_int4_pso = get_pipeline(mtl_device, mtl_library, @"gqa_flash_decode_split_kv_int4_kernel");
            gqa_tree_verify_split_kv_fp8_pso = get_pipeline(mtl_device, mtl_library, @"gqa_tree_verify_split_kv_fp8_kernel");
            gqa_tree_verify_split_kv_int4_pso = get_pipeline(mtl_device, mtl_library, @"gqa_tree_verify_split_kv_int4_kernel");
            compact_tree_kv_fp8_pso = get_pipeline(mtl_device, mtl_library, @"compact_tree_kv_fp8_kernel");
            compact_tree_kv_int4_pso = get_pipeline(mtl_device, mtl_library, @"compact_tree_kv_int4_kernel");
            qmv_ternary_half_pso = get_pipeline(mtl_device, mtl_library, @"qmv_ternary_half_kernel");
            qmm_ternary_half_pso = get_pipeline(mtl_device, mtl_library, @"qmm_ternary_half_kernel");
            qmv_q2_k_half_pso = get_pipeline(mtl_device, mtl_library, @"qmv_q2_k_half_kernel");
            qmm_q2_k_half_pso = get_pipeline(mtl_device, mtl_library, @"qmm_q2_k_half_kernel");
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

static inline void dispatch_embed_multi(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> idx_buf,
    const torch::Tensor& wte,
    torch::Tensor& x_out,
    uint32_t D,
    uint32_t T) {
    [encoder setComputePipelineState:embed_multi_half_pso];
    [encoder setBuffer:idx_buf offset:0 atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(wte) offset:wte.storage_offset() * wte.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(x_out) offset:x_out.storage_offset() * x_out.element_size() atIndex:2];
    [encoder setBytes:&D length:sizeof(uint32_t) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static inline void dispatch_embed_q8_g64_gather_multi(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> idx_buf,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales_biases,
    torch::Tensor& x_out,
    uint32_t D,
    uint32_t T) {
    [encoder setComputePipelineState:embed_q8_g64_gather_half_pso];
    [encoder setBuffer:idx_buf offset:0 atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() * w_packed.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales_biases) offset:scales_biases.storage_offset() * scales_biases.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(x_out) offset:x_out.storage_offset() * x_out.element_size() atIndex:3];
    [encoder setBytes:&D length:sizeof(uint32_t) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static inline void dispatch_embed_q8_g64_gather_multi(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& idx,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales_biases,
    torch::Tensor& x_out,
    uint32_t D,
    uint32_t T) {
    [encoder setComputePipelineState:embed_q8_g64_gather_half_pso];
    [encoder setBuffer:getMTLBufferStorage(idx) offset:idx.storage_offset() * idx.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() * w_packed.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales_biases) offset:scales_biases.storage_offset() * scales_biases.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(x_out) offset:x_out.storage_offset() * x_out.element_size() atIndex:3];
    [encoder setBytes:&D length:sizeof(uint32_t) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static inline void dispatch_argmax_multi(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& logits,
    id<MTLBuffer> out_idx_buf,
    uint32_t V,
    uint32_t T) {
    [encoder setComputePipelineState:argmax_multi_half_pso];
    [encoder setBuffer:getMTLBufferStorage(logits) offset:logits.storage_offset() * logits.element_size() atIndex:0];
    [encoder setBuffer:out_idx_buf offset:0 atIndex:1];
    [encoder setBytes:&V length:sizeof(uint32_t) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static inline std::vector<int64_t> find_candidate_ngram_cpp(
    const std::vector<int64_t>& input_ids,
    int64_t ngram_size,
    int64_t max_spec_tokens) {
    int64_t len = (int64_t)input_ids.size();
    if (len < ngram_size + 1) {
        return {};
    }
    int64_t search_end = len - ngram_size;
    const int64_t* target = input_ids.data() + search_end;

    for (int64_t i = search_end - 1; i >= 0; --i) {
        bool match = true;
        for (int64_t k = 0; k < ngram_size; ++k) {
            if (input_ids[i + k] != target[k]) {
                match = false;
                break;
            }
        }
        if (match) {
            int64_t start = i + ngram_size;
            int64_t end = std::min(start + max_spec_tokens, search_end);
            if (start < end) {
                return std::vector<int64_t>(input_ids.begin() + start, input_ids.begin() + end);
            }
        }
    }
    return {};
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
    uint32_t write_pos,
    const torch::Tensor& qkv_bias = torch::Tensor()) {
    uint32_t total_pairs = q_pairs + (kv_pairs << 1);
    bool is_half = (qkv.scalar_type() == at::kHalf);
    id<MTLComputePipelineState> pso = is_half ? rope_kv_write_half_pso : rope_kv_write_pso;
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
    if (is_half) {
        uint32_t has_bias = qkv_bias.defined() ? 1u : 0u;
        if (has_bias) {
            [encoder setBuffer:getMTLBufferStorage(qkv_bias)
                        offset:qkv_bias.storage_offset() * qkv_bias.element_size()
                       atIndex:9];
        } else {
            [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:9];
        }
        [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:10];
    }

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_pairs + threadGroupSize - 1) / threadGroupSize, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

static inline void dispatch_rope_kv_cache_write_offset(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> qkv_buf,
    NSUInteger qkv_offset,
    const torch::Tensor& freqs_cis,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    uint32_t q_pairs,
    uint32_t kv_pairs,
    uint32_t half_head_dim,
    uint32_t max_seq_len,
    int64_t start_pos,
    uint32_t write_pos,
    const torch::Tensor& qkv_bias = torch::Tensor()) {
    uint32_t total_pairs = q_pairs + (kv_pairs << 1);
    [encoder setComputePipelineState:rope_kv_write_half_pso];

    NSUInteger freqs_offset = (freqs_cis.storage_offset() + start_pos * half_head_dim * 2) * freqs_cis.element_size();
    NSUInteger k_offset = k_cache.storage_offset() * k_cache.element_size();
    NSUInteger v_offset = v_cache.storage_offset() * v_cache.element_size();

    [encoder setBuffer:qkv_buf offset:qkv_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(freqs_cis) offset:freqs_offset atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(k_cache) offset:k_offset atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(v_cache) offset:v_offset atIndex:3];
    [encoder setBytes:&q_pairs length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&kv_pairs length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&write_pos length:sizeof(uint32_t) atIndex:8];
    uint32_t has_bias = qkv_bias.defined() ? 1u : 0u;
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(qkv_bias)
                    offset:qkv_bias.storage_offset() * qkv_bias.element_size()
                   atIndex:9];
    } else {
        [encoder setBuffer:qkv_buf offset:qkv_offset atIndex:9];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:10];

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_pairs + threadGroupSize - 1) / threadGroupSize, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

static inline void dispatch_rope_tree_kv_cache_write_into(
    id<MTLComputeCommandEncoder> encoder,
    torch::Tensor& qkv,
    const torch::Tensor& freqs_cis,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    id<MTLBuffer> tree_depths_buf,
    NSUInteger depths_offset,
    uint32_t q_pairs,
    uint32_t kv_pairs,
    uint32_t half_head_dim,
    uint32_t max_seq_len,
    uint32_t start_pos,
    uint32_t write_pos,
    uint32_t T,
    const torch::Tensor& qkv_bias = torch::Tensor()) {
    uint32_t total_pairs = q_pairs + (kv_pairs << 1);
    uint32_t max_rope_len = (uint32_t)freqs_cis.size(0);
    [encoder setComputePipelineState:rope_tree_kv_write_half_pso];

    NSUInteger qkv_offset = qkv.storage_offset() * qkv.element_size();
    NSUInteger freqs_offset = freqs_cis.storage_offset() * freqs_cis.element_size();
    NSUInteger k_offset = k_cache.storage_offset() * k_cache.element_size();
    NSUInteger v_offset = v_cache.storage_offset() * v_cache.element_size();

    [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(freqs_cis) offset:freqs_offset atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(k_cache) offset:k_offset atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(v_cache) offset:v_offset atIndex:3];
    [encoder setBuffer:tree_depths_buf offset:depths_offset atIndex:4];
    [encoder setBytes:&q_pairs length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&kv_pairs length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&max_rope_len length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&start_pos length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&write_pos length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&T length:sizeof(uint32_t) atIndex:12];
    uint32_t has_bias = qkv_bias.defined() ? 1u : 0u;
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(qkv_bias)
                    offset:qkv_bias.storage_offset() * qkv_bias.element_size()
                   atIndex:13];
    } else {
        [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:13];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:14];

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_pairs + threadGroupSize - 1) / threadGroupSize, T, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

static inline void dispatch_rope_tree_kv_cache_write_bytes(
    id<MTLComputeCommandEncoder> encoder,
    torch::Tensor& qkv,
    const torch::Tensor& freqs_cis,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    const int32_t* tree_depths,
    uint32_t q_pairs,
    uint32_t kv_pairs,
    uint32_t half_head_dim,
    uint32_t max_seq_len,
    uint32_t start_pos,
    uint32_t write_pos,
    uint32_t T,
    const torch::Tensor& qkv_bias = torch::Tensor()) {
    uint32_t total_pairs = q_pairs + (kv_pairs << 1);
    uint32_t max_rope_len = (uint32_t)freqs_cis.size(0);
    [encoder setComputePipelineState:rope_tree_kv_write_half_pso];

    NSUInteger qkv_offset = qkv.storage_offset() * qkv.element_size();
    NSUInteger freqs_offset = freqs_cis.storage_offset() * freqs_cis.element_size();
    NSUInteger k_offset = k_cache.storage_offset() * k_cache.element_size();
    NSUInteger v_offset = v_cache.storage_offset() * v_cache.element_size();

    [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(freqs_cis) offset:freqs_offset atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(k_cache) offset:k_offset atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(v_cache) offset:v_offset atIndex:3];
    [encoder setBytes:tree_depths length:T * sizeof(int32_t) atIndex:4];
    [encoder setBytes:&q_pairs length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&kv_pairs length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&max_rope_len length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&start_pos length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&write_pos length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&T length:sizeof(uint32_t) atIndex:12];
    uint32_t has_bias = qkv_bias.defined() ? 1u : 0u;
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(qkv_bias)
                    offset:qkv_bias.storage_offset() * qkv_bias.element_size()
                   atIndex:13];
    } else {
        [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:13];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:14];

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_pairs + threadGroupSize - 1) / threadGroupSize, T, 1)
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

void rope_kv_cache_write_fp8_mps(
    torch::Tensor qkv,
    torch::Tensor freqs_cis,
    torch::Tensor k_cache_fp8,
    torch::Tensor v_cache_fp8,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    int64_t q_dim,
    int64_t kv_dim,
    int64_t head_dim,
    int64_t start_pos,
    int64_t write_pos,
    c10::optional<torch::Tensor> qkv_bias = c10::nullopt) {
    uint32_t q_pairs = q_dim / 2;
    uint32_t kv_pairs = kv_dim / 2;
    uint32_t half_head_dim = head_dim / 2;
    uint32_t max_seq_len = (k_cache_fp8.dim() == 4) ? (uint32_t)k_cache_fp8.size(2) : (uint32_t)k_cache_fp8.size(1);
    uint32_t total_pairs = q_pairs + (kv_pairs << 1);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:rope_kv_write_fp8_pso];

    NSUInteger qkv_offset = qkv.storage_offset() * qkv.element_size();
    NSUInteger freqs_offset = (freqs_cis.storage_offset() + start_pos * half_head_dim * 2) * freqs_cis.element_size();
    NSUInteger k_offset = k_cache_fp8.storage_offset() * k_cache_fp8.element_size();
    NSUInteger v_offset = v_cache_fp8.storage_offset() * v_cache_fp8.element_size();
    NSUInteger ks_offset = k_scales.storage_offset() * k_scales.element_size();
    NSUInteger vs_offset = v_scales.storage_offset() * v_scales.element_size();

    [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(freqs_cis) offset:freqs_offset atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(k_cache_fp8) offset:k_offset atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(v_cache_fp8) offset:v_offset atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(k_scales) offset:ks_offset atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(v_scales) offset:vs_offset atIndex:5];
    [encoder setBytes:&q_pairs length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&kv_pairs length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:9];
    uint32_t w_pos = (uint32_t)write_pos;
    [encoder setBytes:&w_pos length:sizeof(uint32_t) atIndex:10];

    torch::Tensor bias_t = qkv_bias.has_value() ? qkv_bias.value() : torch::Tensor();
    uint32_t has_bias = bias_t.defined() ? 1u : 0u;
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(bias_t) offset:bias_t.storage_offset() * bias_t.element_size() atIndex:11];
    } else {
        [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:11];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:12];

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_pairs + threadGroupSize - 1) / threadGroupSize, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

void rope_kv_cache_write_int4_mps(
    torch::Tensor qkv,
    torch::Tensor freqs_cis,
    torch::Tensor k_cache_int4,
    torch::Tensor v_cache_int4,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    int64_t q_dim,
    int64_t kv_dim,
    int64_t head_dim,
    int64_t start_pos,
    int64_t write_pos,
    c10::optional<torch::Tensor> qkv_bias = c10::nullopt) {
    uint32_t q_pairs = q_dim / 2;
    uint32_t kv_pairs = kv_dim / 2;
    uint32_t half_head_dim = head_dim / 2;
    uint32_t max_seq_len = (k_cache_int4.dim() == 4) ? (uint32_t)k_cache_int4.size(2) : (uint32_t)k_cache_int4.size(1);
    uint32_t total_pairs = q_pairs + (kv_pairs << 1);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:rope_kv_write_int4_pso];

    NSUInteger qkv_offset = qkv.storage_offset() * qkv.element_size();
    NSUInteger freqs_offset = (freqs_cis.storage_offset() + start_pos * half_head_dim * 2) * freqs_cis.element_size();
    NSUInteger k_offset = k_cache_int4.storage_offset() * k_cache_int4.element_size();
    NSUInteger v_offset = v_cache_int4.storage_offset() * v_cache_int4.element_size();
    NSUInteger ks_offset = k_scales.storage_offset() * k_scales.element_size();
    NSUInteger vs_offset = v_scales.storage_offset() * v_scales.element_size();

    [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(freqs_cis) offset:freqs_offset atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(k_cache_int4) offset:k_offset atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(v_cache_int4) offset:v_offset atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(k_scales) offset:ks_offset atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(v_scales) offset:vs_offset atIndex:5];
    [encoder setBytes:&q_pairs length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&kv_pairs length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:9];
    uint32_t w_pos = (uint32_t)write_pos;
    [encoder setBytes:&w_pos length:sizeof(uint32_t) atIndex:10];

    torch::Tensor bias_t = qkv_bias.has_value() ? qkv_bias.value() : torch::Tensor();
    uint32_t has_bias = bias_t.defined() ? 1u : 0u;
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(bias_t) offset:bias_t.storage_offset() * bias_t.element_size() atIndex:11];
    } else {
        [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:11];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:12];

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_pairs + threadGroupSize - 1) / threadGroupSize, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

torch::Tensor dequantize_kv_cache_fp8_mps(
    torch::Tensor cache_fp8,
    torch::Tensor scales,
    int64_t n_kv_head,
    int64_t seq_len,
    int64_t head_dim) {
    uint32_t half_head_dim = head_dim / 2;
    bool is_4d = (cache_fp8.dim() == 4);
    int64_t B = is_4d ? cache_fp8.size(0) : 1;
    uint32_t total_elements = (uint32_t)(B * n_kv_head * seq_len * half_head_dim);
    auto out = is_4d
        ? torch::empty({B, n_kv_head, seq_len, head_dim}, torch::TensorOptions().dtype(torch::kHalf).device(cache_fp8.device()))
        : torch::empty({n_kv_head, seq_len, head_dim}, torch::TensorOptions().dtype(torch::kHalf).device(cache_fp8.device()));

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:dequantize_kv_fp8_pso];

    NSUInteger c_offset = cache_fp8.storage_offset() * cache_fp8.element_size();
    NSUInteger s_offset = scales.storage_offset() * scales.element_size();
    NSUInteger out_offset = out.storage_offset() * out.element_size();

    [encoder setBuffer:getMTLBufferStorage(cache_fp8) offset:c_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(scales) offset:s_offset atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out_offset atIndex:2];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:3];
    uint32_t slen = (uint32_t)seq_len;
    [encoder setBytes:&slen length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&total_elements length:sizeof(uint32_t) atIndex:5];

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_elements + threadGroupSize - 1) / threadGroupSize, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];

    return out;
}

torch::Tensor dequantize_kv_cache_int4_mps(
    torch::Tensor cache_int4,
    torch::Tensor scales,
    int64_t n_kv_head,
    int64_t seq_len,
    int64_t head_dim) {
    uint32_t half_head_dim = head_dim / 2;
    bool is_4d = (cache_int4.dim() == 4);
    int64_t B = is_4d ? cache_int4.size(0) : 1;
    uint32_t total_elements = (uint32_t)(B * n_kv_head * seq_len * half_head_dim);
    auto out = is_4d
        ? torch::empty({B, n_kv_head, seq_len, head_dim}, torch::TensorOptions().dtype(torch::kHalf).device(cache_int4.device()))
        : torch::empty({n_kv_head, seq_len, head_dim}, torch::TensorOptions().dtype(torch::kHalf).device(cache_int4.device()));

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:dequantize_kv_int4_pso];

    NSUInteger c_offset = cache_int4.storage_offset() * cache_int4.element_size();
    NSUInteger s_offset = scales.storage_offset() * scales.element_size();
    NSUInteger out_offset = out.storage_offset() * out.element_size();

    [encoder setBuffer:getMTLBufferStorage(cache_int4) offset:c_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(scales) offset:s_offset atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out_offset atIndex:2];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:3];
    uint32_t slen = (uint32_t)seq_len;
    [encoder setBytes:&slen length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&total_elements length:sizeof(uint32_t) atIndex:5];

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_elements + threadGroupSize - 1) / threadGroupSize, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];

    return out;
}

void rope_tree_kv_write_fp8_mps(
    torch::Tensor qkv,
    torch::Tensor freqs_cis,
    torch::Tensor k_cache_fp8,
    torch::Tensor v_cache_fp8,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    torch::Tensor tree_depths,
    int64_t q_dim,
    int64_t kv_dim,
    int64_t head_dim,
    int64_t start_pos,
    int64_t write_pos,
    int64_t T,
    c10::optional<torch::Tensor> qkv_bias = c10::nullopt) {
    uint32_t q_pairs = q_dim / 2;
    uint32_t kv_pairs = kv_dim / 2;
    uint32_t half_head_dim = head_dim / 2;
    uint32_t max_seq_len = k_cache_fp8.size(2);
    uint32_t total_pairs = q_pairs + (kv_pairs << 1);
    uint32_t max_rope_len = (uint32_t)freqs_cis.size(0);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:rope_tree_kv_write_fp8_pso];

    NSUInteger qkv_offset = qkv.storage_offset() * qkv.element_size();
    NSUInteger freqs_offset = freqs_cis.storage_offset() * freqs_cis.element_size();
    NSUInteger k_offset = k_cache_fp8.storage_offset() * k_cache_fp8.element_size();
    NSUInteger v_offset = v_cache_fp8.storage_offset() * v_cache_fp8.element_size();
    NSUInteger ks_offset = k_scales.storage_offset() * k_scales.element_size();
    NSUInteger vs_offset = v_scales.storage_offset() * v_scales.element_size();
    NSUInteger depths_offset = tree_depths.storage_offset() * tree_depths.element_size();

    [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(freqs_cis) offset:freqs_offset atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(k_cache_fp8) offset:k_offset atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(v_cache_fp8) offset:v_offset atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(k_scales) offset:ks_offset atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(v_scales) offset:vs_offset atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(tree_depths) offset:depths_offset atIndex:6];
    [encoder setBytes:&q_pairs length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&kv_pairs length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&max_rope_len length:sizeof(uint32_t) atIndex:11];
    uint32_t s_pos = (uint32_t)start_pos;
    uint32_t w_pos = (uint32_t)write_pos;
    uint32_t t_cnt = (uint32_t)T;
    [encoder setBytes:&s_pos length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&w_pos length:sizeof(uint32_t) atIndex:13];
    [encoder setBytes:&t_cnt length:sizeof(uint32_t) atIndex:14];

    torch::Tensor bias_t = qkv_bias.has_value() ? qkv_bias.value() : torch::Tensor();
    uint32_t has_bias = bias_t.defined() ? 1u : 0u;
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(bias_t) offset:bias_t.storage_offset() * bias_t.element_size() atIndex:15];
    } else {
        [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:15];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:16];

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_pairs + threadGroupSize - 1) / threadGroupSize, T, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

void rope_tree_kv_write_int4_mps(
    torch::Tensor qkv,
    torch::Tensor freqs_cis,
    torch::Tensor k_cache_int4,
    torch::Tensor v_cache_int4,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    torch::Tensor tree_depths,
    int64_t q_dim,
    int64_t kv_dim,
    int64_t head_dim,
    int64_t start_pos,
    int64_t write_pos,
    int64_t T,
    c10::optional<torch::Tensor> qkv_bias = c10::nullopt) {
    uint32_t q_pairs = q_dim / 2;
    uint32_t kv_pairs = kv_dim / 2;
    uint32_t half_head_dim = head_dim / 2;
    uint32_t max_seq_len = k_cache_int4.size(2);
    uint32_t total_pairs = q_pairs + (kv_pairs << 1);
    uint32_t max_rope_len = (uint32_t)freqs_cis.size(0);

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:rope_tree_kv_write_int4_pso];

    NSUInteger qkv_offset = qkv.storage_offset() * qkv.element_size();
    NSUInteger freqs_offset = freqs_cis.storage_offset() * freqs_cis.element_size();
    NSUInteger k_offset = k_cache_int4.storage_offset() * k_cache_int4.element_size();
    NSUInteger v_offset = v_cache_int4.storage_offset() * v_cache_int4.element_size();
    NSUInteger ks_offset = k_scales.storage_offset() * k_scales.element_size();
    NSUInteger vs_offset = v_scales.storage_offset() * v_scales.element_size();
    NSUInteger depths_offset = tree_depths.storage_offset() * tree_depths.element_size();

    [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(freqs_cis) offset:freqs_offset atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(k_cache_int4) offset:k_offset atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(v_cache_int4) offset:v_offset atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(k_scales) offset:ks_offset atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(v_scales) offset:vs_offset atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(tree_depths) offset:depths_offset atIndex:6];
    [encoder setBytes:&q_pairs length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&kv_pairs length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&max_rope_len length:sizeof(uint32_t) atIndex:11];
    uint32_t s_pos = (uint32_t)start_pos;
    uint32_t w_pos = (uint32_t)write_pos;
    uint32_t t_cnt = (uint32_t)T;
    [encoder setBytes:&s_pos length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&w_pos length:sizeof(uint32_t) atIndex:13];
    [encoder setBytes:&t_cnt length:sizeof(uint32_t) atIndex:14];

    torch::Tensor bias_t = qkv_bias.has_value() ? qkv_bias.value() : torch::Tensor();
    uint32_t has_bias = bias_t.defined() ? 1u : 0u;
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(bias_t) offset:bias_t.storage_offset() * bias_t.element_size() atIndex:15];
    } else {
        [encoder setBuffer:getMTLBufferStorage(qkv) offset:qkv_offset atIndex:15];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:16];

    NSUInteger threadGroupSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_pairs + threadGroupSize - 1) / threadGroupSize, T, 1)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
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
    uint32_t m_tiles = (M + 7) / 8;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, m_tiles, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static inline void dispatch_qmv_q4_marlin_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& w_marlin,
    const torch::Tensor& sb_marlin,
    torch::Tensor& y,
    const torch::Tensor& lin_bias = torch::Tensor()) {
    uint32_t K = (uint32_t)x.size(-1);
    uint32_t N = (uint32_t)y.size(-1);
    uint32_t N_tiles = (N + 7u) / 8u;
    uint32_t K_groups = K >> 6;

    uint32_t split_k = 2u;
    uint32_t tiles_per_tg = 2u;
    if (K_groups < 4u) {
        split_k = 1u;
        tiles_per_tg = 4u;
    } else if (N_tiles <= 160u && K_groups >= 8u) {
        split_k = 4u;
        tiles_per_tg = 1u;
    }
    uint32_t num_tg = (N_tiles + tiles_per_tg - 1u) / tiles_per_tg;
    uint32_t threads_per_tg = 32u * split_k * tiles_per_tg;
    uint32_t has_bias = lin_bias.defined() ? 1u : 0u;

    [encoder setComputePipelineState:qmv_q4_marlin_pso];
    [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_marlin) offset:w_marlin.storage_offset() * w_marlin.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(sb_marlin) offset:sb_marlin.storage_offset() * sb_marlin.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&split_k length:sizeof(uint32_t) atIndex:6];
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(lin_bias) offset:lin_bias.storage_offset() * lin_bias.element_size() atIndex:7];
    } else {
        [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:7];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_tg, 1, 1)];
}

static inline void dispatch_qmm_q4_marlin_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& w_marlin,
    const torch::Tensor& sb_marlin,
    torch::Tensor& y,
    const torch::Tensor& lin_bias = torch::Tensor()) {
    uint32_t K = (uint32_t)x.size(-1);
    uint32_t M = (uint32_t)(x.numel() / K);
    uint32_t N = (uint32_t)y.size(-1);

    if (M == 1u) {
        dispatch_qmv_q4_marlin_into(encoder, x, w_marlin, sb_marlin, y, lin_bias);
        return;
    }

    uint32_t N_tiles = (N + 7u) / 8u;
    uint32_t K_groups = K >> 6;
    uint32_t has_bias = lin_bias.defined() ? 1u : 0u;

    if (M <= 8u) {
        uint32_t split_k = 2u;
        uint32_t tiles_per_tg = 2u;
        if (K_groups < 4u) {
            split_k = 1u;
            tiles_per_tg = 4u;
        } else if (N_tiles <= 160u && K_groups >= 8u) {
            split_k = 4u;
            tiles_per_tg = 1u;
        }
        uint32_t num_tg = (N_tiles + tiles_per_tg - 1u) / tiles_per_tg;
        uint32_t threads_per_tg = 32u * split_k * tiles_per_tg;

        [encoder setComputePipelineState:qmm_q4_marlin_m8_pso];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(w_marlin) offset:w_marlin.storage_offset() * w_marlin.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(sb_marlin) offset:sb_marlin.storage_offset() * sb_marlin.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
        [encoder setBytes:&M length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&split_k length:sizeof(uint32_t) atIndex:7];
        if (has_bias) {
            [encoder setBuffer:getMTLBufferStorage(lin_bias) offset:lin_bias.storage_offset() * lin_bias.element_size() atIndex:8];
        } else {
            [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:8];
        }
        [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_tg, 1, 1)];
    } else {
        uint32_t m_tiles = (M + 31u) / 32u;
        uint32_t n_pairs = (N_tiles + 1u) / 2u;
        uint32_t split_k = 1u;
        uint32_t pairs_per_tg = 2u;
        if (K_groups >= 4u && (n_pairs * m_tiles) <= 384u) {
            if (n_pairs <= 80u && m_tiles == 1u && K_groups >= 8u) {
                split_k = 4u;
                pairs_per_tg = 1u;
            } else {
                split_k = 2u;
                pairs_per_tg = 2u;
            }
        }
        uint32_t num_tg_x = (n_pairs + pairs_per_tg - 1u) / pairs_per_tg;
        uint32_t threads_per_tg = 32u * split_k * pairs_per_tg;

        [encoder setComputePipelineState:qmm_q4_marlin_pso];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(w_marlin) offset:w_marlin.storage_offset() * w_marlin.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(sb_marlin) offset:sb_marlin.storage_offset() * sb_marlin.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
        [encoder setBytes:&M length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&split_k length:sizeof(uint32_t) atIndex:7];
        if (has_bias) {
            [encoder setBuffer:getMTLBufferStorage(lin_bias) offset:lin_bias.storage_offset() * lin_bias.element_size() atIndex:8];
        } else {
            [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:8];
        }
        [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(num_tg_x, m_tiles, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_tg, 1, 1)];
    }
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
    uint32_t num_splits,
    uint32_t n_head,
    uint32_t n_kv_head,
    uint32_t head_dim) {
    float scale = 1.0f / std::sqrt(float(head_dim));
    uint32_t n_rep = n_head / n_kv_head;
    uint32_t rep_tiles = (n_rep + 7u) / 8u;

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
    [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:13];

    MTLSize gridSize = MTLSizeMake(n_kv_head, num_splits, rep_tiles);
    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1) {
        [encoder setComputePipelineState:gqa_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:5];

        MTLSize reduceGrid = MTLSizeMake(n_head, 1, 1);
        [encoder dispatchThreadgroups:reduceGrid threadsPerThreadgroup:tgSize];
    }
}

static inline void dispatch_gqa_flash_decode_into_offset(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> q_buf,
    NSUInteger q_offset,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    id<MTLBuffer> out_buf,
    NSUInteger out_offset,
    torch::Tensor& partial_O,
    torch::Tensor& partial_m,
    torch::Tensor& partial_l,
    uint32_t max_seq_len,
    uint32_t active_len,
    uint32_t num_splits,
    uint32_t n_head,
    uint32_t n_kv_head,
    uint32_t head_dim) {
    float scale = 1.0f / std::sqrt(float(head_dim));
    uint32_t n_rep = n_head / n_kv_head;
    uint32_t rep_tiles = (n_rep + 7u) / 8u;

    [encoder setComputePipelineState:gqa_split_kv_pso];
    [encoder setBuffer:q_buf offset:q_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(k_cache) offset:k_cache.storage_offset() * k_cache.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(v_cache) offset:v_cache.storage_offset() * v_cache.element_size() atIndex:2];
    [encoder setBuffer:out_buf offset:out_offset atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:6];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&active_len length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&scale length:sizeof(float) atIndex:10];
    [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:13];

    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    MTLSize gridSize = MTLSizeMake(n_kv_head, num_splits, rep_tiles);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:gqa_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:out_buf offset:out_offset atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:5];

        MTLSize reduceGrid = MTLSizeMake(n_head, 1, 1);
        [encoder dispatchThreadgroups:reduceGrid threadsPerThreadgroup:tgSize];
    }
}

static inline void dispatch_gqa_tree_verify_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    torch::Tensor& out,
    torch::Tensor& partial_O,
    torch::Tensor& partial_m,
    torch::Tensor& partial_l,
    id<MTLBuffer> tree_mask_buf,
    NSUInteger mask_offset,
    uint32_t max_seq_len,
    uint32_t prefix_len,
    uint32_t write_pos,
    uint32_t T,
    uint32_t num_splits,
    uint32_t n_head,
    uint32_t n_kv_head,
    uint32_t head_dim,
    uint32_t q_stride) {
    float scale = 1.0f / std::sqrt(float(head_dim));
    uint32_t node_tiles = (T + 3u) / 4u;

    [encoder setComputePipelineState:gqa_tree_verify_split_kv_pso];
    [encoder setBuffer:getMTLBufferStorage(q) offset:q.storage_offset() * q.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(k_cache) offset:k_cache.storage_offset() * k_cache.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(v_cache) offset:v_cache.storage_offset() * v_cache.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:6];
    [encoder setBuffer:tree_mask_buf offset:mask_offset atIndex:7];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&prefix_len length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&write_pos length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&T length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&scale length:sizeof(float) atIndex:13];
    [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:14];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:15];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:16];
    [encoder setBytes:&q_stride length:sizeof(uint32_t) atIndex:17];

    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    MTLSize gridSize = MTLSizeMake(n_head, node_tiles, num_splits);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1u) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:gqa_tree_verify_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:6];

        MTLSize reduceGrid = MTLSizeMake(n_head, T, 1);
        [encoder dispatchThreadgroups:reduceGrid threadsPerThreadgroup:tgSize];
    }
}

static inline void dispatch_gqa_tree_verify_bytes(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    torch::Tensor& out,
    torch::Tensor& partial_O,
    torch::Tensor& partial_m,
    torch::Tensor& partial_l,
    const uint32_t* tree_mask,
    uint32_t max_seq_len,
    uint32_t prefix_len,
    uint32_t write_pos,
    uint32_t T,
    uint32_t num_splits,
    uint32_t n_head,
    uint32_t n_kv_head,
    uint32_t head_dim,
    uint32_t q_stride) {
    float scale = 1.0f / std::sqrt(float(head_dim));
    uint32_t node_tiles = (T + 3u) / 4u;

    [encoder setComputePipelineState:gqa_tree_verify_split_kv_pso];
    [encoder setBuffer:getMTLBufferStorage(q) offset:q.storage_offset() * q.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(k_cache) offset:k_cache.storage_offset() * k_cache.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(v_cache) offset:v_cache.storage_offset() * v_cache.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:6];
    [encoder setBytes:tree_mask length:T * sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&prefix_len length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&write_pos length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&T length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&scale length:sizeof(float) atIndex:13];
    [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:14];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:15];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:16];
    [encoder setBytes:&q_stride length:sizeof(uint32_t) atIndex:17];

    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    MTLSize gridSize = MTLSizeMake(n_head, node_tiles, num_splits);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1u) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:gqa_tree_verify_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:6];

        MTLSize reduceGrid = MTLSizeMake(n_head, T, 1);
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
    uint32_t head_dim = (uint32_t)k_cache.size(-1);
    uint32_t n_kv_head = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(1) : (uint32_t)k_cache.size(0);
    uint32_t max_seq_len = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(2) : (uint32_t)k_cache.size(1);
    uint32_t n_head = (uint32_t)(q.numel() / head_dim);
    uint32_t half_hd = head_dim >> 1;
    uint32_t n_splits = (num_splits > 0) ? (uint32_t)num_splits : ((active_len <= 64) ? 1 : ((active_len <= 256) ? 4 : 8));
    auto opts = q.options();
    auto partial_O = torch::empty({(int64_t)n_head, (int64_t)n_splits, (int64_t)half_hd, 2}, opts);
    auto partial_m = torch::empty({(int64_t)n_head, (int64_t)n_splits}, opts.dtype(torch::kFloat32));
    auto partial_l = torch::empty({(int64_t)n_head, (int64_t)n_splits}, opts.dtype(torch::kFloat32));

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
        n_splits,
        n_head,
        n_kv_head,
        head_dim);
    return out;
}

static inline void dispatch_gqa_flash_decode_fp8_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& q,
    const torch::Tensor& k_cache_fp8,
    const torch::Tensor& v_cache_fp8,
    const torch::Tensor& k_scales,
    const torch::Tensor& v_scales,
    torch::Tensor& out,
    torch::Tensor& partial_O,
    torch::Tensor& partial_m,
    torch::Tensor& partial_l,
    uint32_t max_seq_len,
    uint32_t active_len,
    uint32_t num_splits,
    uint32_t n_head,
    uint32_t n_kv_head,
    uint32_t head_dim) {
    float scale = 1.0f / std::sqrt(float(head_dim));
    uint32_t n_rep = n_head / n_kv_head;
    uint32_t rep_tiles = (n_rep + 7u) / 8u;

    [encoder setComputePipelineState:gqa_split_kv_fp8_pso];
    [encoder setBuffer:getMTLBufferStorage(q) offset:q.storage_offset() * q.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(k_cache_fp8) offset:k_cache_fp8.storage_offset() * k_cache_fp8.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(v_cache_fp8) offset:v_cache_fp8.storage_offset() * v_cache_fp8.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:6];
    [encoder setBuffer:getMTLBufferStorage(k_scales) offset:k_scales.storage_offset() * k_scales.element_size() atIndex:7];
    [encoder setBuffer:getMTLBufferStorage(v_scales) offset:v_scales.storage_offset() * v_scales.element_size() atIndex:8];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&active_len length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&scale length:sizeof(float) atIndex:12];
    [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:13];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:14];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:15];

    MTLSize gridSize = MTLSizeMake(n_kv_head, num_splits, rep_tiles);
    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:gqa_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:5];

        MTLSize reduceGrid = MTLSizeMake(n_head, 1, 1);
        [encoder dispatchThreadgroups:reduceGrid threadsPerThreadgroup:tgSize];
    }
}

static inline void dispatch_gqa_flash_decode_int4_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& q,
    const torch::Tensor& k_cache_int4,
    const torch::Tensor& v_cache_int4,
    const torch::Tensor& k_scales,
    const torch::Tensor& v_scales,
    torch::Tensor& out,
    torch::Tensor& partial_O,
    torch::Tensor& partial_m,
    torch::Tensor& partial_l,
    uint32_t max_seq_len,
    uint32_t active_len,
    uint32_t num_splits,
    uint32_t n_head,
    uint32_t n_kv_head,
    uint32_t head_dim) {
    float scale = 1.0f / std::sqrt(float(head_dim));
    uint32_t n_rep = n_head / n_kv_head;
    uint32_t rep_tiles = (n_rep + 7u) / 8u;

    [encoder setComputePipelineState:gqa_split_kv_int4_pso];
    [encoder setBuffer:getMTLBufferStorage(q) offset:q.storage_offset() * q.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(k_cache_int4) offset:k_cache_int4.storage_offset() * k_cache_int4.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(v_cache_int4) offset:v_cache_int4.storage_offset() * v_cache_int4.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:6];
    [encoder setBuffer:getMTLBufferStorage(k_scales) offset:k_scales.storage_offset() * k_scales.element_size() atIndex:7];
    [encoder setBuffer:getMTLBufferStorage(v_scales) offset:v_scales.storage_offset() * v_scales.element_size() atIndex:8];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&active_len length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&scale length:sizeof(float) atIndex:12];
    [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:13];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:14];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:15];

    MTLSize gridSize = MTLSizeMake(n_kv_head, num_splits, rep_tiles);
    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:gqa_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:5];

        MTLSize reduceGrid = MTLSizeMake(n_head, 1, 1);
        [encoder dispatchThreadgroups:reduceGrid threadsPerThreadgroup:tgSize];
    }
}

static inline void dispatch_gqa_flash_decode_fp8_into_offset(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> q_buf,
    NSUInteger q_offset,
    const torch::Tensor& k_cache_fp8,
    const torch::Tensor& v_cache_fp8,
    const torch::Tensor& k_scales,
    const torch::Tensor& v_scales,
    id<MTLBuffer> out_buf,
    NSUInteger out_offset,
    torch::Tensor& partial_O,
    torch::Tensor& partial_m,
    torch::Tensor& partial_l,
    uint32_t max_seq_len,
    uint32_t active_len,
    uint32_t num_splits,
    uint32_t n_head,
    uint32_t n_kv_head,
    uint32_t head_dim) {
    float scale = 1.0f / std::sqrt(float(head_dim));
    uint32_t n_rep = n_head / n_kv_head;
    uint32_t rep_tiles = (n_rep + 7u) / 8u;

    [encoder setComputePipelineState:gqa_split_kv_fp8_pso];
    [encoder setBuffer:q_buf offset:q_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(k_cache_fp8) offset:k_cache_fp8.storage_offset() * k_cache_fp8.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(v_cache_fp8) offset:v_cache_fp8.storage_offset() * v_cache_fp8.element_size() atIndex:2];
    [encoder setBuffer:out_buf offset:out_offset atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:6];
    [encoder setBuffer:getMTLBufferStorage(k_scales) offset:k_scales.storage_offset() * k_scales.element_size() atIndex:7];
    [encoder setBuffer:getMTLBufferStorage(v_scales) offset:v_scales.storage_offset() * v_scales.element_size() atIndex:8];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&active_len length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&scale length:sizeof(float) atIndex:12];
    [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:13];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:14];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:15];

    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    MTLSize gridSize = MTLSizeMake(n_kv_head, num_splits, rep_tiles);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:gqa_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:out_buf offset:out_offset atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:5];

        MTLSize reduceGrid = MTLSizeMake(n_head, 1, 1);
        [encoder dispatchThreadgroups:reduceGrid threadsPerThreadgroup:tgSize];
    }
}

static inline void dispatch_gqa_flash_decode_int4_into_offset(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> q_buf,
    NSUInteger q_offset,
    const torch::Tensor& k_cache_int4,
    const torch::Tensor& v_cache_int4,
    const torch::Tensor& k_scales,
    const torch::Tensor& v_scales,
    id<MTLBuffer> out_buf,
    NSUInteger out_offset,
    torch::Tensor& partial_O,
    torch::Tensor& partial_m,
    torch::Tensor& partial_l,
    uint32_t max_seq_len,
    uint32_t active_len,
    uint32_t num_splits,
    uint32_t n_head,
    uint32_t n_kv_head,
    uint32_t head_dim) {
    float scale = 1.0f / std::sqrt(float(head_dim));
    uint32_t n_rep = n_head / n_kv_head;
    uint32_t rep_tiles = (n_rep + 7u) / 8u;

    [encoder setComputePipelineState:gqa_split_kv_int4_pso];
    [encoder setBuffer:q_buf offset:q_offset atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(k_cache_int4) offset:k_cache_int4.storage_offset() * k_cache_int4.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(v_cache_int4) offset:v_cache_int4.storage_offset() * v_cache_int4.element_size() atIndex:2];
    [encoder setBuffer:out_buf offset:out_offset atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:6];
    [encoder setBuffer:getMTLBufferStorage(k_scales) offset:k_scales.storage_offset() * k_scales.element_size() atIndex:7];
    [encoder setBuffer:getMTLBufferStorage(v_scales) offset:v_scales.storage_offset() * v_scales.element_size() atIndex:8];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&active_len length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&scale length:sizeof(float) atIndex:12];
    [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:13];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:14];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:15];

    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    MTLSize gridSize = MTLSizeMake(n_kv_head, num_splits, rep_tiles);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:gqa_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:out_buf offset:out_offset atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:5];

        MTLSize reduceGrid = MTLSizeMake(n_head, 1, 1);
        [encoder dispatchThreadgroups:reduceGrid threadsPerThreadgroup:tgSize];
    }
}

static inline void dispatch_gqa_tree_verify_fp8_bytes(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& k_scales,
    const torch::Tensor& v_scales,
    torch::Tensor& out,
    torch::Tensor& partial_O,
    torch::Tensor& partial_m,
    torch::Tensor& partial_l,
    const uint32_t* tree_mask,
    uint32_t max_seq_len,
    uint32_t prefix_len,
    uint32_t write_pos,
    uint32_t T,
    uint32_t num_splits,
    uint32_t n_head,
    uint32_t n_kv_head,
    uint32_t head_dim,
    uint32_t q_stride) {
    float scale = 1.0f / std::sqrt(float(head_dim));
    uint32_t node_tiles = (T + 3u) / 4u;

    [encoder setComputePipelineState:gqa_tree_verify_split_kv_fp8_pso];
    [encoder setBuffer:getMTLBufferStorage(q) offset:q.storage_offset() * q.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(k_cache) offset:k_cache.storage_offset() * k_cache.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(v_cache) offset:v_cache.storage_offset() * v_cache.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:6];
    [encoder setBuffer:getMTLBufferStorage(k_scales) offset:k_scales.storage_offset() * k_scales.element_size() atIndex:7];
    [encoder setBuffer:getMTLBufferStorage(v_scales) offset:v_scales.storage_offset() * v_scales.element_size() atIndex:8];
    [encoder setBytes:tree_mask length:T * sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&prefix_len length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&write_pos length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&T length:sizeof(uint32_t) atIndex:13];
    [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:14];
    [encoder setBytes:&scale length:sizeof(float) atIndex:15];
    [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:16];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:17];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:18];
    [encoder setBytes:&q_stride length:sizeof(uint32_t) atIndex:19];

    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    MTLSize gridSize = MTLSizeMake(n_head, node_tiles, num_splits);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1u) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:gqa_tree_verify_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:6];

        MTLSize reduceGrid = MTLSizeMake(n_head, T, 1);
        [encoder dispatchThreadgroups:reduceGrid threadsPerThreadgroup:tgSize];
    }
}

static inline void dispatch_gqa_tree_verify_int4_bytes(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& k_scales,
    const torch::Tensor& v_scales,
    torch::Tensor& out,
    torch::Tensor& partial_O,
    torch::Tensor& partial_m,
    torch::Tensor& partial_l,
    const uint32_t* tree_mask,
    uint32_t max_seq_len,
    uint32_t prefix_len,
    uint32_t write_pos,
    uint32_t T,
    uint32_t num_splits,
    uint32_t n_head,
    uint32_t n_kv_head,
    uint32_t head_dim,
    uint32_t q_stride) {
    float scale = 1.0f / std::sqrt(float(head_dim));
    uint32_t node_tiles = (T + 3u) / 4u;

    [encoder setComputePipelineState:gqa_tree_verify_split_kv_int4_pso];
    [encoder setBuffer:getMTLBufferStorage(q) offset:q.storage_offset() * q.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(k_cache) offset:k_cache.storage_offset() * k_cache.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(v_cache) offset:v_cache.storage_offset() * v_cache.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
    [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:5];
    [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:6];
    [encoder setBuffer:getMTLBufferStorage(k_scales) offset:k_scales.storage_offset() * k_scales.element_size() atIndex:7];
    [encoder setBuffer:getMTLBufferStorage(v_scales) offset:v_scales.storage_offset() * v_scales.element_size() atIndex:8];
    [encoder setBytes:tree_mask length:T * sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&prefix_len length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&write_pos length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&T length:sizeof(uint32_t) atIndex:13];
    [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:14];
    [encoder setBytes:&scale length:sizeof(float) atIndex:15];
    [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:16];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:17];
    [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:18];
    [encoder setBytes:&q_stride length:sizeof(uint32_t) atIndex:19];

    MTLSize tgSize = MTLSizeMake(32, 1, 1);
    MTLSize gridSize = MTLSizeMake(n_head, node_tiles, num_splits);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

    if (num_splits > 1u) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:gqa_tree_verify_reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_O) offset:partial_O.storage_offset() * partial_O.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_m) offset:partial_m.storage_offset() * partial_m.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_l) offset:partial_l.storage_offset() * partial_l.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
        [encoder setBytes:&num_splits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&n_head length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:6];

        MTLSize reduceGrid = MTLSizeMake(n_head, T, 1);
        [encoder dispatchThreadgroups:reduceGrid threadsPerThreadgroup:tgSize];
    }
}

torch::Tensor gqa_flash_decode_fp8_mps(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    int64_t active_len,
    int64_t num_splits) {
    auto out = torch::empty_like(q);
    uint32_t head_dim = (uint32_t)q.size(-1);
    uint32_t n_kv_head = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(1) : (uint32_t)k_cache.size(0);
    uint32_t max_seq_len = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(2) : (uint32_t)k_cache.size(1);
    uint32_t n_head = (uint32_t)(q.numel() / head_dim);
    uint32_t half_hd = head_dim >> 1;
    uint32_t n_splits = (num_splits > 0) ? (uint32_t)num_splits : ((active_len <= 64) ? 1 : ((active_len <= 256) ? 4 : 8));
    auto opts = q.options();
    auto partial_O = torch::empty({(int64_t)n_head, (int64_t)n_splits, (int64_t)half_hd, 2}, opts);
    auto partial_m = torch::empty({(int64_t)n_head, (int64_t)n_splits}, opts.dtype(torch::kFloat32));
    auto partial_l = torch::empty({(int64_t)n_head, (int64_t)n_splits}, opts.dtype(torch::kFloat32));

    auto stream = at::mps::getCurrentMPSStream();
    dispatch_gqa_flash_decode_fp8_into(
        stream->commandEncoder(),
        q,
        k_cache,
        v_cache,
        k_scales,
        v_scales,
        out,
        partial_O,
        partial_m,
        partial_l,
        max_seq_len,
        (uint32_t)active_len,
        n_splits,
        n_head,
        n_kv_head,
        head_dim);
    return out;
}

torch::Tensor gqa_flash_decode_int4_mps(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    int64_t active_len,
    int64_t num_splits) {
    auto out = torch::empty_like(q);
    uint32_t head_dim = (uint32_t)q.size(-1);
    uint32_t n_kv_head = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(1) : (uint32_t)k_cache.size(0);
    uint32_t max_seq_len = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(2) : (uint32_t)k_cache.size(1);
    uint32_t n_head = (uint32_t)(q.numel() / head_dim);
    uint32_t half_hd = head_dim >> 1;
    uint32_t n_splits = (num_splits > 0) ? (uint32_t)num_splits : ((active_len <= 64) ? 1 : ((active_len <= 256) ? 4 : 8));
    auto opts = q.options();
    auto partial_O = torch::empty({(int64_t)n_head, (int64_t)n_splits, (int64_t)half_hd, 2}, opts);
    auto partial_m = torch::empty({(int64_t)n_head, (int64_t)n_splits}, opts.dtype(torch::kFloat32));
    auto partial_l = torch::empty({(int64_t)n_head, (int64_t)n_splits}, opts.dtype(torch::kFloat32));

    auto stream = at::mps::getCurrentMPSStream();
    dispatch_gqa_flash_decode_int4_into(
        stream->commandEncoder(),
        q,
        k_cache,
        v_cache,
        k_scales,
        v_scales,
        out,
        partial_O,
        partial_m,
        partial_l,
        max_seq_len,
        (uint32_t)active_len,
        n_splits,
        n_head,
        n_kv_head,
        head_dim);
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

    uint32_t num_tg = (N + 15) / 16;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
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

    uint32_t num_tg = (N + 31) / 32;
    uint32_t m_tiles = (M + 7) / 8;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, m_tiles, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static inline void dispatch_qmv_ternary_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales,
    torch::Tensor& y,
    const torch::Tensor& bias = torch::Tensor()) {
    uint32_t K = x.size(-1);
    uint32_t N = w_packed.size(0);
    uint32_t has_bias = bias.defined() ? 1u : 0u;

    [encoder setComputePipelineState:qmv_ternary_half_pso];
    [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() * w_packed.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales) offset:scales.storage_offset() * scales.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:5];
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(bias) offset:bias.storage_offset() * bias.element_size() atIndex:6];
    } else {
        [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:6];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:7];

    uint32_t num_tg = (N + 15) / 16;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static inline void dispatch_qmm_ternary_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales,
    torch::Tensor& y,
    const torch::Tensor& bias = torch::Tensor()) {
    uint32_t K = x.size(-1);
    uint32_t M = x.numel() / K;
    uint32_t N = w_packed.size(0);
    uint32_t has_bias = bias.defined() ? 1u : 0u;

    if (M == 1) {
        dispatch_qmv_ternary_into(encoder, x, w_packed, scales, y, bias);
        return;
    }

    [encoder setComputePipelineState:qmm_ternary_half_pso];
    [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() * w_packed.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales) offset:scales.storage_offset() * scales.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
    [encoder setBytes:&M length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:6];
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(bias) offset:bias.storage_offset() * bias.element_size() atIndex:7];
    } else {
        [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:7];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:8];

    uint32_t num_tg = (N + 31) / 32;
    uint32_t m_tiles = (M + 7) / 8;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, m_tiles, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static inline void dispatch_qmv_q2_k_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales_biases,
    torch::Tensor& y,
    const torch::Tensor& bias = torch::Tensor()) {
    uint32_t K = x.size(-1);
    uint32_t N = w_packed.size(0);
    uint32_t has_bias = bias.defined() ? 1u : 0u;

    [encoder setComputePipelineState:qmv_q2_k_half_pso];
    [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() * w_packed.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales_biases) offset:scales_biases.storage_offset() * scales_biases.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:5];
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(bias) offset:bias.storage_offset() * bias.element_size() atIndex:6];
    } else {
        [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:6];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:7];

    uint32_t num_tg = (N + 15) / 16;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static inline void dispatch_qmm_q2_k_into(
    id<MTLComputeCommandEncoder> encoder,
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales_biases,
    torch::Tensor& y,
    const torch::Tensor& bias = torch::Tensor()) {
    uint32_t K = x.size(-1);
    uint32_t M = x.numel() / K;
    uint32_t N = w_packed.size(0);
    uint32_t has_bias = bias.defined() ? 1u : 0u;

    if (M == 1) {
        dispatch_qmv_q2_k_into(encoder, x, w_packed, scales_biases, y, bias);
        return;
    }

    [encoder setComputePipelineState:qmm_q2_k_half_pso];
    [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() * w_packed.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales_biases) offset:scales_biases.storage_offset() * scales_biases.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:3];
    [encoder setBytes:&M length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:6];
    if (has_bias) {
        [encoder setBuffer:getMTLBufferStorage(bias) offset:bias.storage_offset() * bias.element_size() atIndex:7];
    } else {
        [encoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:7];
    }
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:8];

    uint32_t num_tg = (N + 31) / 32;
    uint32_t m_tiles = (M + 7) / 8;
    [encoder dispatchThreadgroups:MTLSizeMake(num_tg, m_tiles, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

struct DecodeGraphState {
    torch::Tensor wte_weight;
    torch::Tensor lm_head_weight;
    torch::Tensor freqs_real;
    std::vector<torch::Tensor> ln1_weights;
    std::vector<torch::Tensor> c_attn_weights;
    std::vector<torch::Tensor> c_attn_biases;
    std::vector<torch::Tensor> c_proj_weights;
    std::vector<torch::Tensor> ln2_weights;
    std::vector<torch::Tensor> gate_up_weights;
    std::vector<torch::Tensor> w_down_weights;
    torch::Tensor lnf_weight;
    std::vector<torch::Tensor> k_caches;
    std::vector<torch::Tensor> v_caches;
    torch::Tensor empty_bias;
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
    int64_t qkv_dim = 0;
    int64_t vocab_size = 0;
    uint32_t q_pairs = 0;
    uint32_t kv_pairs = 0;
    uint32_t half_head_dim = 0;
    uint32_t max_seq_len = 0;
    bool enable_gqa = false;
    float eps = 1e-6f;
    bool initialized = false;
    id<MTLBuffer> shared_idx_buf = nil;
    id<MTLBuffer> shared_argmax_buf = nil;

    inline const torch::Tensor& attn_bias(int64_t i) const {
        if ((size_t)i < c_attn_biases.size() && c_attn_biases[i].defined()) {
            return c_attn_biases[i];
        }
        return empty_bias;
    }

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
        float epsilon,
        torch::Tensor lm_head_w = torch::Tensor(),
        std::vector<torch::Tensor> c_attn_bs = {})
        : wte_weight(wte_w.detach()),
          lm_head_weight(lm_head_w.defined() ? lm_head_w.detach() : wte_w.detach()),
          freqs_real(freqs_r),
          ln1_weights(std::move(ln1_ws)),
          c_attn_weights(std::move(c_attn_ws)),
          c_attn_biases(std::move(c_attn_bs)),
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
          qkv_dim(n_h * hd + 2 * n_kvh * hd),
          vocab_size(this->lm_head_weight.size(0)),
          q_pairs((uint32_t)((n_h * hd) / 2)),
          kv_pairs((uint32_t)((n_kvh * hd) / 2)),
          half_head_dim((uint32_t)(hd / 2)),
          max_seq_len((uint32_t)this->k_caches[0].size(2)),
          enable_gqa(n_h != n_kvh),
          eps(epsilon),
          initialized(true) {
        auto opts = wte_weight.options();
        int64_t d_model = q_dim;
        int64_t gu_dim = gate_up_weights[0].size(0);
        int64_t hidden_dim = gu_dim / 2;

        x_buf = torch::empty({1, 1, d_model}, opts);
        x_buf_2d = x_buf.view({1, d_model});
        norm_buf = torch::empty({1, 1, d_model}, opts);
        qkv_buf = torch::empty({1, 1, qkv_dim}, opts);
        q_view = qkv_buf.slice(2, 0, d_model).view({1, n_head, 1, head_dim});
        attn_out_buf = torch::empty({1, 1, d_model}, opts);
        attn_y_buf = torch::empty({1, 1, d_model}, opts);
        partial_O_buf = torch::empty({32 * n_head, 16, (int64_t)half_head_dim, 2}, opts);
        partial_m_buf = torch::empty({32 * n_head, 16}, opts.dtype(torch::kFloat32));
        partial_l_buf = torch::empty({32 * n_head, 16}, opts.dtype(torch::kFloat32));
        gate_up_buf = torch::empty({1, 1, gu_dim}, opts);
        swiglu_buf = torch::empty({1, 1, hidden_dim}, opts);
        mlp_out_buf = torch::empty({1, 1, d_model}, opts);
        logits_buf = torch::empty({1, 1, vocab_size}, opts);
        x_buf_multi = torch::empty({1, 32, d_model}, opts);
        norm_buf_multi = torch::empty({1, 32, d_model}, opts);
        qkv_buf_multi = torch::empty({1, 32, qkv_dim}, opts);
        attn_y_buf_multi = torch::empty({1, 32, d_model}, opts);
        attn_out_buf_multi = torch::empty({1, 32, d_model}, opts);
        gate_up_buf_multi = torch::empty({1, 32, gu_dim}, opts);
        swiglu_buf_multi = torch::empty({1, 32, hidden_dim}, opts);
        mlp_out_buf_multi = torch::empty({1, 32, d_model}, opts);
        logits_buf_multi = torch::empty({1, 32, vocab_size}, opts);
        shared_idx_buf = [mtl_device newBufferWithLength:64 * sizeof(int64_t) options:MTLResourceStorageModeShared];
        shared_argmax_buf = [mtl_device newBufferWithLength:64 * sizeof(int64_t) options:MTLResourceStorageModeShared];
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
            const auto& b_i = attn_bias(i);
            if (b_i.defined() && qkv_buf.scalar_type() != at::kHalf) {
                at::linear_out(qkv_buf, norm_buf, c_attn_weights[i], b_i);
            } else {
                at::linear_out(qkv_buf, norm_buf, c_attn_weights[i]);
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
                (uint32_t)write_pos,
                b_i);

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
                n_splits,
                (uint32_t)n_head,
                (uint32_t)n_kv_head,
                (uint32_t)head_dim);
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

        at::linear_out(logits_buf, norm_buf, lm_head_weight);
        return logits_buf;
    }

    torch::Tensor verify_tree_step(
        torch::Tensor idx,
        int64_t start_pos,
        int64_t write_pos,
        int64_t prefix_len,
        const std::vector<int32_t>& tree_depths,
        const std::vector<uint32_t>& tree_mask) {
        at::NoGradGuard no_grad;
        int64_t T = idx.numel();
        TORCH_CHECK(T >= 1 && T <= 32, "verify_tree_step supports T in [1, 32], got ", T);
        TORCH_CHECK((int64_t)tree_depths.size() == T && (int64_t)tree_mask.size() == T,
                    "tree_depths and tree_mask must match T");

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
        dispatch_rmsnorm_infer_into(stream->commandEncoder(), x_cur, ln1_weights[0], norm_cur, eps);

        uint32_t n_splits = (prefix_len <= 64) ? 1u : ((prefix_len <= 256) ? 4u : 8u);

        for (int64_t i = 0; i < n_layer; ++i) {
            at::linear_out(qkv_cur, norm_cur, c_attn_weights[i]);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            dispatch_rope_tree_kv_cache_write_bytes(
                encoder,
                qkv_cur,
                freqs_real,
                k_caches[i],
                v_caches[i],
                tree_depths.data(),
                q_pairs,
                kv_pairs,
                half_head_dim,
                max_seq_len,
                (uint32_t)start_pos,
                (uint32_t)write_pos,
                (uint32_t)T,
                attn_bias(i));

            dispatch_gqa_tree_verify_bytes(
                encoder,
                qkv_cur,
                k_caches[i],
                v_caches[i],
                attn_y_cur,
                partial_O_buf,
                partial_m_buf,
                partial_l_buf,
                tree_mask.data(),
                max_seq_len,
                (uint32_t)prefix_len,
                (uint32_t)write_pos,
                (uint32_t)T,
                n_splits,
                (uint32_t)n_head,
                (uint32_t)n_kv_head,
                (uint32_t)head_dim,
                (uint32_t)qkv_dim);

            at::linear_out(attn_out_cur, attn_y_cur, c_proj_weights[i]);
            dispatch_add_rmsnorm_infer_into(
                stream->commandEncoder(), x_cur, attn_out_cur, ln2_weights[i], norm_cur, eps);

            at::linear_out(gate_up_cur, norm_cur, gate_up_weights[i]);
            dispatch_swiglu_gate_up_into(stream->commandEncoder(), gate_up_cur, swiglu_cur);
            at::linear_out(mlp_out_cur, swiglu_cur, w_down_weights[i]);

            const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
            dispatch_add_rmsnorm_infer_into(
                stream->commandEncoder(), x_cur, mlp_out_cur, next_ln_w, norm_cur, eps);
        }

        at::linear_out(logits_cur, norm_cur, lm_head_weight);
        return logits_cur;
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
        TORCH_CHECK(T <= 32, "verify_step supports up to 32 tokens, got ", T);

        if (x_buf_multi.scalar_type() == at::kHalf &&
            write_pos + T <= (int64_t)max_seq_len &&
            active_len == write_pos + T) {
            std::vector<int32_t> depths(T);
            std::vector<uint32_t> masks(T);
            for (int64_t t = 0; t < T; ++t) {
                depths[t] = (int32_t)t;
                masks[t] = (t == 31) ? 0xFFFFFFFFu : ((1u << (uint32_t)(t + 1)) - 1u);
            }
            return verify_tree_step(idx, start_pos, write_pos, write_pos, depths, masks);
        }

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
            const auto& b_i = attn_bias(i);
            if (b_i.defined() && qkv_cur.scalar_type() != at::kHalf) {
                at::linear_out(qkv_cur, norm_cur, c_attn_weights[i], b_i);
            } else {
                at::linear_out(qkv_cur, norm_cur, c_attn_weights[i]);
            }

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
                    (start_pos + t) % freqs_real.size(0),
                    (uint32_t)((write_pos + t) % max_seq_len),
                    b_i);
            }

            for (int64_t t = 0; t < T; ++t) {
                auto qkv_t = qkv_cur.narrow(1, t, 1);
                auto attn_y_t = attn_y_cur.narrow(1, t, 1);
                uint32_t cur_active = (uint32_t)((active_len == write_pos + T) ? (write_pos + t + 1) : max_seq_len);
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
                    n_splits,
                    (uint32_t)n_head,
                    (uint32_t)n_kv_head,
                    (uint32_t)head_dim);
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

        at::linear_out(logits_cur, norm_cur, lm_head_weight);
        return logits_cur;
    }

    void encode_step_internal(int64_t start_pos, int64_t write_pos, int64_t active_len) {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        dispatch_embed_multi(encoder, shared_idx_buf, wte_weight, x_buf, (uint32_t)q_dim, 1);
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
                (uint32_t)write_pos,
                attn_bias(i));

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
                n_splits,
                (uint32_t)n_head,
                (uint32_t)n_kv_head,
                (uint32_t)head_dim);
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

        at::linear_out(logits_buf, norm_buf, lm_head_weight);
        dispatch_argmax_multi(stream->commandEncoder(), logits_buf, shared_argmax_buf, (uint32_t)vocab_size, 1);
    }

    void encode_verify_step_internal(int64_t T, int64_t start_pos, int64_t write_pos) {
        auto x_cur = x_buf_multi.narrow(1, 0, T);
        auto norm_cur = norm_buf_multi.narrow(1, 0, T);
        auto qkv_cur = qkv_buf_multi.narrow(1, 0, T);
        auto attn_y_cur = attn_y_buf_multi.narrow(1, 0, T);
        auto attn_out_cur = attn_out_buf_multi.narrow(1, 0, T);
        auto gate_up_cur = gate_up_buf_multi.narrow(1, 0, T);
        auto swiglu_cur = swiglu_buf_multi.narrow(1, 0, T);
        auto mlp_out_cur = mlp_out_buf_multi.narrow(1, 0, T);
        auto logits_cur = logits_buf_multi.narrow(1, 0, T);

        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        dispatch_embed_multi(encoder, shared_idx_buf, wte_weight, x_cur, (uint32_t)q_dim, (uint32_t)T);
        dispatch_rmsnorm_infer_into(encoder, x_cur, ln1_weights[0], norm_cur, eps);

        id<MTLBuffer> qkv_buf_storage = getMTLBufferStorage(qkv_cur);
        id<MTLBuffer> attn_y_buf_storage = getMTLBufferStorage(attn_y_cur);
        NSUInteger elem_sz = qkv_cur.element_size();

        for (int64_t i = 0; i < n_layer; ++i) {
            at::linear_out(qkv_cur, norm_cur, c_attn_weights[i]);
            const auto& b_i = attn_bias(i);

            encoder = stream->commandEncoder();
            if (write_pos + T <= (int64_t)max_seq_len) {
                std::vector<int32_t> depths(T);
                std::vector<uint32_t> masks(T);
                for (int64_t t = 0; t < T; ++t) {
                    depths[t] = (int32_t)t;
                    masks[t] = (t == 31) ? 0xFFFFFFFFu : ((1u << (uint32_t)(t + 1)) - 1u);
                }
                uint32_t n_splits = (write_pos <= 64) ? 1u : ((write_pos <= 256) ? 4u : 8u);
                dispatch_rope_tree_kv_cache_write_bytes(
                    encoder, qkv_cur, freqs_real, k_caches[i], v_caches[i], depths.data(),
                    q_pairs, kv_pairs, half_head_dim, max_seq_len, (uint32_t)start_pos,
                    (uint32_t)write_pos, (uint32_t)T, b_i);
                dispatch_gqa_tree_verify_bytes(
                    encoder, qkv_cur, k_caches[i], v_caches[i], attn_y_cur,
                    partial_O_buf, partial_m_buf, partial_l_buf, masks.data(),
                    max_seq_len, (uint32_t)write_pos, (uint32_t)write_pos, (uint32_t)T,
                    n_splits, (uint32_t)n_head, (uint32_t)n_kv_head, (uint32_t)head_dim,
                    (uint32_t)qkv_dim);
            } else {
                for (int64_t t = 0; t < T; ++t) {
                    NSUInteger qkv_t_offset = (qkv_cur.storage_offset() + t * qkv_dim) * elem_sz;
                    dispatch_rope_kv_cache_write_offset(
                        encoder,
                        qkv_buf_storage,
                        qkv_t_offset,
                        freqs_real,
                        k_caches[i],
                        v_caches[i],
                        q_pairs,
                        kv_pairs,
                        half_head_dim,
                        max_seq_len,
                        (start_pos + t) % freqs_real.size(0),
                        (uint32_t)((write_pos + t) % max_seq_len),
                        b_i);
                }

                for (int64_t t = 0; t < T; ++t) {
                    NSUInteger qkv_t_offset = (qkv_cur.storage_offset() + t * qkv_dim) * elem_sz;
                    NSUInteger attn_y_t_offset = (attn_y_cur.storage_offset() + t * q_dim) * elem_sz;
                    uint32_t cur_active = max_seq_len;
                    uint32_t n_splits = (cur_active <= 64) ? 1 : ((cur_active <= 256) ? 4 : 8);

                    dispatch_gqa_flash_decode_into_offset(
                        encoder,
                        qkv_buf_storage,
                        qkv_t_offset,
                        k_caches[i],
                        v_caches[i],
                        attn_y_buf_storage,
                        attn_y_t_offset,
                        partial_O_buf,
                        partial_m_buf,
                        partial_l_buf,
                        max_seq_len,
                        cur_active,
                        n_splits,
                        (uint32_t)n_head,
                        (uint32_t)n_kv_head,
                        (uint32_t)head_dim);
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
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

        at::linear_out(logits_cur, norm_cur, lm_head_weight);
        dispatch_argmax_multi(stream->commandEncoder(), logits_cur, shared_argmax_buf, (uint32_t)vocab_size, (uint32_t)T);
    }

    std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t, int64_t, double>
    speculative_decode(
        const std::vector<int64_t>& initial_tokens,
        int64_t max_new_tokens,
        int64_t ngram_size,
        int64_t max_spec_tokens,
        int64_t start_pos) {
        at::NoGradGuard no_grad;
        std::vector<int64_t> generated = initial_tokens;
        int64_t target_total = (int64_t)initial_tokens.size() + max_new_tokens;
        int64_t checkpoint_seen = start_pos;
        int64_t model_forward_calls = 0;
        int64_t total_drafted = 0;
        int64_t total_accepted = 0;

        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
        auto t0 = std::chrono::high_resolution_clock::now();

        long* idx_ptr = (long*)[shared_idx_buf contents];
        long* preds = (long*)[shared_argmax_buf contents];

        while ((int64_t)generated.size() < target_total) {
            std::vector<int64_t> candidates = find_candidate_ngram_cpp(generated, ngram_size, max_spec_tokens);
            int64_t k_cand = (int64_t)candidates.size();

            if (k_cand == 0) {
                int64_t rope_pos = checkpoint_seen % freqs_real.size(0);
                int64_t write_pos = checkpoint_seen % max_seq_len;
                int64_t cur_active = std::min(checkpoint_seen + 1, (int64_t)max_seq_len);

                idx_ptr[0] = generated.back();

                encode_step_internal(rope_pos, write_pos, cur_active);
                stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

                generated.push_back(preds[0]);
                checkpoint_seen += 1;
                model_forward_calls += 1;
            } else {
                int64_t T = 1 + k_cand;
                int64_t rope_pos = checkpoint_seen % freqs_real.size(0);
                int64_t write_pos = checkpoint_seen % max_seq_len;

                idx_ptr[0] = generated.back();
                for (int64_t c = 0; c < k_cand; ++c) {
                    idx_ptr[1 + c] = candidates[c];
                }

                encode_verify_step_internal(T, rope_pos, write_pos);
                stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

                total_drafted += k_cand;
                model_forward_calls += 1;

                int64_t accepted = 0;
                bool all_matched = true;

                for (int64_t j = 0; j < k_cand; ++j) {
                    int64_t pred_tok = preds[j];
                    if (pred_tok == candidates[j]) {
                        generated.push_back(pred_tok);
                        accepted += 1;
                        if ((int64_t)generated.size() >= target_total) {
                            all_matched = false;
                            break;
                        }
                    } else {
                        generated.push_back(pred_tok);
                        accepted += 1;
                        all_matched = false;
                        break;
                    }
                }

                if (all_matched && (int64_t)generated.size() < target_total) {
                    int64_t bonus_tok = preds[k_cand];
                    generated.push_back(bonus_tok);
                    accepted += 1;
                }

                total_accepted += std::max((int64_t)0, accepted - 1);
                checkpoint_seen += accepted;
            }
        }

        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();

        return std::make_tuple(generated, checkpoint_seen, model_forward_calls, total_drafted, total_accepted, elapsed_seconds);
    }

    torch::Tensor get_last_norm(int64_t T = 1) {
        if (T == 1) {
            return norm_buf.clone();
        }
        return norm_buf_multi.narrow(1, 0, T).clone();
    }
};

struct QuantizedDecodeGraphState {
    torch::Tensor wte_embed;
    torch::Tensor wte_packed;
    torch::Tensor wte_sb;
    bool use_q8_wte = false;
    torch::Tensor lm_head_packed;
    torch::Tensor lm_head_sb;
    torch::Tensor freqs_real;
    std::vector<torch::Tensor> ln1_weights;
    std::vector<torch::Tensor> c_attn_packed;
    std::vector<torch::Tensor> c_attn_sb;
    std::vector<torch::Tensor> c_attn_biases;
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
    torch::Tensor empty_bias;
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
    int64_t qkv_dim = 0;
    int64_t vocab_size = 0;
    uint32_t q_pairs = 0;
    uint32_t kv_pairs = 0;
    uint32_t half_head_dim = 0;
    uint32_t max_seq_len = 0;
    bool enable_gqa = false;
    bool is_q4 = true;
    bool is_marlin = false;
    float eps = 1e-6f;
    bool initialized = false;
    id<MTLBuffer> shared_idx_buf = nil;
    id<MTLBuffer> shared_argmax_buf = nil;

    inline const torch::Tensor& attn_bias(int64_t i) const {
        if ((size_t)i < c_attn_biases.size() && c_attn_biases[i].defined()) {
            return c_attn_biases[i];
        }
        return empty_bias;
    }

    inline void dispatch_qmv_into(
        id<MTLComputeCommandEncoder> encoder,
        const torch::Tensor& x,
        const torch::Tensor& w_p,
        const torch::Tensor& w_sb,
        torch::Tensor& y) const {
        if (w_p.dim() == 4) {
            dispatch_qmv_q4_marlin_into(encoder, x, w_p, w_sb, y);
        } else if (w_p.size(1) == (x.size(-1) >> 2)) {
            dispatch_qmv_q8_g64_into(encoder, x, w_p, w_sb, y);
        } else {
            dispatch_qmv_q4_g64_into(encoder, x, w_p, w_sb, y);
        }
    }

    inline void dispatch_qmm_into(
        id<MTLComputeCommandEncoder> encoder,
        const torch::Tensor& x,
        const torch::Tensor& w_p,
        const torch::Tensor& w_sb,
        torch::Tensor& y) const {
        if (w_p.dim() == 4) {
            dispatch_qmm_q4_marlin_into(encoder, x, w_p, w_sb, y);
        } else if (w_p.size(1) == (x.size(-1) >> 2)) {
            dispatch_qmm_q8_g64_into(encoder, x, w_p, w_sb, y);
        } else {
            dispatch_qmm_q4_g64_into(encoder, x, w_p, w_sb, y);
        }
    }

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
        float epsilon,
        std::vector<torch::Tensor> c_attn_bs = {},
        int64_t vocab_sz = -1,
        torch::Tensor wte_p = torch::Tensor(),
        torch::Tensor wte_s = torch::Tensor())
        : wte_embed(wte_emb.detach()),
          wte_packed(wte_p.defined() ? wte_p : torch::Tensor()),
          wte_sb(wte_s.defined() ? wte_s : torch::Tensor()),
          use_q8_wte(wte_p.defined() && wte_p.numel() > 0 && wte_s.defined() && wte_s.numel() > 0),
          lm_head_packed(lmh_packed),
          lm_head_sb(lmh_sb),
          freqs_real(freqs_r),
          ln1_weights(std::move(ln1_ws)),
          c_attn_packed(std::move(c_attn_p)),
          c_attn_sb(std::move(c_attn_s)),
          c_attn_biases(std::move(c_attn_bs)),
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
          qkv_dim(n_h * hd + 2 * n_kvh * hd),
          q_pairs((uint32_t)((n_h * hd) / 2)),
          kv_pairs((uint32_t)((n_kvh * hd) / 2)),
          half_head_dim((uint32_t)(hd / 2)),
          max_seq_len((uint32_t)this->k_caches[0].size(2)),
          enable_gqa(n_h != n_kvh),
          is_q4(use_q4),
          is_marlin(use_q4 && (this->c_attn_packed[0].dim() == 4)),
          eps(epsilon),
          initialized(true) {
        if (wte_p.defined() && wte_p.numel() > 0 && wte_s.defined() && wte_s.numel() > 0 && wte_p.dim() == 2 && wte_p.dtype() == torch::kInt32) {
            wte_packed = wte_p;
            wte_sb = wte_s;
            use_q8_wte = true;
        } else if (wte_embed.defined() && wte_embed.dtype() == torch::kInt32 && wte_embed.dim() == 2 && lm_head_sb.defined() && lm_head_sb.numel() > 0 && lm_head_sb.dim() == 3) {
            wte_packed = wte_embed;
            wte_sb = lm_head_sb;
            use_q8_wte = true;
        }
        auto opts = (wte_embed.defined() && wte_embed.dtype() == torch::kFloat16 && wte_embed.numel() > 0)
            ? wte_embed.options()
            : torch::TensorOptions().device(lm_head_packed.device()).dtype(torch::kFloat16);
        int64_t d_model = q_dim;
        int64_t hidden_dim = this->w_down_sb[0].size(1) * 64;
        int64_t gu_dim = hidden_dim * 2;
        vocab_size = (vocab_sz > 0)
            ? vocab_sz
            : ((lm_head_packed.dim() == 4) ? (wte_packed.defined() ? wte_packed.size(0) : (wte_embed.defined() ? wte_embed.size(0) : lm_head_packed.size(0) * 16)) : lm_head_packed.size(0));

        x_buf = torch::empty({1, 1, d_model}, opts);
        x_buf_2d = x_buf.view({1, d_model});
        norm_buf = torch::empty({1, 1, d_model}, opts);
        qkv_buf = torch::empty({1, 1, qkv_dim}, opts);
        q_view = qkv_buf.slice(2, 0, d_model).view({1, n_head, 1, head_dim});
        attn_out_buf = torch::empty({1, 1, d_model}, opts);
        attn_y_buf = torch::empty({1, 1, d_model}, opts);
        partial_O_buf = torch::empty({32 * n_head, 16, (int64_t)half_head_dim, 2}, opts);
        partial_m_buf = torch::empty({32 * n_head, 16}, opts.dtype(torch::kFloat32));
        partial_l_buf = torch::empty({32 * n_head, 16}, opts.dtype(torch::kFloat32));
        gate_up_buf = torch::empty({1, 1, gu_dim}, opts);
        swiglu_buf = torch::empty({1, 1, hidden_dim}, opts);
        mlp_out_buf = torch::empty({1, 1, d_model}, opts);
        logits_buf = torch::empty({1, 1, vocab_size}, opts);
        x_buf_multi = torch::empty({1, 32, d_model}, opts);
        norm_buf_multi = torch::empty({1, 32, d_model}, opts);
        qkv_buf_multi = torch::empty({1, 32, qkv_dim}, opts);
        attn_y_buf_multi = torch::empty({1, 32, d_model}, opts);
        attn_out_buf_multi = torch::empty({1, 32, d_model}, opts);
        gate_up_buf_multi = torch::empty({1, 32, gu_dim}, opts);
        swiglu_buf_multi = torch::empty({1, 32, hidden_dim}, opts);
        mlp_out_buf_multi = torch::empty({1, 32, d_model}, opts);
        logits_buf_multi = torch::empty({1, 32, vocab_size}, opts);
        shared_idx_buf = [mtl_device newBufferWithLength:64 * sizeof(int64_t) options:MTLResourceStorageModeShared];
        shared_argmax_buf = [mtl_device newBufferWithLength:64 * sizeof(int64_t) options:MTLResourceStorageModeShared];
    }

    torch::Tensor step(
        torch::Tensor idx,
        int64_t start_pos,
        int64_t write_pos,
        int64_t active_len) {
        at::NoGradGuard no_grad;
        if (!use_q8_wte) {
            at::index_select_out(x_buf_2d, wte_embed, 0, idx.view({-1}));
        }
        auto idx_mps = idx.to(torch::kMPS).to(torch::kInt64).contiguous();
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        if (use_q8_wte) {
            dispatch_embed_q8_g64_gather_multi(encoder, idx_mps, wte_packed, wte_sb, x_buf, (uint32_t)q_dim, 1);
        }
        dispatch_rmsnorm_infer_into(encoder, x_buf, ln1_weights[0], norm_buf, eps);

        uint32_t n_splits = (active_len <= 64) ? 1 : ((active_len <= 256) ? 4 : 8);

        for (int64_t i = 0; i < n_layer; ++i) {
            dispatch_qmv_into(encoder, norm_buf, c_attn_packed[i], c_attn_sb[i], qkv_buf);

            dispatch_rope_kv_cache_write(
                encoder,
                qkv_buf,
                freqs_real,
                k_caches[i],
                v_caches[i],
                q_pairs,
                kv_pairs,
                half_head_dim,
                max_seq_len,
                start_pos,
                (uint32_t)write_pos,
                attn_bias(i));

            dispatch_gqa_flash_decode_into(
                encoder,
                qkv_buf,
                k_caches[i],
                v_caches[i],
                attn_y_buf,
                partial_O_buf,
                partial_m_buf,
                partial_l_buf,
                max_seq_len,
                (uint32_t)active_len,
                n_splits,
                (uint32_t)n_head,
                (uint32_t)n_kv_head,
                (uint32_t)head_dim);

            dispatch_qmv_into(encoder, attn_y_buf, c_proj_packed[i], c_proj_sb[i], attn_out_buf);

            dispatch_add_rmsnorm_infer_into(
                encoder, x_buf, attn_out_buf, ln2_weights[i], norm_buf, eps);

            dispatch_qmv_into(encoder, norm_buf, gate_up_packed[i], gate_up_sb[i], gate_up_buf);
            dispatch_swiglu_gate_up_into(encoder, gate_up_buf, swiglu_buf);
            dispatch_qmv_into(encoder, swiglu_buf, w_down_packed[i], w_down_sb[i], mlp_out_buf);

            const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
            dispatch_add_rmsnorm_infer_into(
                encoder, x_buf, mlp_out_buf, next_ln_w, norm_buf, eps);
        }

        dispatch_qmv_into(encoder, norm_buf, lm_head_packed, lm_head_sb, logits_buf);
        return logits_buf;
    }

    torch::Tensor verify_tree_step(
        torch::Tensor idx,
        int64_t start_pos,
        int64_t write_pos,
        int64_t prefix_len,
        const std::vector<int32_t>& tree_depths,
        const std::vector<uint32_t>& tree_mask) {
        at::NoGradGuard no_grad;
        int64_t T = idx.numel();
        TORCH_CHECK(T >= 1 && T <= 32, "verify_tree_step supports T in [1, 32], got ", T);
        TORCH_CHECK((int64_t)tree_depths.size() == T && (int64_t)tree_mask.size() == T,
                    "tree_depths and tree_mask must match T");

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
        if (!use_q8_wte) {
            at::index_select_out(x_cur_2d, wte_embed, 0, idx.view({-1}));
        }
        auto idx_mps = idx.to(torch::kMPS).to(torch::kInt64).contiguous();
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        if (use_q8_wte) {
            dispatch_embed_q8_g64_gather_multi(encoder, idx_mps, wte_packed, wte_sb, x_cur, (uint32_t)q_dim, (uint32_t)T);
        }
        dispatch_rmsnorm_infer_into(encoder, x_cur, ln1_weights[0], norm_cur, eps);

        uint32_t n_splits = (prefix_len <= 64) ? 1u : ((prefix_len <= 256) ? 4u : 8u);

        for (int64_t i = 0; i < n_layer; ++i) {
            dispatch_qmm_into(encoder, norm_cur, c_attn_packed[i], c_attn_sb[i], qkv_cur);

            dispatch_rope_tree_kv_cache_write_bytes(
                encoder,
                qkv_cur,
                freqs_real,
                k_caches[i],
                v_caches[i],
                tree_depths.data(),
                q_pairs,
                kv_pairs,
                half_head_dim,
                max_seq_len,
                (uint32_t)start_pos,
                (uint32_t)write_pos,
                (uint32_t)T,
                attn_bias(i));

            dispatch_gqa_tree_verify_bytes(
                encoder,
                qkv_cur,
                k_caches[i],
                v_caches[i],
                attn_y_cur,
                partial_O_buf,
                partial_m_buf,
                partial_l_buf,
                tree_mask.data(),
                max_seq_len,
                (uint32_t)prefix_len,
                (uint32_t)write_pos,
                (uint32_t)T,
                n_splits,
                (uint32_t)n_head,
                (uint32_t)n_kv_head,
                (uint32_t)head_dim,
                (uint32_t)qkv_dim);

            dispatch_qmm_into(encoder, attn_y_cur, c_proj_packed[i], c_proj_sb[i], attn_out_cur);

            dispatch_add_rmsnorm_infer_into(
                encoder, x_cur, attn_out_cur, ln2_weights[i], norm_cur, eps);

            dispatch_qmm_into(encoder, norm_cur, gate_up_packed[i], gate_up_sb[i], gate_up_cur);
            dispatch_swiglu_gate_up_into(encoder, gate_up_cur, swiglu_cur);
            dispatch_qmm_into(encoder, swiglu_cur, w_down_packed[i], w_down_sb[i], mlp_out_cur);

            const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
            dispatch_add_rmsnorm_infer_into(
                encoder, x_cur, mlp_out_cur, next_ln_w, norm_cur, eps);
        }

        dispatch_qmm_into(encoder, norm_cur, lm_head_packed, lm_head_sb, logits_cur);
        return logits_cur;
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
        TORCH_CHECK(T <= 32, "verify_step supports up to 32 tokens, got ", T);

        if (write_pos + T <= (int64_t)max_seq_len && active_len == write_pos + T) {
            std::vector<int32_t> depths(T);
            std::vector<uint32_t> masks(T);
            for (int64_t t = 0; t < T; ++t) {
                depths[t] = (int32_t)t;
                masks[t] = (t == 31) ? 0xFFFFFFFFu : ((1u << (uint32_t)(t + 1)) - 1u);
            }
            return verify_tree_step(idx, start_pos, write_pos, write_pos, depths, masks);
        }

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
        if (!use_q8_wte) {
            at::index_select_out(x_cur_2d, wte_embed, 0, idx.view({-1}));
        }
        auto idx_mps = idx.to(torch::kMPS).to(torch::kInt64).contiguous();
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        if (use_q8_wte) {
            dispatch_embed_q8_g64_gather_multi(encoder, idx_mps, wte_packed, wte_sb, x_cur, (uint32_t)q_dim, (uint32_t)T);
        }
        dispatch_rmsnorm_infer_into(encoder, x_cur, ln1_weights[0], norm_cur, eps);

        for (int64_t i = 0; i < n_layer; ++i) {
            dispatch_qmm_into(encoder, norm_cur, c_attn_packed[i], c_attn_sb[i], qkv_cur);
            const auto& b_i = attn_bias(i);

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
                    (start_pos + t) % freqs_real.size(0),
                    (uint32_t)((write_pos + t) % max_seq_len),
                    b_i);
            }

            for (int64_t t = 0; t < T; ++t) {
                auto qkv_t = qkv_cur.narrow(1, t, 1);
                auto attn_y_t = attn_y_cur.narrow(1, t, 1);
                uint32_t cur_active = max_seq_len;
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
                    n_splits,
                    (uint32_t)n_head,
                    (uint32_t)n_kv_head,
                    (uint32_t)head_dim);
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            dispatch_qmm_into(encoder, attn_y_cur, c_proj_packed[i], c_proj_sb[i], attn_out_cur);

            dispatch_add_rmsnorm_infer_into(
                encoder, x_cur, attn_out_cur, ln2_weights[i], norm_cur, eps);

            dispatch_qmm_into(encoder, norm_cur, gate_up_packed[i], gate_up_sb[i], gate_up_cur);
            dispatch_swiglu_gate_up_into(encoder, gate_up_cur, swiglu_cur);
            dispatch_qmm_into(encoder, swiglu_cur, w_down_packed[i], w_down_sb[i], mlp_out_cur);

            const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
            dispatch_add_rmsnorm_infer_into(
                encoder, x_cur, mlp_out_cur, next_ln_w, norm_cur, eps);
        }

        dispatch_qmm_into(encoder, norm_cur, lm_head_packed, lm_head_sb, logits_cur);
        return logits_cur;
    }

    void encode_step_internal(int64_t start_pos, int64_t write_pos, int64_t active_len) {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        if (use_q8_wte) {
            dispatch_embed_q8_g64_gather_multi(encoder, shared_idx_buf, wte_packed, wte_sb, x_buf, (uint32_t)q_dim, 1);
        } else {
            dispatch_embed_multi(encoder, shared_idx_buf, wte_embed, x_buf, (uint32_t)q_dim, 1);
        }
        dispatch_rmsnorm_infer_into(encoder, x_buf, ln1_weights[0], norm_buf, eps);

        uint32_t n_splits = (active_len <= 64) ? 1 : ((active_len <= 256) ? 4 : 8);

        for (int64_t i = 0; i < n_layer; ++i) {
            dispatch_qmv_into(encoder, norm_buf, c_attn_packed[i], c_attn_sb[i], qkv_buf);

            dispatch_rope_kv_cache_write(
                encoder,
                qkv_buf,
                freqs_real,
                k_caches[i],
                v_caches[i],
                q_pairs,
                kv_pairs,
                half_head_dim,
                max_seq_len,
                start_pos,
                (uint32_t)write_pos,
                attn_bias(i));

            dispatch_gqa_flash_decode_into(
                encoder,
                qkv_buf,
                k_caches[i],
                v_caches[i],
                attn_y_buf,
                partial_O_buf,
                partial_m_buf,
                partial_l_buf,
                max_seq_len,
                (uint32_t)active_len,
                n_splits,
                (uint32_t)n_head,
                (uint32_t)n_kv_head,
                (uint32_t)head_dim);

            dispatch_qmv_into(encoder, attn_y_buf, c_proj_packed[i], c_proj_sb[i], attn_out_buf);

            dispatch_add_rmsnorm_infer_into(
                encoder, x_buf, attn_out_buf, ln2_weights[i], norm_buf, eps);

            dispatch_qmv_into(encoder, norm_buf, gate_up_packed[i], gate_up_sb[i], gate_up_buf);
            dispatch_swiglu_gate_up_into(encoder, gate_up_buf, swiglu_buf);
            dispatch_qmv_into(encoder, swiglu_buf, w_down_packed[i], w_down_sb[i], mlp_out_buf);

            const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
            dispatch_add_rmsnorm_infer_into(
                encoder, x_buf, mlp_out_buf, next_ln_w, norm_buf, eps);
        }

        dispatch_qmv_into(encoder, norm_buf, lm_head_packed, lm_head_sb, logits_buf);
        dispatch_argmax_multi(encoder, logits_buf, shared_argmax_buf, (uint32_t)vocab_size, 1);
    }

    void encode_verify_step_internal(int64_t T, int64_t start_pos, int64_t write_pos) {
        auto x_cur = x_buf_multi.narrow(1, 0, T);
        auto norm_cur = norm_buf_multi.narrow(1, 0, T);
        auto qkv_cur = qkv_buf_multi.narrow(1, 0, T);
        auto attn_y_cur = attn_y_buf_multi.narrow(1, 0, T);
        auto attn_out_cur = attn_out_buf_multi.narrow(1, 0, T);
        auto gate_up_cur = gate_up_buf_multi.narrow(1, 0, T);
        auto swiglu_cur = swiglu_buf_multi.narrow(1, 0, T);
        auto mlp_out_cur = mlp_out_buf_multi.narrow(1, 0, T);
        auto logits_cur = logits_buf_multi.narrow(1, 0, T);

        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        if (use_q8_wte) {
            dispatch_embed_q8_g64_gather_multi(encoder, shared_idx_buf, wte_packed, wte_sb, x_cur, (uint32_t)q_dim, (uint32_t)T);
        } else {
            dispatch_embed_multi(encoder, shared_idx_buf, wte_embed, x_cur, (uint32_t)q_dim, (uint32_t)T);
        }
        dispatch_rmsnorm_infer_into(encoder, x_cur, ln1_weights[0], norm_cur, eps);

        id<MTLBuffer> qkv_buf_storage = getMTLBufferStorage(qkv_cur);
        id<MTLBuffer> attn_y_buf_storage = getMTLBufferStorage(attn_y_cur);
        NSUInteger elem_sz = qkv_cur.element_size();

        for (int64_t i = 0; i < n_layer; ++i) {
            dispatch_qmm_into(encoder, norm_cur, c_attn_packed[i], c_attn_sb[i], qkv_cur);
            const auto& b_i = attn_bias(i);

            if (write_pos + T <= (int64_t)max_seq_len) {
                std::vector<int32_t> depths(T);
                std::vector<uint32_t> masks(T);
                for (int64_t t = 0; t < T; ++t) {
                    depths[t] = (int32_t)t;
                    masks[t] = (t == 31) ? 0xFFFFFFFFu : ((1u << (uint32_t)(t + 1)) - 1u);
                }
                uint32_t n_splits = (write_pos <= 64) ? 1u : ((write_pos <= 256) ? 4u : 8u);
                dispatch_rope_tree_kv_cache_write_bytes(
                    encoder, qkv_cur, freqs_real, k_caches[i], v_caches[i], depths.data(),
                    q_pairs, kv_pairs, half_head_dim, max_seq_len, (uint32_t)start_pos,
                    (uint32_t)write_pos, (uint32_t)T, b_i);
                dispatch_gqa_tree_verify_bytes(
                    encoder, qkv_cur, k_caches[i], v_caches[i], attn_y_cur,
                    partial_O_buf, partial_m_buf, partial_l_buf, masks.data(),
                    max_seq_len, (uint32_t)write_pos, (uint32_t)write_pos, (uint32_t)T,
                    n_splits, (uint32_t)n_head, (uint32_t)n_kv_head, (uint32_t)head_dim,
                    (uint32_t)qkv_dim);
            } else {
                for (int64_t t = 0; t < T; ++t) {
                    NSUInteger qkv_t_offset = (qkv_cur.storage_offset() + t * qkv_dim) * elem_sz;
                    dispatch_rope_kv_cache_write_offset(
                        encoder,
                        qkv_buf_storage,
                        qkv_t_offset,
                        freqs_real,
                        k_caches[i],
                        v_caches[i],
                        q_pairs,
                        kv_pairs,
                        half_head_dim,
                        max_seq_len,
                        (start_pos + t) % freqs_real.size(0),
                        (uint32_t)((write_pos + t) % max_seq_len),
                        b_i);
                }

                for (int64_t t = 0; t < T; ++t) {
                    NSUInteger qkv_t_offset = (qkv_cur.storage_offset() + t * qkv_dim) * elem_sz;
                    NSUInteger attn_y_t_offset = (attn_y_cur.storage_offset() + t * q_dim) * elem_sz;
                    uint32_t cur_active = max_seq_len;
                    uint32_t n_splits = (cur_active <= 64) ? 1 : ((cur_active <= 256) ? 4 : 8);

                    dispatch_gqa_flash_decode_into_offset(
                        encoder,
                        qkv_buf_storage,
                        qkv_t_offset,
                        k_caches[i],
                        v_caches[i],
                        attn_y_buf_storage,
                        attn_y_t_offset,
                        partial_O_buf,
                        partial_m_buf,
                        partial_l_buf,
                        max_seq_len,
                        cur_active,
                        n_splits,
                        (uint32_t)n_head,
                        (uint32_t)n_kv_head,
                        (uint32_t)head_dim);
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }

            dispatch_qmm_into(encoder, attn_y_cur, c_proj_packed[i], c_proj_sb[i], attn_out_cur);

            dispatch_add_rmsnorm_infer_into(
                encoder, x_cur, attn_out_cur, ln2_weights[i], norm_cur, eps);

            dispatch_qmm_into(encoder, norm_cur, gate_up_packed[i], gate_up_sb[i], gate_up_cur);
            dispatch_swiglu_gate_up_into(encoder, gate_up_cur, swiglu_cur);
            dispatch_qmm_into(encoder, swiglu_cur, w_down_packed[i], w_down_sb[i], mlp_out_cur);

            const torch::Tensor& next_ln_w = (i + 1 < n_layer) ? ln1_weights[i + 1] : lnf_weight;
            dispatch_add_rmsnorm_infer_into(
                encoder, x_cur, mlp_out_cur, next_ln_w, norm_cur, eps);
        }

        dispatch_qmm_into(encoder, norm_cur, lm_head_packed, lm_head_sb, logits_cur);
        dispatch_argmax_multi(encoder, logits_cur, shared_argmax_buf, (uint32_t)vocab_size, (uint32_t)T);
    }

    std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t, int64_t, double>
    speculative_decode(
        const std::vector<int64_t>& initial_tokens,
        int64_t max_new_tokens,
        int64_t ngram_size,
        int64_t max_spec_tokens,
        int64_t start_pos) {
        at::NoGradGuard no_grad;
        std::vector<int64_t> generated = initial_tokens;
        int64_t target_total = (int64_t)initial_tokens.size() + max_new_tokens;
        int64_t checkpoint_seen = start_pos;
        int64_t model_forward_calls = 0;
        int64_t total_drafted = 0;
        int64_t total_accepted = 0;

        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
        auto t0 = std::chrono::high_resolution_clock::now();

        long* idx_ptr = (long*)[shared_idx_buf contents];
        long* preds = (long*)[shared_argmax_buf contents];

        while ((int64_t)generated.size() < target_total) {
            std::vector<int64_t> candidates = find_candidate_ngram_cpp(generated, ngram_size, max_spec_tokens);
            int64_t k_cand = (int64_t)candidates.size();

            if (k_cand == 0) {
                int64_t rope_pos = checkpoint_seen % freqs_real.size(0);
                int64_t write_pos = checkpoint_seen % max_seq_len;
                int64_t cur_active = std::min(checkpoint_seen + 1, (int64_t)max_seq_len);

                idx_ptr[0] = generated.back();

                encode_step_internal(rope_pos, write_pos, cur_active);
                stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

                generated.push_back(preds[0]);
                checkpoint_seen += 1;
                model_forward_calls += 1;
            } else {
                int64_t T = 1 + k_cand;
                int64_t rope_pos = checkpoint_seen % freqs_real.size(0);
                int64_t write_pos = checkpoint_seen % max_seq_len;

                idx_ptr[0] = generated.back();
                for (int64_t c = 0; c < k_cand; ++c) {
                    idx_ptr[1 + c] = candidates[c];
                }

                encode_verify_step_internal(T, rope_pos, write_pos);
                stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

                total_drafted += k_cand;
                model_forward_calls += 1;

                int64_t accepted = 0;
                bool all_matched = true;

                for (int64_t j = 0; j < k_cand; ++j) {
                    int64_t pred_tok = preds[j];
                    if (pred_tok == candidates[j]) {
                        generated.push_back(pred_tok);
                        accepted += 1;
                        if ((int64_t)generated.size() >= target_total) {
                            all_matched = false;
                            break;
                        }
                    } else {
                        generated.push_back(pred_tok);
                        accepted += 1;
                        all_matched = false;
                        break;
                    }
                }

                if (all_matched && (int64_t)generated.size() < target_total) {
                    int64_t bonus_tok = preds[k_cand];
                    generated.push_back(bonus_tok);
                    accepted += 1;
                }

                total_accepted += std::max((int64_t)0, accepted - 1);
                checkpoint_seen += accepted;
            }
        }

        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();

        return std::make_tuple(generated, checkpoint_seen, model_forward_calls, total_drafted, total_accepted, elapsed_seconds);
    }

    torch::Tensor get_last_norm(int64_t T = 1) {
        if (T == 1) {
            return norm_buf.clone();
        }
        return norm_buf_multi.narrow(1, 0, T).clone();
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

torch::Tensor qmv_ternary_mps(torch::Tensor x, torch::Tensor w_packed, torch::Tensor scales, c10::optional<torch::Tensor> bias = c10::nullopt) {
    int64_t N = w_packed.size(0);
    auto sizes = x.sizes().vec();
    sizes.back() = N;
    auto y = torch::empty(sizes, x.options());
    auto stream = at::mps::getCurrentMPSStream();
    torch::Tensor b_ten = (bias.has_value() && bias->defined()) ? *bias : torch::Tensor();
    dispatch_qmv_ternary_into(stream->commandEncoder(), x, w_packed, scales, y, b_ten);
    return y;
}

torch::Tensor qmm_ternary_mps(torch::Tensor x, torch::Tensor w_packed, torch::Tensor scales, c10::optional<torch::Tensor> bias = c10::nullopt) {
    int64_t N = w_packed.size(0);
    auto sizes = x.sizes().vec();
    sizes.back() = N;
    auto y = torch::empty(sizes, x.options());
    auto stream = at::mps::getCurrentMPSStream();
    torch::Tensor b_ten = (bias.has_value() && bias->defined()) ? *bias : torch::Tensor();
    dispatch_qmm_ternary_into(stream->commandEncoder(), x, w_packed, scales, y, b_ten);
    return y;
}

torch::Tensor qmv_q2_k_mps(torch::Tensor x, torch::Tensor w_packed, torch::Tensor scales_biases, c10::optional<torch::Tensor> bias = c10::nullopt) {
    int64_t N = w_packed.size(0);
    auto sizes = x.sizes().vec();
    sizes.back() = N;
    auto y = torch::empty(sizes, x.options());
    auto stream = at::mps::getCurrentMPSStream();
    torch::Tensor b_ten = (bias.has_value() && bias->defined()) ? *bias : torch::Tensor();
    dispatch_qmv_q2_k_into(stream->commandEncoder(), x, w_packed, scales_biases, y, b_ten);
    return y;
}

torch::Tensor qmm_q2_k_mps(torch::Tensor x, torch::Tensor w_packed, torch::Tensor scales_biases, c10::optional<torch::Tensor> bias = c10::nullopt) {
    int64_t N = w_packed.size(0);
    auto sizes = x.sizes().vec();
    sizes.back() = N;
    auto y = torch::empty(sizes, x.options());
    auto stream = at::mps::getCurrentMPSStream();
    torch::Tensor b_ten = (bias.has_value() && bias->defined()) ? *bias : torch::Tensor();
    dispatch_qmm_q2_k_into(stream->commandEncoder(), x, w_packed, scales_biases, y, b_ten);
    return y;
}

torch::Tensor qmv_q4_marlin_mps(
    torch::Tensor x,
    torch::Tensor w_marlin,
    torch::Tensor sb_marlin,
    int64_t out_features,
    c10::optional<torch::Tensor> bias) {
    int64_t N = (out_features > 0) ? out_features : (w_marlin.size(0) * 8);
    auto x_c = x.is_contiguous() ? x : x.contiguous();
    auto sizes = x_c.sizes().vec();
    sizes.back() = N;
    auto y = torch::empty(sizes, x_c.options());
    auto stream = at::mps::getCurrentMPSStream();
    torch::Tensor b_ten = (bias.has_value() && bias->defined()) ? *bias : torch::Tensor();
    dispatch_qmv_q4_marlin_into(stream->commandEncoder(), x_c, w_marlin, sb_marlin, y, b_ten);
    return y;
}

torch::Tensor qmm_q4_marlin_mps(
    torch::Tensor x,
    torch::Tensor w_marlin,
    torch::Tensor sb_marlin,
    int64_t out_features,
    c10::optional<torch::Tensor> bias) {
    int64_t N = (out_features > 0) ? out_features : (w_marlin.size(0) * 8);
    auto x_c = x.is_contiguous() ? x : x.contiguous();
    auto sizes = x_c.sizes().vec();
    sizes.back() = N;
    auto y = torch::empty(sizes, x_c.options());
    auto stream = at::mps::getCurrentMPSStream();
    torch::Tensor b_ten = (bias.has_value() && bias->defined()) ? *bias : torch::Tensor();
    dispatch_qmm_q4_marlin_into(stream->commandEncoder(), x_c, w_marlin, sb_marlin, y, b_ten);
    return y;
}

void rope_tree_kv_cache_write_mps(
    torch::Tensor qkv,
    torch::Tensor freqs_cis,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    const std::vector<int32_t>& tree_depths,
    int64_t q_dim,
    int64_t kv_dim,
    int64_t head_dim,
    int64_t start_pos,
    int64_t write_pos,
    c10::optional<torch::Tensor> qkv_bias) {
    uint32_t T = (uint32_t)tree_depths.size();
    TORCH_CHECK(T >= 1 && T <= 32, "rope_tree_kv_cache_write_mps supports T in [1, 32], got ", T);
    uint32_t q_pairs = (uint32_t)(q_dim / 2);
    uint32_t kv_pairs = (uint32_t)(kv_dim / 2);
    uint32_t half_head_dim = (uint32_t)(head_dim / 2);
    uint32_t max_seq_len = (uint32_t)k_cache.size(2);
    torch::Tensor b_ten = (qkv_bias.has_value() && qkv_bias->defined()) ? *qkv_bias : torch::Tensor();
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_rope_tree_kv_cache_write_bytes(
        stream->commandEncoder(),
        qkv,
        freqs_cis,
        k_cache,
        v_cache,
        tree_depths.data(),
        q_pairs,
        kv_pairs,
        half_head_dim,
        max_seq_len,
        (uint32_t)start_pos,
        (uint32_t)write_pos,
        T,
        b_ten);
}

torch::Tensor gqa_tree_verify_mps(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    const std::vector<uint32_t>& tree_mask,
    int64_t prefix_len,
    int64_t write_pos,
    int64_t n_head,
    int64_t num_splits) {
    int64_t q_dim = (q.dim() == 4) ? (n_head * q.size(-1)) : q.size(-1);
    uint32_t head_dim = (uint32_t)(q_dim / n_head);
    uint32_t n_kv_head = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(1) : (uint32_t)k_cache.size(0);
    uint32_t max_seq_len = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(2) : (uint32_t)k_cache.size(1);

    torch::Tensor q_2d;
    if (q.dim() == 4) {
        q_2d = q.transpose(1, 2).reshape({-1, q_dim}).contiguous();
    } else if (q.dim() == 3 && q.size(0) == 1 && q.size(2) == q_dim && q.stride(2) == 1) {
        q_2d = q.select(0, 0);
    } else if (q.dim() == 2 && q.size(1) == q_dim && q.stride(1) == 1) {
        q_2d = q;
    } else {
        q_2d = q.reshape({-1, q_dim}).contiguous();
    }

    uint32_t T = (uint32_t)q_2d.size(0);
    TORCH_CHECK(T >= 1 && T <= 32, "gqa_tree_verify_mps supports T in [1, 32], got ", T);
    TORCH_CHECK(tree_mask.size() == (size_t)T, "tree_mask size must match T");
    uint32_t q_stride = (uint32_t)q_2d.stride(0);
    uint32_t half_hd = head_dim >> 1;
    uint32_t n_splits = (num_splits > 0)
        ? (uint32_t)num_splits
        : ((prefix_len <= 64) ? 1u : ((prefix_len <= 256) ? 4u : 8u));

    auto opts = q.options();
    auto out = torch::empty({1, (int64_t)T, q_dim}, opts);
    auto partial_O = torch::empty({(int64_t)(T * n_head), (int64_t)n_splits, (int64_t)half_hd, 2}, opts);
    auto partial_m = torch::empty({(int64_t)(T * n_head), (int64_t)n_splits}, opts.dtype(torch::kFloat32));
    auto partial_l = torch::empty({(int64_t)(T * n_head), (int64_t)n_splits}, opts.dtype(torch::kFloat32));

    auto stream = at::mps::getCurrentMPSStream();
    dispatch_gqa_tree_verify_bytes(
        stream->commandEncoder(),
        q_2d,
        k_cache,
        v_cache,
        out,
        partial_O,
        partial_m,
        partial_l,
        tree_mask.data(),
        max_seq_len,
        (uint32_t)prefix_len,
        (uint32_t)write_pos,
        T,
        n_splits,
        (uint32_t)n_head,
        n_kv_head,
        head_dim,
        q_stride);
    if (q.dim() == 4) {
        return out.view({1, (int64_t)T, (int64_t)n_head, (int64_t)head_dim}).transpose(1, 2).contiguous();
    }
    return out.view(q.sizes());
}

void compact_tree_kv_mps(
    torch::Tensor cache,
    int64_t write_pos,
    const std::vector<int32_t>& accepted_indices) {
    uint32_t num_accepted = (uint32_t)accepted_indices.size();
    if (num_accepted == 0u) return;
    TORCH_CHECK(num_accepted <= 32u, "compact_tree_kv_mps supports up to 32 accepted tokens");
    TORCH_CHECK(cache.is_contiguous() && cache.scalar_type() == at::kHalf && cache.dim() == 6,
                "compact_tree_kv_mps expects contiguous FP16 6D StaticKVCache tensor");

    bool already_compact = true;
    for (uint32_t m = 0; m < num_accepted; ++m) {
        if (accepted_indices[m] != (int32_t)m) {
            already_compact = false;
            break;
        }
    }
    if (already_compact) return;

    uint32_t n_layer = (uint32_t)cache.size(0);
    uint32_t max_batch_size = (uint32_t)cache.size(2);
    uint32_t n_kv_head = (uint32_t)cache.size(3);
    uint32_t max_seq_len = (uint32_t)cache.size(4);
    uint32_t half_head_dim = (uint32_t)(cache.size(5) / 2);
    uint32_t w_pos = (uint32_t)write_pos;
    uint32_t total_streams = n_layer * 2u * n_kv_head * half_head_dim;

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:compact_tree_kv_half_pso];
    [encoder setBuffer:getMTLBufferStorage(cache) offset:cache.storage_offset() * cache.element_size() atIndex:0];
    [encoder setBytes:accepted_indices.data() length:num_accepted * sizeof(int32_t) atIndex:1];
    [encoder setBytes:&n_layer length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&max_batch_size length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&w_pos length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&num_accepted length:sizeof(uint32_t) atIndex:8];

    NSUInteger tgSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_streams + tgSize - 1) / tgSize, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
}

torch::Tensor gqa_tree_verify_fp8_mps(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    const std::vector<uint32_t>& tree_mask,
    int64_t prefix_len,
    int64_t write_pos,
    int64_t n_head,
    int64_t num_splits) {
    int64_t q_dim = (q.dim() == 4) ? (n_head * q.size(-1)) : q.size(-1);
    uint32_t head_dim = (uint32_t)(q_dim / n_head);
    uint32_t n_kv_head = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(1) : (uint32_t)k_cache.size(0);
    uint32_t max_seq_len = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(2) : (uint32_t)k_cache.size(1);

    torch::Tensor q_2d;
    if (q.dim() == 4) {
        q_2d = q.transpose(1, 2).reshape({-1, q_dim}).contiguous();
    } else if (q.dim() == 3 && q.size(0) == 1 && q.size(2) == q_dim && q.stride(2) == 1) {
        q_2d = q.select(0, 0);
    } else if (q.dim() == 2 && q.size(1) == q_dim && q.stride(1) == 1) {
        q_2d = q;
    } else {
        q_2d = q.reshape({-1, q_dim}).contiguous();
    }

    uint32_t T = (uint32_t)q_2d.size(0);
    TORCH_CHECK(T >= 1 && T <= 32, "gqa_tree_verify_fp8_mps supports T in [1, 32], got ", T);
    TORCH_CHECK(tree_mask.size() == (size_t)T, "tree_mask size must match T");
    uint32_t q_stride = (uint32_t)q_2d.stride(0);
    uint32_t half_hd = head_dim >> 1;
    uint32_t n_splits = (num_splits > 0)
        ? (uint32_t)num_splits
        : ((prefix_len <= 64) ? 1u : ((prefix_len <= 256) ? 4u : 8u));

    auto opts = q.options();
    auto out = torch::empty({1, (int64_t)T, q_dim}, opts);
    auto partial_O = torch::empty({(int64_t)(T * n_head), (int64_t)n_splits, (int64_t)half_hd, 2}, opts);
    auto partial_m = torch::empty({(int64_t)(T * n_head), (int64_t)n_splits}, opts.dtype(torch::kFloat32));
    auto partial_l = torch::empty({(int64_t)(T * n_head), (int64_t)n_splits}, opts.dtype(torch::kFloat32));

    auto stream = at::mps::getCurrentMPSStream();
    dispatch_gqa_tree_verify_fp8_bytes(
        stream->commandEncoder(),
        q_2d,
        k_cache,
        v_cache,
        k_scales,
        v_scales,
        out,
        partial_O,
        partial_m,
        partial_l,
        tree_mask.data(),
        max_seq_len,
        (uint32_t)prefix_len,
        (uint32_t)write_pos,
        T,
        n_splits,
        (uint32_t)n_head,
        n_kv_head,
        head_dim,
        q_stride);
    if (q.dim() == 4) {
        return out.view({1, (int64_t)T, (int64_t)n_head, (int64_t)head_dim}).transpose(1, 2).contiguous();
    }
    return out.view(q.sizes());
}

torch::Tensor gqa_tree_verify_int4_mps(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    const std::vector<uint32_t>& tree_mask,
    int64_t prefix_len,
    int64_t write_pos,
    int64_t n_head,
    int64_t num_splits) {
    int64_t q_dim = (q.dim() == 4) ? (n_head * q.size(-1)) : q.size(-1);
    uint32_t head_dim = (uint32_t)(q_dim / n_head);
    uint32_t n_kv_head = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(1) : (uint32_t)k_cache.size(0);
    uint32_t max_seq_len = (k_cache.dim() == 4) ? (uint32_t)k_cache.size(2) : (uint32_t)k_cache.size(1);

    torch::Tensor q_2d;
    if (q.dim() == 4) {
        q_2d = q.transpose(1, 2).reshape({-1, q_dim}).contiguous();
    } else if (q.dim() == 3 && q.size(0) == 1 && q.size(2) == q_dim && q.stride(2) == 1) {
        q_2d = q.select(0, 0);
    } else if (q.dim() == 2 && q.size(1) == q_dim && q.stride(1) == 1) {
        q_2d = q;
    } else {
        q_2d = q.reshape({-1, q_dim}).contiguous();
    }

    uint32_t T = (uint32_t)q_2d.size(0);
    TORCH_CHECK(T >= 1 && T <= 32, "gqa_tree_verify_int4_mps supports T in [1, 32], got ", T);
    TORCH_CHECK(tree_mask.size() == (size_t)T, "tree_mask size must match T");
    uint32_t q_stride = (uint32_t)q_2d.stride(0);
    uint32_t half_hd = head_dim >> 1;
    uint32_t n_splits = (num_splits > 0)
        ? (uint32_t)num_splits
        : ((prefix_len <= 64) ? 1u : ((prefix_len <= 256) ? 4u : 8u));

    auto opts = q.options();
    auto out = torch::empty({1, (int64_t)T, q_dim}, opts);
    auto partial_O = torch::empty({(int64_t)(T * n_head), (int64_t)n_splits, (int64_t)half_hd, 2}, opts);
    auto partial_m = torch::empty({(int64_t)(T * n_head), (int64_t)n_splits}, opts.dtype(torch::kFloat32));
    auto partial_l = torch::empty({(int64_t)(T * n_head), (int64_t)n_splits}, opts.dtype(torch::kFloat32));

    auto stream = at::mps::getCurrentMPSStream();
    dispatch_gqa_tree_verify_int4_bytes(
        stream->commandEncoder(),
        q_2d,
        k_cache,
        v_cache,
        k_scales,
        v_scales,
        out,
        partial_O,
        partial_m,
        partial_l,
        tree_mask.data(),
        max_seq_len,
        (uint32_t)prefix_len,
        (uint32_t)write_pos,
        T,
        n_splits,
        (uint32_t)n_head,
        n_kv_head,
        head_dim,
        q_stride);
    if (q.dim() == 4) {
        return out.view({1, (int64_t)T, (int64_t)n_head, (int64_t)head_dim}).transpose(1, 2).contiguous();
    }
    return out.view(q.sizes());
}

void compact_tree_kv_quant_mps(
    torch::Tensor cache,
    torch::Tensor scales,
    int64_t write_pos,
    const std::vector<int32_t>& accepted_indices,
    const std::string& cache_dtype) {
    uint32_t num_accepted = (uint32_t)accepted_indices.size();
    if (num_accepted == 0u) return;
    TORCH_CHECK(num_accepted <= 32u, "compact_tree_kv_quant_mps supports up to 32 accepted tokens");

    bool already_compact = true;
    for (uint32_t m = 0; m < num_accepted; ++m) {
        if (accepted_indices[m] != (int32_t)m) {
            already_compact = false;
            break;
        }
    }
    if (already_compact) return;

    uint32_t n_layer = (uint32_t)cache.size(0);
    uint32_t max_batch_size = (uint32_t)cache.size(2);
    uint32_t n_kv_head = (uint32_t)cache.size(3);
    uint32_t max_seq_len = (uint32_t)cache.size(4);
    uint32_t half_head_dim = (cache_dtype == "fp8") ? (uint32_t)(cache.size(5) / 2) : (uint32_t)cache.size(5);
    uint32_t w_pos = (uint32_t)write_pos;
    uint32_t total_streams = n_layer * 2u * n_kv_head * half_head_dim;

    auto stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    if (cache_dtype == "fp8") {
        [encoder setComputePipelineState:compact_tree_kv_fp8_pso];
    } else {
        [encoder setComputePipelineState:compact_tree_kv_int4_pso];
    }
    [encoder setBuffer:getMTLBufferStorage(cache) offset:cache.storage_offset() * cache.element_size() atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(scales) offset:scales.storage_offset() * scales.element_size() atIndex:1];
    [encoder setBytes:accepted_indices.data() length:num_accepted * sizeof(int32_t) atIndex:2];
    [encoder setBytes:&n_layer length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&max_batch_size length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&n_kv_head length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&max_seq_len length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&half_head_dim length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&w_pos length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&num_accepted length:sizeof(uint32_t) atIndex:9];

    NSUInteger tgSize = 256;
    [encoder dispatchThreadgroups:MTLSizeMake((total_streams + tgSize - 1) / tgSize, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
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
  m.def("rope_tree_kv_cache_write_mps", &rope_tree_kv_cache_write_mps,
        "MPS Topology-Aware Tree RoPE + KV Cache Write",
        pybind11::arg("qkv"),
        pybind11::arg("freqs_cis"),
        pybind11::arg("k_cache"),
        pybind11::arg("v_cache"),
        pybind11::arg("tree_depths"),
        pybind11::arg("q_dim"),
        pybind11::arg("kv_dim"),
        pybind11::arg("head_dim"),
        pybind11::arg("start_pos"),
        pybind11::arg("write_pos"),
        pybind11::arg("qkv_bias") = pybind11::none());
  m.def("transformer_decode_step_mps", &transformer_decode_step_mps, "MPS C++ 12-Layer Whole-Model Decode Step");
  pybind11::class_<DecodeGraphState, std::shared_ptr<DecodeGraphState>>(m, "MPSDecodeRunner")
      .def(pybind11::init([](
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
          float epsilon) {
        return std::make_shared<DecodeGraphState>(
            wte_w, freqs_r, std::move(ln1_ws), std::move(c_attn_ws), std::move(c_proj_ws),
            std::move(ln2_ws), std::move(gate_up_ws), std::move(w_down_ws), lnf_w,
            std::move(k_cs), std::move(v_cs), n_h, n_kvh, hd, epsilon);
      }))
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
          float,
          torch::Tensor,
          std::vector<torch::Tensor>>())
      .def("step", &DecodeGraphState::step)
      .def("verify_step", &DecodeGraphState::verify_step)
      .def("verify_tree_step", &DecodeGraphState::verify_tree_step)
      .def("speculative_decode", &DecodeGraphState::speculative_decode)
      .def("get_last_norm", &DecodeGraphState::get_last_norm, pybind11::arg("T") = 1);
  pybind11::class_<QuantizedDecodeGraphState, std::shared_ptr<QuantizedDecodeGraphState>>(m, "MPSQuantizedDecodeRunner")
      .def(pybind11::init([](
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
          float epsilon) {
        return std::make_shared<QuantizedDecodeGraphState>(
            wte_emb, lmh_packed, lmh_sb, freqs_r, std::move(ln1_ws),
            std::move(c_attn_p), std::move(c_attn_s), std::move(c_proj_p), std::move(c_proj_s),
            std::move(ln2_ws), std::move(gate_up_p), std::move(gate_up_s),
            std::move(w_down_p), std::move(w_down_s), lnf_w, std::move(k_cs), std::move(v_cs),
            n_h, n_kvh, hd, use_q4, epsilon);
      }))
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
          float,
          std::vector<torch::Tensor>,
          int64_t,
          torch::Tensor,
          torch::Tensor>(),
          pybind11::arg("wte_emb"),
          pybind11::arg("lmh_packed"),
          pybind11::arg("lmh_sb"),
          pybind11::arg("freqs_r"),
          pybind11::arg("ln1_ws"),
          pybind11::arg("c_attn_p"),
          pybind11::arg("c_attn_s"),
          pybind11::arg("c_proj_p"),
          pybind11::arg("c_proj_s"),
          pybind11::arg("ln2_ws"),
          pybind11::arg("gate_up_p"),
          pybind11::arg("gate_up_s"),
          pybind11::arg("w_down_p"),
          pybind11::arg("w_down_s"),
          pybind11::arg("lnf_w"),
          pybind11::arg("k_cs"),
          pybind11::arg("v_cs"),
          pybind11::arg("n_h"),
          pybind11::arg("n_kvh"),
          pybind11::arg("hd"),
          pybind11::arg("use_q4"),
          pybind11::arg("epsilon"),
          pybind11::arg("c_attn_bs") = std::vector<torch::Tensor>{},
          pybind11::arg("vocab_sz") = -1,
          pybind11::arg("wte_p") = torch::Tensor(),
          pybind11::arg("wte_s") = torch::Tensor())
      .def("step", &QuantizedDecodeGraphState::step)
      .def("verify_step", &QuantizedDecodeGraphState::verify_step)
      .def("verify_tree_step", &QuantizedDecodeGraphState::verify_tree_step)
      .def("speculative_decode", &QuantizedDecodeGraphState::speculative_decode)
      .def("get_last_norm", &QuantizedDecodeGraphState::get_last_norm, pybind11::arg("T") = 1);
  m.def("cross_entropy_forward_mps", &cross_entropy_forward_mps, "MPS CE Fwd");
  m.def("cross_entropy_backward_mps", &cross_entropy_backward_mps, "MPS CE Bwd");
  m.def("flash_attention_forward_mps", &flash_attention_forward_mps, "MPS FlashAttention Fwd");
  m.def("qmv_q4_g64_mps", &qmv_q4_g64_mps, "MPS Fast Q4_G64 Quantized GEMV");
  m.def("qmm_q4_g64_mps", &qmm_q4_g64_mps, "MPS Batched Q4_G64 Quantized GEMM");
  m.def("qmv_q8_g64_mps", &qmv_q8_g64_mps, "MPS Fast Q8_G64 Quantized GEMV");
  m.def("qmm_q8_g64_mps", &qmm_q8_g64_mps, "MPS Batched Q8_G64 Quantized GEMM");
  m.def("qmv_q4_marlin_mps", &qmv_q4_marlin_mps, "MPS MARLIN + Ladder Q4 GEMV",
        pybind11::arg("x"),
        pybind11::arg("w_marlin"),
        pybind11::arg("sb_marlin"),
        pybind11::arg("out_features") = -1,
        pybind11::arg("bias") = pybind11::none());
  m.def("qmm_q4_marlin_mps", &qmm_q4_marlin_mps, "MPS MARLIN + Ladder Register-Direct Q4 GEMM",
        pybind11::arg("x"),
        pybind11::arg("w_marlin"),
        pybind11::arg("sb_marlin"),
        pybind11::arg("out_features") = -1,
        pybind11::arg("bias") = pybind11::none());
  m.def("gqa_flash_decode_mps", &gqa_flash_decode_mps, "MPS Fast Split-KV Flash-Decode for GQA");
  m.def("gqa_tree_verify_mps", &gqa_tree_verify_mps,
        "MPS SpecInfer Topology-Aware Tree Verification Attention",
        pybind11::arg("q"),
        pybind11::arg("k_cache"),
        pybind11::arg("v_cache"),
        pybind11::arg("tree_mask"),
        pybind11::arg("prefix_len"),
        pybind11::arg("write_pos"),
        pybind11::arg("n_head"),
        pybind11::arg("num_splits") = 0);
  m.def("compact_tree_kv_mps", &compact_tree_kv_mps,
        "MPS In-Place Topological KV-Cache Compaction");
  m.def("rope_kv_cache_write_fp8_mps", &rope_kv_cache_write_fp8_mps,
        "MPS In-Flight FP8 KV Cache Direct Write",
        pybind11::arg("qkv"),
        pybind11::arg("freqs_cis"),
        pybind11::arg("k_cache_fp8"),
        pybind11::arg("v_cache_fp8"),
        pybind11::arg("k_scales"),
        pybind11::arg("v_scales"),
        pybind11::arg("q_dim"),
        pybind11::arg("kv_dim"),
        pybind11::arg("head_dim"),
        pybind11::arg("start_pos"),
        pybind11::arg("write_pos"),
        pybind11::arg("qkv_bias") = pybind11::none());
  m.def("rope_kv_cache_write_int4_mps", &rope_kv_cache_write_int4_mps,
        "MPS In-Flight INT4 KV Cache Direct Write",
        pybind11::arg("qkv"),
        pybind11::arg("freqs_cis"),
        pybind11::arg("k_cache_int4"),
        pybind11::arg("v_cache_int4"),
        pybind11::arg("k_scales"),
        pybind11::arg("v_scales"),
        pybind11::arg("q_dim"),
        pybind11::arg("kv_dim"),
        pybind11::arg("head_dim"),
        pybind11::arg("start_pos"),
        pybind11::arg("write_pos"),
        pybind11::arg("qkv_bias") = pybind11::none());
  m.def("dequantize_kv_cache_fp8_mps", &dequantize_kv_cache_fp8_mps,
        "MPS FP8 KV Cache Dequantization",
        pybind11::arg("cache_fp8"),
        pybind11::arg("scales"),
        pybind11::arg("n_kv_head"),
        pybind11::arg("seq_len"),
        pybind11::arg("head_dim"));
  m.def("dequantize_kv_cache_int4_mps", &dequantize_kv_cache_int4_mps,
        "MPS INT4 KV Cache Dequantization",
        pybind11::arg("cache_int4"),
        pybind11::arg("scales"),
        pybind11::arg("n_kv_head"),
        pybind11::arg("seq_len"),
        pybind11::arg("head_dim"));
  m.def("rope_tree_kv_write_fp8_mps", &rope_tree_kv_write_fp8_mps,
        "MPS In-Flight FP8 Tree Speculative KV Cache Write",
        pybind11::arg("qkv"),
        pybind11::arg("freqs_cis"),
        pybind11::arg("k_cache_fp8"),
        pybind11::arg("v_cache_fp8"),
        pybind11::arg("k_scales"),
        pybind11::arg("v_scales"),
        pybind11::arg("tree_depths"),
        pybind11::arg("q_dim"),
        pybind11::arg("kv_dim"),
        pybind11::arg("head_dim"),
        pybind11::arg("start_pos"),
        pybind11::arg("write_pos"),
        pybind11::arg("T"),
        pybind11::arg("qkv_bias") = pybind11::none());
  m.def("rope_tree_kv_write_int4_mps", &rope_tree_kv_write_int4_mps,
        "MPS In-Flight INT4 Tree Speculative KV Cache Write",
        pybind11::arg("qkv"),
        pybind11::arg("freqs_cis"),
        pybind11::arg("k_cache_int4"),
        pybind11::arg("v_cache_int4"),
        pybind11::arg("k_scales"),
        pybind11::arg("v_scales"),
        pybind11::arg("tree_depths"),
        pybind11::arg("q_dim"),
        pybind11::arg("kv_dim"),
        pybind11::arg("head_dim"),
        pybind11::arg("start_pos"),
        pybind11::arg("write_pos"),
        pybind11::arg("T"),
        pybind11::arg("qkv_bias") = pybind11::none());
  m.def("gqa_flash_decode_fp8_mps", &gqa_flash_decode_fp8_mps,
        "MPS Fused In-Register FP8 Split-KV Flash-Decode",
        pybind11::arg("q"),
        pybind11::arg("k_cache"),
        pybind11::arg("v_cache"),
        pybind11::arg("k_scales"),
        pybind11::arg("v_scales"),
        pybind11::arg("active_len"),
        pybind11::arg("num_splits") = 0);
  m.def("gqa_flash_decode_int4_mps", &gqa_flash_decode_int4_mps,
        "MPS Fused In-Register INT4 Split-KV Flash-Decode",
        pybind11::arg("q"),
        pybind11::arg("k_cache"),
        pybind11::arg("v_cache"),
        pybind11::arg("k_scales"),
        pybind11::arg("v_scales"),
        pybind11::arg("active_len"),
        pybind11::arg("num_splits") = 0);
  m.def("gqa_tree_verify_fp8_mps", &gqa_tree_verify_fp8_mps,
        "MPS Fused In-Register FP8 Tree Verification Attention",
        pybind11::arg("q"),
        pybind11::arg("k_cache"),
        pybind11::arg("v_cache"),
        pybind11::arg("k_scales"),
        pybind11::arg("v_scales"),
        pybind11::arg("tree_mask"),
        pybind11::arg("prefix_len"),
        pybind11::arg("write_pos"),
        pybind11::arg("n_head"),
        pybind11::arg("num_splits") = 0);
  m.def("gqa_tree_verify_int4_mps", &gqa_tree_verify_int4_mps,
        "MPS Fused In-Register INT4 Tree Verification Attention",
        pybind11::arg("q"),
        pybind11::arg("k_cache"),
        pybind11::arg("v_cache"),
        pybind11::arg("k_scales"),
        pybind11::arg("v_scales"),
        pybind11::arg("tree_mask"),
        pybind11::arg("prefix_len"),
        pybind11::arg("write_pos"),
        pybind11::arg("n_head"),
        pybind11::arg("num_splits") = 0);
  m.def("compact_tree_kv_quant_mps", &compact_tree_kv_quant_mps,
        "MPS In-Place Quantized KV-Cache Compaction",
        pybind11::arg("cache"),
        pybind11::arg("scales"),
        pybind11::arg("write_pos"),
        pybind11::arg("accepted_indices"),
        pybind11::arg("cache_dtype") = "fp8");
  m.def("qmv_ternary_mps", &qmv_ternary_mps,
        "MPS BitNet 1.58-bit Ternary GEMV",
        pybind11::arg("x"),
        pybind11::arg("w_packed"),
        pybind11::arg("scales"),
        pybind11::arg("bias") = pybind11::none());
  m.def("qmm_ternary_mps", &qmm_ternary_mps,
        "MPS BitNet 1.58-bit Ternary Batched GEMM",
        pybind11::arg("x"),
        pybind11::arg("w_packed"),
        pybind11::arg("scales"),
        pybind11::arg("bias") = pybind11::none());
  m.def("qmv_q2_k_mps", &qmv_q2_k_mps,
        "MPS Q2_K 2-bit Quantized GEMV",
        pybind11::arg("x"),
        pybind11::arg("w_packed"),
        pybind11::arg("scales_biases"),
        pybind11::arg("bias") = pybind11::none());
  m.def("qmm_q2_k_mps", &qmm_q2_k_mps,
        "MPS Q2_K 2-bit Quantized Batched GEMM",
        pybind11::arg("x"),
        pybind11::arg("w_packed"),
        pybind11::arg("scales_biases"),
        pybind11::arg("bias") = pybind11::none());
}
