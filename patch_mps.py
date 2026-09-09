import sys
import re

with open("axiomlm/kernels/mps_kernels.mm", "r") as f:
    content = f.read()

# Add rope_pso
content = re.sub(r'static id<MTLComputePipelineState> swiglu_bwd_pso = nil;', 
                 'static id<MTLComputePipelineState> swiglu_bwd_pso = nil;\nstatic id<MTLComputePipelineState> rope_pso = nil;', 
                 content)

# Init rope_pso
content = re.sub(r'swiglu_bwd_pso = get_pipeline\(mtl_device, mtl_library, @"swiglu_backward_kernel"\);',
                 'swiglu_bwd_pso = get_pipeline(mtl_device, mtl_library, @"swiglu_backward_kernel");\n            rope_pso = get_pipeline(mtl_device, mtl_library, @"rope_kernel");',
                 content)

rope_func = """

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
"""

content = content.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {", rope_func + "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")

content = content.replace('m.def("swiglu_backward_mps", &swiglu_backward_mps, "MPS SwiGLU Backward");',
                          'm.def("swiglu_backward_mps", &swiglu_backward_mps, "MPS SwiGLU Backward");\n  m.def("apply_rope_mps", &apply_rope_mps, "MPS Apply RoPE");')

with open("axiomlm/kernels/mps_kernels.mm", "w") as f:
    f.write(content)
