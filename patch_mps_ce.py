import re

with open("axiomlm/kernels/mps_kernels.mm", "r") as f:
    content = f.read()

# Add pso
content = re.sub(r'static id<MTLComputePipelineState> rope_pso = nil;', 
                 'static id<MTLComputePipelineState> rope_pso = nil;\nstatic id<MTLComputePipelineState> ce_fwd_pso = nil;\nstatic id<MTLComputePipelineState> ce_bwd_pso = nil;', 
                 content)

# Init pso
content = re.sub(r'rope_pso = get_pipeline\(mtl_device, mtl_library, @"rope_kernel"\);',
                 'rope_pso = get_pipeline(mtl_device, mtl_library, @"rope_kernel");\n            ce_fwd_pso = get_pipeline(mtl_device, mtl_library, @"cross_entropy_forward_kernel");\n            ce_bwd_pso = get_pipeline(mtl_device, mtl_library, @"cross_entropy_backward_kernel");',
                 content)

ce_func = """
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
"""

content = content.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {", ce_func + "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")

content = content.replace('m.def("apply_rope_mps", &apply_rope_mps, "MPS Apply RoPE");',
                          'm.def("apply_rope_mps", &apply_rope_mps, "MPS Apply RoPE");\n  m.def("cross_entropy_forward_mps", &cross_entropy_forward_mps, "MPS CE Fwd");\n  m.def("cross_entropy_backward_mps", &cross_entropy_backward_mps, "MPS CE Bwd");')

with open("axiomlm/kernels/mps_kernels.mm", "w") as f:
    f.write(content)
