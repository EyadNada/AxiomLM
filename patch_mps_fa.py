import re

with open("axiomlm/kernels/mps_kernels.mm", "r") as f:
    content = f.read()

content = re.sub(r'static id<MTLComputePipelineState> ce_bwd_pso = nil;', 
                 'static id<MTLComputePipelineState> ce_bwd_pso = nil;\nstatic id<MTLComputePipelineState> fa_fwd_pso = nil;', 
                 content)

content = re.sub(r'ce_bwd_pso = get_pipeline\(mtl_device, mtl_library, @"cross_entropy_backward_kernel"\);',
                 'ce_bwd_pso = get_pipeline(mtl_device, mtl_library, @"cross_entropy_backward_kernel");\n            fa_fwd_pso = get_pipeline(mtl_device, mtl_library, @"flash_attention_forward_kernel");',
                 content)

fa_func = """
torch::Tensor flash_attention_forward_mps(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal) {
    // q, k, v expected to be (B, n_head, T, head_dim)
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
"""

content = content.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {", fa_func + "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
content = content.replace('m.def("cross_entropy_backward_mps", &cross_entropy_backward_mps, "MPS CE Bwd");',
                          'm.def("cross_entropy_backward_mps", &cross_entropy_backward_mps, "MPS CE Bwd");\n  m.def("flash_attention_forward_mps", &flash_attention_forward_mps, "MPS FlashAttention Fwd");')

with open("axiomlm/kernels/mps_kernels.mm", "w") as f:
    f.write(content)
