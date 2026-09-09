#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Apple Silicon Metal Compute Kernels for Fused Operations (Axiom-LM Engine)
// Optimized for Apple Silicon M3 Pro GPU Architecture (SIMD-width = 32)
// ============================================================================

// ----------------------------------------------------------------------------
// 1. Fused RMSNorm Forward Kernel
// Computes: y = (x / sqrt(mean(x^2) + eps)) * weight
// Each threadgroup processes one row (token) of dimension D.
// ----------------------------------------------------------------------------
kernel void rmsnorm_forward_kernel(
    device const float *x [[buffer(0)]],      // Input:  (N_rows, D)
    device const float *weight [[buffer(1)]], // Weight: (D,)
    device float *y [[buffer(2)]],            // Output: (N_rows, D)
    device float *rsqrt_out [[buffer(3)]],    // RMS Cache: (N_rows,)
    constant uint &D [[buffer(4)]],           // Hidden Dimension
    constant float &eps [[buffer(5)]],        // Epsilon
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]) {
  uint row_idx = tgpig.x;
  uint tid = tid_in_tg.x;
  uint block_dim = tg_size.x;

  device const float *row_x = x + row_idx * D;
  device float *row_y = y + row_idx * D;

  // Accumulate sum of squares in thread local register
  float thread_sum_sq = 0.0f;
  for (uint i = tid; i < D; i += block_dim) {
    float val = row_x[i];
    thread_sum_sq += val * val;
  }

  // Threadgroup SRAM reduction
  threadgroup float shared_sq[1024];
  shared_sq[tid] = thread_sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction tree inside threadgroup
  for (uint s = block_dim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_sq[tid] += shared_sq[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Leader thread computes scale factor
  float rsqrt_val = rsqrt((shared_sq[0] / float(D)) + eps);
  if (tid == 0 && rsqrt_out != nullptr) {
    rsqrt_out[row_idx] = rsqrt_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Broadcast rsqrt_val to shared memory
  if (tid == 0) {
    shared_sq[0] = rsqrt_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = shared_sq[0];

  // Fused write-back: scale and affine multiply
  for (uint i = tid; i < D; i += block_dim) {
    row_y[i] = row_x[i] * scale * weight[i];
  }
}

// ----------------------------------------------------------------------------
// 2. Fused SwiGLU Forward Kernel
// Computes: y = (x_gate / (1 + exp(-x_gate))) * x_up
// Vectorized SIMD float4 execution.
// ----------------------------------------------------------------------------
kernel void swiglu_forward_kernel(
    device const float4 *x_gate [[buffer(0)]], // Gate tensor (N_elements / 4)
    device const float4 *x_up [[buffer(1)]],   // Up tensor   (N_elements / 4)
    device float4 *y [[buffer(2)]],            // Output      (N_elements / 4)
    constant uint &num_vec4 [[buffer(3)]],     // Total vec4 elements
    uint gid [[thread_position_in_grid]]) {
  if (gid >= num_vec4)
    return;

  float4 g = x_gate[gid];
  float4 u = x_up[gid];

  // Vectorized SiLU: silu(g) = g / (1 + exp(-g))
  float4 sig = 1.0f / (1.0f + exp(-g));
  float4 silu_g = g * sig;

  // Fused element-wise multiplication
  y[gid] = silu_g * u;
}

// ----------------------------------------------------------------------------
// 3. Fused SwiGLU Backward Kernel
// Computes:
// grad_up   = grad_y * silu(x_gate)
// grad_gate = grad_y * x_up * (sig(x_gate) * (1 + x_gate * (1 - sig(x_gate))))
// ----------------------------------------------------------------------------
kernel void swiglu_backward_kernel(
    device const float *grad_y [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device const float *up [[buffer(2)]],
    device float *grad_gate [[buffer(3)]],
    device float *grad_up [[buffer(4)]],
    constant uint &num_elements [[buffer(5)]], // Number of float4 vectors
    uint tid [[thread_position_in_grid]]
) {
  if (tid >= num_elements)
    return;

  device const float4 *gy_vec = (device const float4 *)grad_y;
  device const float4 *gate_vec = (device const float4 *)gate;
  device const float4 *up_vec = (device const float4 *)up;
  device float4 *gg_vec = (device float4 *)grad_gate;
  device float4 *gu_vec = (device float4 *)grad_up;

  float4 gy = gy_vec[tid];
  float4 g = gate_vec[tid];
  float4 u = up_vec[tid];

  // sigmoid(g) = 1 / (1 + exp(-g))
  float4 sig_g = 1.0f / (1.0f + fast::exp(-g));
  float4 silu_g = g * sig_g;

  // grad_up = grad_y * silu(gate)
  gu_vec[tid] = gy * silu_g;

  // d_silu/d_gate = sig_g * (1 + gate * (1 - sig_g))
  float4 d_silu_dg = sig_g * (1.0f + g * (1.0f - sig_g));
  gg_vec[tid] = gy * u * d_silu_dg;
}

// ----------------------------------------------------------------------------
// RMSNorm Backward kernels
// ----------------------------------------------------------------------------

kernel void rmsnorm_backward_kernel(
    device const float *grad_y [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device const float *rsqrt [[buffer(3)]],
    device float *grad_x [[buffer(4)]],
    constant uint &D [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_dim [[threads_per_threadgroup]]
) {
    uint row_idx = bid;
    device const float *row_gy = grad_y + row_idx * D;
    device const float *row_x = x + row_idx * D;
    device float *row_gx = grad_x + row_idx * D;
    float rsqrt_val = rsqrt[row_idx];

    float thread_inner = 0.0f;
    for (uint i = tid; i < D; i += block_dim) {
        thread_inner += row_gy[i] * weight[i] * row_x[i];
    }

    threadgroup float shared_inner[1024];
    shared_inner[tid] = thread_inner;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_inner[tid] += shared_inner[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inner_sum = shared_inner[0];
    float scale = (inner_sum / (float)D) * (rsqrt_val * rsqrt_val * rsqrt_val);

    for (uint i = tid; i < D; i += block_dim) {
        row_gx[i] = (row_gy[i] * weight[i] * rsqrt_val) - (row_x[i] * scale);
    }
}

kernel void rmsnorm_backward_weight_kernel(
    device const float *grad_y [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *rsqrt [[buffer(2)]],
    device float *grad_w [[buffer(3)]],
    constant uint &D [[buffer(4)]],
    constant uint &num_rows [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= D) return;

    float gw_val = 0.0f;
    for (uint r = 0; r < num_rows; ++r) {
        gw_val += grad_y[r * D + tid] * x[r * D + tid] * rsqrt[r];
    }
    grad_w[tid] = gw_val;
}

// ----------------------------------------------------------------------------
// Rotary Position Embeddings (RoPE) kernel
// ----------------------------------------------------------------------------
kernel void rope_kernel(
    device const float2 *x [[buffer(0)]],
    device const float2 *freqs_cis [[buffer(1)]],
    device float2 *out [[buffer(2)]],
    constant uint &T [[buffer(3)]],
    constant uint &half_head_dim [[buffer(4)]],
    constant uint &forward [[buffer(5)]], // 1 for forward, 0 for backward
    uint tid [[thread_position_in_grid]],
    constant uint &num_elements [[buffer(6)]]
) {
    if (tid >= num_elements) return;

    uint hd_idx = tid % half_head_dim;
    uint t_idx = (tid / half_head_dim) % T;
    
    float2 x_val = x[tid];
    float2 f_val = freqs_cis[t_idx * half_head_dim + hd_idx]; // f_val.x = cos, f_val.y = sin
    
    if (forward == 0) {
        f_val.y = -f_val.y;
    }
    
    float2 res;
    res.x = x_val.x * f_val.x - x_val.y * f_val.y;
    res.y = x_val.x * f_val.y + x_val.y * f_val.x;
    
    out[tid] = res;
}

// ----------------------------------------------------------------------------
// Fused Cross Entropy
// ----------------------------------------------------------------------------

kernel void cross_entropy_forward_kernel(
    device const float *logits [[buffer(0)]],
    device const int *targets [[buffer(1)]],
    device float *losses [[buffer(2)]],
    constant uint &V [[buffer(3)]],
    constant int &ignore_index [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_dim [[threads_per_threadgroup]]
) {
    uint row_idx = bid;
    int target_class = targets[row_idx];
    
    if (target_class == ignore_index) {
        if (tid == 0) losses[row_idx] = 0.0f;
        return;
    }

    device const float *row_logits = logits + row_idx * V;
    
    float thread_max = -1e38f;
    for (uint i = tid; i < V; i += block_dim) {
        thread_max = max(thread_max, row_logits[i]);
    }
    
    threadgroup float shared_max[1024];
    shared_max[tid] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];
    
    float thread_sum = 0.0f;
    for (uint i = tid; i < V; i += block_dim) {
        thread_sum += fast::exp(row_logits[i] - row_max);
    }
    
    threadgroup float shared_sum[1024];
    shared_sum[tid] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared_sum[0];
    
    if (tid == 0) {
        float lse = row_max + fast::log(row_sum);
        losses[row_idx] = lse - row_logits[target_class];
    }
}

kernel void cross_entropy_backward_kernel(
    device const float *logits [[buffer(0)]],
    device const int *targets [[buffer(1)]],
    device const float *grad_losses [[buffer(2)]],
    device float *grad_logits [[buffer(3)]],
    constant uint &V [[buffer(4)]],
    constant int &ignore_index [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_dim [[threads_per_threadgroup]]
) {
    uint row_idx = bid;
    int target_class = targets[row_idx];
    device const float *row_logits = logits + row_idx * V;
    device float *row_grad = grad_logits + row_idx * V;
    
    if (target_class == ignore_index) {
        for (uint i = tid; i < V; i += block_dim) {
            row_grad[i] = 0.0f;
        }
        return;
    }
    
    float go = grad_losses[row_idx];
    
    float thread_max = -1e38f;
    for (uint i = tid; i < V; i += block_dim) {
        thread_max = max(thread_max, row_logits[i]);
    }
    threadgroup float shared_max[1024];
    shared_max[tid] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];
    
    float thread_sum = 0.0f;
    for (uint i = tid; i < V; i += block_dim) {
        thread_sum += fast::exp(row_logits[i] - row_max);
    }
    threadgroup float shared_sum[1024];
    shared_sum[tid] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared_sum[0];
    
    float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < V; i += block_dim) {
        float prob = fast::exp(row_logits[i] - row_max) * inv_sum;
        float grad = prob;
        if ((int)i == target_class) {
            grad -= 1.0f;
        }
        row_grad[i] = grad * go;
    }
}

// ----------------------------------------------------------------------------
// Flash Attention Forward (Custom Apple Silicon AMX/SRAM Optimized)
// ----------------------------------------------------------------------------
#define BLOCK_Q 32
#define BLOCK_K 32

kernel void flash_attention_forward_kernel(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant uint &seq_len_q [[buffer(4)]],
    constant uint &seq_len_k [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &is_causal [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]]
) {
    uint batch_head_idx = bid.x;
    uint block_q_idx = bid.y;
    
    uint q_offset = batch_head_idx * seq_len_q * head_dim;
    uint kv_offset = batch_head_idx * seq_len_k * head_dim;
    
    device const float *q_ptr = Q + q_offset;
    device const float *k_ptr = K + kv_offset;
    device const float *v_ptr = V + kv_offset;
    device float *o_ptr = O + q_offset;
    
    uint q_start = block_q_idx * BLOCK_Q;
    uint q_idx = q_start + tid;
    
    bool valid_q = q_idx < seq_len_q;
    
    // Thread-local state for streaming softmax
    float m_i = -1e38f;
    float l_i = 0.0f;
    float O_i[128]; // support up to head_dim = 128
    for(uint d=0; d<128; d++) O_i[d] = 0.0f;
    
    float Q_i[128];
    if (valid_q) {
        for(uint d=0; d<head_dim; d++) {
            Q_i[d] = q_ptr[q_idx * head_dim + d];
        }
    }
    
    threadgroup float K_shared[BLOCK_K * 128];
    threadgroup float V_shared[BLOCK_K * 128];
    
    float scale = 1.0f / fast::sqrt((float)head_dim);
    uint num_blocks_k = (seq_len_k + BLOCK_K - 1) / BLOCK_K;
    
    for (uint bk = 0; bk < num_blocks_k; bk++) {
        uint k_start = bk * BLOCK_K;
        
        // Collaborative load K and V into SRAM
        uint load_k_idx = k_start + tid;
        if (load_k_idx < seq_len_k) {
            for(uint d=0; d<head_dim; d++) {
                K_shared[tid * head_dim + d] = k_ptr[load_k_idx * head_dim + d];
                V_shared[tid * head_dim + d] = v_ptr[load_k_idx * head_dim + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (valid_q) {
            uint k_limit = min((uint)BLOCK_K, seq_len_k - k_start);
            for (uint j = 0; j < k_limit; j++) {
                uint k_global = k_start + j;
                if (is_causal > 0 && k_global > q_idx) {
                    continue;
                }
                
                // dot product
                float score = 0.0f;
                for (uint d = 0; d < head_dim; d++) {
                    score += Q_i[d] * K_shared[j * head_dim + d];
                }
                score *= scale;
                
                float m_new = max(m_i, score);
                float exp_score = fast::exp(score - m_new);
                float exp_m_diff = fast::exp(m_i - m_new);
                
                l_i = l_i * exp_m_diff + exp_score;
                
                for (uint d = 0; d < head_dim; d++) {
                    O_i[d] = O_i[d] * exp_m_diff + exp_score * V_shared[j * head_dim + d];
                }
                
                m_i = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (valid_q) {
        float inv_l = 1.0f / l_i;
        for (uint d = 0; d < head_dim; d++) {
            o_ptr[q_idx * head_dim + d] = O_i[d] * inv_l;
        }
    }
}
