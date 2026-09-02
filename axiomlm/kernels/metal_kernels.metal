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
  threadgroup float shared_sq[256];
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
kernel void swiglu_backward_kernel(device const float4 *grad_y [[buffer(0)]],
                                   device const float4 *x_gate [[buffer(1)]],
                                   device const float4 *x_up [[buffer(2)]],
                                   device float4 *grad_gate [[buffer(3)]],
                                   device float4 *grad_up [[buffer(4)]],
                                   constant uint &num_vec4 [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
  if (gid >= num_vec4)
    return;

  float4 gy = grad_y[gid];
  float4 g = x_gate[gid];
  float4 u = x_up[gid];

  float4 sig = 1.0f / (1.0f + exp(-g));
  float4 silu_g = g * sig;

  // grad_up = grad_y * silu(g)
  grad_up[gid] = gy * silu_g;

  // d(silu(g))/dg = sig * (1 + g * (1 - sig))
  float4 d_silu = sig * (1.0f + g * (1.0f - sig));
  grad_gate[gid] = gy * u * d_silu;
}
