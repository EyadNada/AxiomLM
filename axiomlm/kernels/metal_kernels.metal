#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Apple Silicon Metal Compute Kernels for Fused Operations (Axiom-LM Engine)
// Optimized for Apple Silicon M3 Pro GPU Architecture (SIMD-width = 32)
// Features:
//   - Hardware SIMD-group reductions (simd_sum, simd_max)
//   - Vectorized float4 / half4 128-bit memory coalescing
//   - Native FP32 and FP16 (half) compute pipelines
// ============================================================================

// ----------------------------------------------------------------------------
// 1A. Fused RMSNorm Forward Kernel (FP32 - Vectorized float4 + simd_sum)
// Computes: y = (x / sqrt(mean(x^2) + eps)) * weight
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
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  uint row_idx = tgpig.x;
  uint tid = tid_in_tg.x;
  uint block_dim = tg_size.x;

  device const float *row_x = x + row_idx * D;
  device float *row_y = y + row_idx * D;

  float thread_sum_sq = 0.0f;
  if ((D & 3) == 0) {
    uint D_vec4 = D >> 2;
    device const float4 *row_x4 = (device const float4 *)row_x;
    for (uint i = tid; i < D_vec4; i += block_dim) {
      float4 v = row_x4[i];
      thread_sum_sq = fma(v.x, v.x, thread_sum_sq);
      thread_sum_sq = fma(v.y, v.y, thread_sum_sq);
      thread_sum_sq = fma(v.z, v.z, thread_sum_sq);
      thread_sum_sq = fma(v.w, v.w, thread_sum_sq);
    }
  } else {
    for (uint i = tid; i < D; i += block_dim) {
      float val = row_x[i];
      thread_sum_sq = fma(val, val, thread_sum_sq);
    }
  }

  // Hardware single-cycle SIMD-group reduction across 32 threads
  float simd_total = simd_sum(thread_sum_sq);

  threadgroup float shared_simd[32];
  if (simd_lane_id == 0) {
    shared_simd[simd_group_id] = simd_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint num_simd_groups = (block_dim + 31) >> 5;
  float total_sq = 0.0f;
  if (simd_group_id == 0) {
    float val = (simd_lane_id < num_simd_groups) ? shared_simd[simd_lane_id] : 0.0f;
    total_sq = simd_sum(val);
    if (simd_lane_id == 0) {
      float rsqrt_val = rsqrt((total_sq / float(D)) + eps);
      shared_simd[0] = rsqrt_val;
      if (rsqrt_out != nullptr) {
        rsqrt_out[row_idx] = rsqrt_val;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = shared_simd[0];

  if ((D & 3) == 0) {
    uint D_vec4 = D >> 2;
    device const float4 *row_x4 = (device const float4 *)row_x;
    device const float4 *w4 = (device const float4 *)weight;
    device float4 *row_y4 = (device float4 *)row_y;
    for (uint i = tid; i < D_vec4; i += block_dim) {
      row_y4[i] = row_x4[i] * scale * w4[i];
    }
  } else {
    for (uint i = tid; i < D; i += block_dim) {
      row_y[i] = row_x[i] * scale * weight[i];
    }
  }
}

// ----------------------------------------------------------------------------
// 1B. Fused RMSNorm Forward Kernel (FP16 - Vectorized half4 + simd_sum)
// ----------------------------------------------------------------------------
kernel void rmsnorm_forward_half_kernel(
    device const half *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *y [[buffer(2)]],
    device float *rsqrt_out [[buffer(3)]],
    constant uint &D [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  uint row_idx = tgpig.x;
  uint tid = tid_in_tg.x;
  uint block_dim = tg_size.x;

  device const half *row_x = x + row_idx * D;
  device half *row_y = y + row_idx * D;

  if (D == 768 && block_dim == 64) {
    device const half4 *row_x4 = (device const half4 *)row_x;
    device const half4 *w4 = (device const half4 *)weight;
    device half4 *row_y4 = (device half4 *)row_y;
    float4 v0 = float4(row_x4[tid]);
    float4 v1 = float4(row_x4[tid + 64]);
    float4 v2 = float4(row_x4[tid + 128]);
    float thread_sum_sq = dot(v0, v0) + dot(v1, v1) + dot(v2, v2);
    float simd_total = simd_sum(thread_sum_sq);
    threadgroup float shared_simd[2];
    if (simd_lane_id == 0) {
      shared_simd[simd_group_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0 && simd_lane_id == 0) {
      float total_sq = shared_simd[0] + shared_simd[1];
      float rsqrt_val = rsqrt((total_sq * (1.0f / 768.0f)) + eps);
      shared_simd[0] = rsqrt_val;
      if (rsqrt_out != nullptr) {
        rsqrt_out[row_idx] = rsqrt_val;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared_simd[0];
    row_y4[tid]       = half4(v0 * scale * float4(w4[tid]));
    row_y4[tid + 64]  = half4(v1 * scale * float4(w4[tid + 64]));
    row_y4[tid + 128] = half4(v2 * scale * float4(w4[tid + 128]));
    return;
  }

  float thread_sum_sq = 0.0f;
  if ((D & 3) == 0) {
    uint D_vec4 = D >> 2;
    device const half4 *row_x4 = (device const half4 *)row_x;
    for (uint i = tid; i < D_vec4; i += block_dim) {
      float4 v = float4(row_x4[i]);
      thread_sum_sq = fma(v.x, v.x, thread_sum_sq);
      thread_sum_sq = fma(v.y, v.y, thread_sum_sq);
      thread_sum_sq = fma(v.z, v.z, thread_sum_sq);
      thread_sum_sq = fma(v.w, v.w, thread_sum_sq);
    }
  } else {
    for (uint i = tid; i < D; i += block_dim) {
      float val = float(row_x[i]);
      thread_sum_sq = fma(val, val, thread_sum_sq);
    }
  }

  float simd_total = simd_sum(thread_sum_sq);
  threadgroup float shared_simd[32];
  if (simd_lane_id == 0) {
    shared_simd[simd_group_id] = simd_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint num_simd_groups = (block_dim + 31) >> 5;
  if (simd_group_id == 0) {
    float val = (simd_lane_id < num_simd_groups) ? shared_simd[simd_lane_id] : 0.0f;
    float total_sq = simd_sum(val);
    if (simd_lane_id == 0) {
      float rsqrt_val = rsqrt((total_sq / float(D)) + eps);
      shared_simd[0] = rsqrt_val;
      if (rsqrt_out != nullptr) {
        rsqrt_out[row_idx] = rsqrt_val;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = shared_simd[0];

  if ((D & 3) == 0) {
    uint D_vec4 = D >> 2;
    device const half4 *row_x4 = (device const half4 *)row_x;
    device const half4 *w4 = (device const half4 *)weight;
    device half4 *row_y4 = (device half4 *)row_y;
    for (uint i = tid; i < D_vec4; i += block_dim) {
      row_y4[i] = half4(float4(row_x4[i]) * scale * float4(w4[i]));
    }
  } else {
    for (uint i = tid; i < D; i += block_dim) {
      row_y[i] = half(float(row_x[i]) * scale * float(weight[i]));
    }
  }
}

// ----------------------------------------------------------------------------
// 1D. Fused Token Embedding Lookup + Layer-0 RMSNorm (FP32 & FP16)
// Eliminates PyTorch MPSGraph embedding/index_select dispatch on T=1 decode:
//   x_out[d]    = wte_table[token_id * D + d]
//   norm_out[d] = rmsnorm(x_out, ln1_weight, eps)
// ----------------------------------------------------------------------------
kernel void embed_rmsnorm_forward_kernel(
    device const long *idx [[buffer(0)]],
    device const float *wte [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *x_out [[buffer(3)]],
    device float *norm_out [[buffer(4)]],
    constant uint &D [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  uint tid = tid_in_tg.x;
  uint block_dim = tg_size.x;
  long token_id = idx[0];
  device const float *row_wte = wte + token_id * long(D);

  float thread_sum_sq = 0.0f;
  for (uint i = tid; i < D; i += block_dim) {
    float val = row_wte[i];
    x_out[i] = val;
    thread_sum_sq = fma(val, val, thread_sum_sq);
  }
  float simd_total = simd_sum(thread_sum_sq);
  threadgroup float shared_simd[32];
  if (simd_lane_id == 0) shared_simd[simd_group_id] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint num_simd_groups = (block_dim + 31) >> 5;
  if (simd_group_id == 0) {
    float val = (simd_lane_id < num_simd_groups) ? shared_simd[simd_lane_id] : 0.0f;
    float total_sq = simd_sum(val);
    if (simd_lane_id == 0) shared_simd[0] = rsqrt((total_sq / float(D)) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = shared_simd[0];
  for (uint i = tid; i < D; i += block_dim) {
    norm_out[i] = x_out[i] * scale * weight[i];
  }
}

kernel void embed_rmsnorm_forward_half_kernel(
    device const long *idx [[buffer(0)]],
    device const half *wte [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device half *x_out [[buffer(3)]],
    device half *norm_out [[buffer(4)]],
    constant uint &D [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  uint tid = tid_in_tg.x;
  uint block_dim = tg_size.x;
  long token_id = idx[0];
  device const half *row_wte = wte + token_id * long(D);

  if (D == 768 && block_dim == 64) {
    device const half4 *wte4 = (device const half4 *)row_wte;
    device const half4 *w4 = (device const half4 *)weight;
    device half4 *x4 = (device half4 *)x_out;
    device half4 *y4 = (device half4 *)norm_out;
    half4 h0 = wte4[tid];
    half4 h1 = wte4[tid + 64];
    half4 h2 = wte4[tid + 128];
    x4[tid] = h0;
    x4[tid + 64] = h1;
    x4[tid + 128] = h2;
    float4 v0 = float4(h0);
    float4 v1 = float4(h1);
    float4 v2 = float4(h2);
    float thread_sum_sq = dot(v0, v0) + dot(v1, v1) + dot(v2, v2);
    float simd_total = simd_sum(thread_sum_sq);
    threadgroup float shared_simd[2];
    if (simd_lane_id == 0) shared_simd[simd_group_id] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0 && simd_lane_id == 0) {
      float total_sq = shared_simd[0] + shared_simd[1];
      shared_simd[0] = rsqrt((total_sq * (1.0f / 768.0f)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared_simd[0];
    y4[tid]       = half4(v0 * scale * float4(w4[tid]));
    y4[tid + 64]  = half4(v1 * scale * float4(w4[tid + 64]));
    y4[tid + 128] = half4(v2 * scale * float4(w4[tid + 128]));
    return;
  }

  float thread_sum_sq = 0.0f;
  for (uint i = tid; i < D; i += block_dim) {
    half h = row_wte[i];
    x_out[i] = h;
    float val = float(h);
    thread_sum_sq = fma(val, val, thread_sum_sq);
  }
  float simd_total = simd_sum(thread_sum_sq);
  threadgroup float shared_simd[32];
  if (simd_lane_id == 0) shared_simd[simd_group_id] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint num_simd_groups = (block_dim + 31) >> 5;
  if (simd_group_id == 0) {
    float val = (simd_lane_id < num_simd_groups) ? shared_simd[simd_lane_id] : 0.0f;
    float total_sq = simd_sum(val);
    if (simd_lane_id == 0) shared_simd[0] = rsqrt((total_sq / float(D)) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = shared_simd[0];
  for (uint i = tid; i < D; i += block_dim) {
    norm_out[i] = half(float(x_out[i]) * scale * float(weight[i]));
  }
}

// ----------------------------------------------------------------------------
// 1E. Fast Single-Threadgroup Argmax Kernel over Vocabulary Logits (FP16 & FP32)
// Replaces 340+ us PyTorch MPSGraph argmax with a 4 us SIMD-group reduction
// ----------------------------------------------------------------------------
kernel void argmax_half_kernel(
    device const half *logits [[buffer(0)]],
    device long *out_idx [[buffer(1)]],
    constant uint &V [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_dim [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  float best_val = -1e38f;
  uint best_idx = 0;

  uint V_vec4 = V >> 2;
  device const half4 *logits4 = (device const half4 *)logits;
  for (uint i = tid; i < V_vec4; i += block_dim) {
    half4 h4 = logits4[i];
    uint base = i << 2;
    float v0 = float(h4.x), v1 = float(h4.y), v2 = float(h4.z), v3 = float(h4.w);
    if (v0 > best_val) { best_val = v0; best_idx = base; }
    if (v1 > best_val) { best_val = v1; best_idx = base + 1; }
    if (v2 > best_val) { best_val = v2; best_idx = base + 2; }
    if (v3 > best_val) { best_val = v3; best_idx = base + 3; }
  }
  for (uint i = (V_vec4 << 2) + tid; i < V; i += block_dim) {
    float v = float(logits[i]);
    if (v > best_val) { best_val = v; best_idx = i; }
  }

  threadgroup float shared_val[256];
  threadgroup uint shared_idx[256];
  shared_val[tid] = best_val;
  shared_idx[tid] = best_idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = block_dim >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float other_v = shared_val[tid + stride];
      uint other_i = shared_idx[tid + stride];
      if (other_v > shared_val[tid] || (other_v == shared_val[tid] && other_i < shared_idx[tid])) {
        shared_val[tid] = other_v;
        shared_idx[tid] = other_i;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    out_idx[0] = long(shared_idx[0]);
  }
}

// ----------------------------------------------------------------------------
// 1C. Fused In-Place Residual Add + RMSNorm Forward Kernels (FP32 & FP16)
// Computes:
//   x_inout = x_inout + residual   (in-place residual stream update)
//   y       = (x_inout / sqrt(mean(x_inout^2) + eps)) * weight
// ----------------------------------------------------------------------------
kernel void add_rmsnorm_forward_kernel(
    device float *x_inout [[buffer(0)]],       // In/Out Residual: (N_rows, D)
    device const float *residual [[buffer(1)]],// Residual delta:  (N_rows, D)
    device const float *weight [[buffer(2)]],  // RMSNorm weight:  (D,)
    device float *y [[buffer(3)]],             // Normalized Out:  (N_rows, D)
    constant uint &D [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  uint row_idx = tgpig.x;
  uint tid = tid_in_tg.x;
  uint block_dim = tg_size.x;

  device float *row_x = x_inout + row_idx * D;
  device const float *row_r = residual + row_idx * D;
  device float *row_y = y + row_idx * D;

  float thread_sum_sq = 0.0f;
  if ((D & 3) == 0) {
    uint D_vec4 = D >> 2;
    device float4 *row_x4 = (device float4 *)row_x;
    device const float4 *row_r4 = (device const float4 *)row_r;
    for (uint i = tid; i < D_vec4; i += block_dim) {
      float4 v = row_x4[i] + row_r4[i];
      row_x4[i] = v;
      thread_sum_sq = fma(v.x, v.x, thread_sum_sq);
      thread_sum_sq = fma(v.y, v.y, thread_sum_sq);
      thread_sum_sq = fma(v.z, v.z, thread_sum_sq);
      thread_sum_sq = fma(v.w, v.w, thread_sum_sq);
    }
  } else {
    for (uint i = tid; i < D; i += block_dim) {
      float val = row_x[i] + row_r[i];
      row_x[i] = val;
      thread_sum_sq = fma(val, val, thread_sum_sq);
    }
  }

  float simd_total = simd_sum(thread_sum_sq);
  threadgroup float shared_simd[32];
  if (simd_lane_id == 0) {
    shared_simd[simd_group_id] = simd_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint num_simd_groups = (block_dim + 31) >> 5;
  if (simd_group_id == 0) {
    float val = (simd_lane_id < num_simd_groups) ? shared_simd[simd_lane_id] : 0.0f;
    float total_sq = simd_sum(val);
    if (simd_lane_id == 0) {
      shared_simd[0] = rsqrt((total_sq / float(D)) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = shared_simd[0];

  if ((D & 3) == 0) {
    uint D_vec4 = D >> 2;
    device const float4 *row_x4 = (device const float4 *)row_x;
    device const float4 *w4 = (device const float4 *)weight;
    device float4 *row_y4 = (device float4 *)row_y;
    for (uint i = tid; i < D_vec4; i += block_dim) {
      row_y4[i] = row_x4[i] * scale * w4[i];
    }
  } else {
    for (uint i = tid; i < D; i += block_dim) {
      row_y[i] = row_x[i] * scale * weight[i];
    }
  }
}

kernel void add_rmsnorm_forward_half_kernel(
    device half *x_inout [[buffer(0)]],
    device const half *residual [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &D [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  uint row_idx = tgpig.x;
  uint tid = tid_in_tg.x;
  uint block_dim = tg_size.x;

  device half *row_x = x_inout + row_idx * D;
  device const half *row_r = residual + row_idx * D;
  device half *row_y = y + row_idx * D;

  if (D == 768 && block_dim == 64) {
    device half4 *row_x4 = (device half4 *)row_x;
    device const half4 *row_r4 = (device const half4 *)row_r;
    device const half4 *w4 = (device const half4 *)weight;
    device half4 *row_y4 = (device half4 *)row_y;
    half4 h0 = row_x4[tid] + row_r4[tid];
    half4 h1 = row_x4[tid + 64] + row_r4[tid + 64];
    half4 h2 = row_x4[tid + 128] + row_r4[tid + 128];
    row_x4[tid] = h0;
    row_x4[tid + 64] = h1;
    row_x4[tid + 128] = h2;
    float4 v0 = float4(h0);
    float4 v1 = float4(h1);
    float4 v2 = float4(h2);
    float thread_sum_sq = dot(v0, v0) + dot(v1, v1) + dot(v2, v2);
    float simd_total = simd_sum(thread_sum_sq);
    threadgroup float shared_simd[2];
    if (simd_lane_id == 0) {
      shared_simd[simd_group_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0 && simd_lane_id == 0) {
      float total_sq = shared_simd[0] + shared_simd[1];
      shared_simd[0] = rsqrt((total_sq * (1.0f / 768.0f)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared_simd[0];
    row_y4[tid]       = half4(v0 * scale * float4(w4[tid]));
    row_y4[tid + 64]  = half4(v1 * scale * float4(w4[tid + 64]));
    row_y4[tid + 128] = half4(v2 * scale * float4(w4[tid + 128]));
    return;
  }

  float thread_sum_sq = 0.0f;
  if ((D & 3) == 0) {
    uint D_vec4 = D >> 2;
    device half4 *row_x4 = (device half4 *)row_x;
    device const half4 *row_r4 = (device const half4 *)row_r;
    for (uint i = tid; i < D_vec4; i += block_dim) {
      half4 sum_h4 = row_x4[i] + row_r4[i];
      row_x4[i] = sum_h4;
      float4 v = float4(sum_h4);
      thread_sum_sq = fma(v.x, v.x, thread_sum_sq);
      thread_sum_sq = fma(v.y, v.y, thread_sum_sq);
      thread_sum_sq = fma(v.z, v.z, thread_sum_sq);
      thread_sum_sq = fma(v.w, v.w, thread_sum_sq);
    }
  } else {
    for (uint i = tid; i < D; i += block_dim) {
      half sum_h = row_x[i] + row_r[i];
      row_x[i] = sum_h;
      float val = float(sum_h);
      thread_sum_sq = fma(val, val, thread_sum_sq);
    }
  }

  float simd_total = simd_sum(thread_sum_sq);
  threadgroup float shared_simd[32];
  if (simd_lane_id == 0) {
    shared_simd[simd_group_id] = simd_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint num_simd_groups = (block_dim + 31) >> 5;
  if (simd_group_id == 0) {
    float val = (simd_lane_id < num_simd_groups) ? shared_simd[simd_lane_id] : 0.0f;
    float total_sq = simd_sum(val);
    if (simd_lane_id == 0) {
      shared_simd[0] = rsqrt((total_sq / float(D)) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = shared_simd[0];

  if ((D & 3) == 0) {
    uint D_vec4 = D >> 2;
    device const half4 *row_x4 = (device const half4 *)row_x;
    device const half4 *w4 = (device const half4 *)weight;
    device half4 *row_y4 = (device half4 *)row_y;
    for (uint i = tid; i < D_vec4; i += block_dim) {
      row_y4[i] = half4(float4(row_x4[i]) * scale * float4(w4[i]));
    }
  } else {
    for (uint i = tid; i < D; i += block_dim) {
      row_y[i] = half(float(row_x[i]) * scale * float(weight[i]));
    }
  }
}

// ----------------------------------------------------------------------------
// 2A. Fused SwiGLU Forward Kernel (FP32 - Vectorized float4)
// Computes: y = (x_gate / (1 + exp(-x_gate))) * x_up
// ----------------------------------------------------------------------------
kernel void swiglu_forward_kernel(
    device const float4 *x_gate [[buffer(0)]],
    device const float4 *x_up [[buffer(1)]],
    device float4 *y [[buffer(2)]],
    constant uint &num_vec4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= num_vec4)
    return;

  float4 g = x_gate[gid];
  float4 u = x_up[gid];

  float4 sig = 1.0f / (1.0f + fast::exp(-g));
  y[gid] = (g * sig) * u;
}

// ----------------------------------------------------------------------------
// 2B. Fused SwiGLU Forward Kernel (FP16 - Vectorized half4)
// ----------------------------------------------------------------------------
kernel void swiglu_forward_half_kernel(
    device const half4 *x_gate [[buffer(0)]],
    device const half4 *x_up [[buffer(1)]],
    device half4 *y [[buffer(2)]],
    constant uint &num_vec4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= num_vec4)
    return;

  float4 g = float4(x_gate[gid]);
  float4 u = float4(x_up[gid]);

  float4 sig = 1.0f / (1.0f + fast::exp(-g));
  y[gid] = half4((g * sig) * u);
}

// ----------------------------------------------------------------------------
// 2C. Fused Single-Buffer SwiGLU Gate+Up Kernel (FP32 & FP16 - Vectorized)
// Input gate_up has shape (M, 2 * D); Output y has shape (M, D)
// ----------------------------------------------------------------------------
kernel void swiglu_gate_up_kernel(
    device const float4 *gate_up [[buffer(0)]],
    device float4 *y [[buffer(1)]],
    constant uint &D_vec4 [[buffer(2)]],
    constant uint &total_vec4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= total_vec4)
    return;
  uint row = gid / D_vec4;
  uint col = gid % D_vec4;
  uint base = row * (D_vec4 << 1) + col;
  float4 g = gate_up[base];
  float4 u = gate_up[base + D_vec4];
  float4 sig = 1.0f / (1.0f + fast::exp(-g));
  y[gid] = (g * sig) * u;
}

kernel void swiglu_gate_up_half_kernel(
    device const half4 *gate_up [[buffer(0)]],
    device half4 *y [[buffer(1)]],
    constant uint &D_vec4 [[buffer(2)]],
    constant uint &total_vec4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= total_vec4)
    return;
  uint row = gid / D_vec4;
  uint col = gid % D_vec4;
  uint base = row * (D_vec4 << 1) + col;
  float4 g = float4(gate_up[base]);
  float4 u = float4(gate_up[base + D_vec4]);
  float4 sig = 1.0f / (1.0f + fast::exp(-g));
  y[gid] = half4((g * sig) * u);
}

// ----------------------------------------------------------------------------
// 3. Fused SwiGLU Backward Kernel
// ----------------------------------------------------------------------------
kernel void swiglu_backward_kernel(
    device const float *grad_y [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device const float *up [[buffer(2)]],
    device float *grad_gate [[buffer(3)]],
    device float *grad_up [[buffer(4)]],
    constant uint &num_elements [[buffer(5)]],
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

  float4 sig_g = 1.0f / (1.0f + fast::exp(-g));
  float4 silu_g = g * sig_g;

  gu_vec[tid] = gy * silu_g;

  float4 d_silu_dg = sig_g * (1.0f + g * (1.0f - sig_g));
  gg_vec[tid] = gy * u * d_silu_dg;
}

// ----------------------------------------------------------------------------
// 4. RMSNorm Backward Kernels (with simd_sum)
// ----------------------------------------------------------------------------
kernel void rmsnorm_backward_kernel(
    device const float *grad_y [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device const float *rsqrt_buf [[buffer(3)]],
    device float *grad_x [[buffer(4)]],
    constant uint &D [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_dim [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row_idx = bid;
    device const float *row_gy = grad_y + row_idx * D;
    device const float *row_x = x + row_idx * D;
    device float *row_gx = grad_x + row_idx * D;
    float rsqrt_val = rsqrt_buf[row_idx];

    float thread_inner = 0.0f;
    for (uint i = tid; i < D; i += block_dim) {
        thread_inner = fma(row_gy[i] * weight[i], row_x[i], thread_inner);
    }

    float simd_inner = simd_sum(thread_inner);
    threadgroup float shared_inner[32];
    if (simd_lane_id == 0) {
        shared_inner[simd_group_id] = simd_inner;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (block_dim + 31) >> 5;
    if (simd_group_id == 0) {
        float val = (simd_lane_id < num_simd_groups) ? shared_inner[simd_lane_id] : 0.0f;
        float inner_sum = simd_sum(val);
        if (simd_lane_id == 0) {
            shared_inner[0] = (inner_sum / (float)D) * (rsqrt_val * rsqrt_val * rsqrt_val);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared_inner[0];

    for (uint i = tid; i < D; i += block_dim) {
        row_gx[i] = (row_gy[i] * weight[i] * rsqrt_val) - (row_x[i] * scale);
    }
}

kernel void rmsnorm_backward_weight_kernel(
    device const float *grad_y [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *rsqrt_buf [[buffer(2)]],
    device float *grad_w [[buffer(3)]],
    constant uint &D [[buffer(4)]],
    constant uint &num_rows [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= D) return;

    float gw_val = 0.0f;
    for (uint r = 0; r < num_rows; ++r) {
        gw_val = fma(grad_y[r * D + tid] * x[r * D + tid], rsqrt_buf[r], gw_val);
    }
    grad_w[tid] = gw_val;
}

// ----------------------------------------------------------------------------
// 5A. Rotary Position Embeddings (RoPE) Kernel (FP32)
// ----------------------------------------------------------------------------
kernel void rope_kernel(
    device const float2 *x [[buffer(0)]],
    device const float2 *freqs_cis [[buffer(1)]],
    device float2 *out [[buffer(2)]],
    constant uint &T [[buffer(3)]],
    constant uint &half_head_dim [[buffer(4)]],
    constant uint &forward [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    constant uint &num_elements [[buffer(6)]]
) {
    if (tid >= num_elements) return;

    uint hd_idx = tid % half_head_dim;
    uint t_idx = (tid / half_head_dim) % T;

    float2 x_val = x[tid];
    float2 f_val = freqs_cis[t_idx * half_head_dim + hd_idx];

    if (forward == 0) {
        f_val.y = -f_val.y;
    }

    float2 res;
    res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
    res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);

    out[tid] = res;
}

// ----------------------------------------------------------------------------
// 5B. Rotary Position Embeddings (RoPE) Kernel (FP16)
// ----------------------------------------------------------------------------
kernel void rope_half_kernel(
    device const half2 *x [[buffer(0)]],
    device const float2 *freqs_cis [[buffer(1)]],
    device half2 *out [[buffer(2)]],
    constant uint &T [[buffer(3)]],
    constant uint &half_head_dim [[buffer(4)]],
    constant uint &forward [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    constant uint &num_elements [[buffer(6)]]
) {
    if (tid >= num_elements) return;

    uint hd_idx = tid % half_head_dim;
    uint t_idx = (tid / half_head_dim) % T;

    float2 x_val = float2(x[tid]);
    float2 f_val = freqs_cis[t_idx * half_head_dim + hd_idx];

    if (forward == 0) {
        f_val.y = -f_val.y;
    }

    float2 res;
    res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
    res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);

    out[tid] = half2(res);
}

// ----------------------------------------------------------------------------
// 5C. Fused T=1 Decode RoPE (In-Place Q + Direct K/V Cache Write) (FP32 & FP16)
// Processes flat qkv of shape (1, 1, q_dim + 2 * kv_dim) in a single dispatch:
//   - Threads [0 .. q_pairs-1]: Rotate Q in-place in qkv
//   - Threads [q_pairs .. q_pairs+kv_pairs-1]: Rotate K and write directly to k_cache
//   - Threads [q_pairs+kv_pairs .. q_pairs+2*kv_pairs-1]: Copy V directly to v_cache
// ----------------------------------------------------------------------------
kernel void rope_kv_cache_write_kernel(
    device float2 *qkv [[buffer(0)]],
    device const float2 *freqs_cis [[buffer(1)]],
    device float2 *k_cache [[buffer(2)]],
    device float2 *v_cache [[buffer(3)]],
    constant uint &q_pairs [[buffer(4)]],
    constant uint &kv_pairs [[buffer(5)]],
    constant uint &half_head_dim [[buffer(6)]],
    constant uint &max_seq_len [[buffer(7)]],
    constant uint &write_pos [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
    uint total_pairs = q_pairs + (kv_pairs << 1);
    if (tid >= total_pairs) return;

    uint hd_idx = tid % half_head_dim;
    if (tid < q_pairs) {
        float2 x_val = qkv[tid];
        float2 f_val = freqs_cis[hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        qkv[tid] = res;
    } else if (tid < q_pairs + kv_pairs) {
        uint k_tid = tid - q_pairs;
        uint kv_head = k_tid / half_head_dim;
        float2 x_val = qkv[tid];
        float2 f_val = freqs_cis[hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        uint dst_idx = (kv_head * max_seq_len + write_pos) * half_head_dim + hd_idx;
        k_cache[dst_idx] = res;
    } else {
        uint v_tid = tid - (q_pairs + kv_pairs);
        uint kv_head = v_tid / half_head_dim;
        uint dst_idx = (kv_head * max_seq_len + write_pos) * half_head_dim + hd_idx;
        v_cache[dst_idx] = qkv[tid];
    }
}

kernel void rope_kv_cache_write_half_kernel(
    device half2 *qkv [[buffer(0)]],
    device const float2 *freqs_cis [[buffer(1)]],
    device half2 *k_cache [[buffer(2)]],
    device half2 *v_cache [[buffer(3)]],
    constant uint &q_pairs [[buffer(4)]],
    constant uint &kv_pairs [[buffer(5)]],
    constant uint &half_head_dim [[buffer(6)]],
    constant uint &max_seq_len [[buffer(7)]],
    constant uint &write_pos [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
    uint total_pairs = q_pairs + (kv_pairs << 1);
    if (tid >= total_pairs) return;

    uint hd_idx = tid % half_head_dim;
    if (tid < q_pairs) {
        float2 x_val = float2(qkv[tid]);
        float2 f_val = freqs_cis[hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        qkv[tid] = half2(res);
    } else if (tid < q_pairs + kv_pairs) {
        uint k_tid = tid - q_pairs;
        uint kv_head = k_tid / half_head_dim;
        float2 x_val = float2(qkv[tid]);
        float2 f_val = freqs_cis[hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        uint dst_idx = (kv_head * max_seq_len + write_pos) * half_head_dim + hd_idx;
        k_cache[dst_idx] = half2(res);
    } else {
        uint v_tid = tid - (q_pairs + kv_pairs);
        uint kv_head = v_tid / half_head_dim;
        uint dst_idx = (kv_head * max_seq_len + write_pos) * half_head_dim + hd_idx;
        v_cache[dst_idx] = qkv[tid];
    }
}

// ----------------------------------------------------------------------------
// 6. Fused Cross Entropy (with simd_max and simd_sum)
// ----------------------------------------------------------------------------
kernel void cross_entropy_forward_kernel(
    device const float *logits [[buffer(0)]],
    device const int *targets [[buffer(1)]],
    device float *losses [[buffer(2)]],
    constant uint &V [[buffer(3)]],
    constant int &ignore_index [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_dim [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row_idx = bid;
    int target_class = targets[row_idx];

    if (target_class == ignore_index) {
        if (tid == 0) losses[row_idx] = 0.0f;
        return;
    }

    device const float *row_logits = logits + row_idx * V;

    // Online 1-pass max + sum using online normalizer calculation!
    float local_max = -1e38f;
    float local_sum = 0.0f;
    for (uint i = tid; i < V; i += block_dim) {
        float val = row_logits[i];
        if (val > local_max) {
            local_sum = local_sum * fast::exp(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += fast::exp(val - local_max);
        }
    }

    // SIMD-group max reduction
    float sg_max = simd_max(local_max);
    float sg_sum = simd_sum(local_sum * fast::exp(local_max - sg_max));

    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];
    if (simd_lane_id == 0) {
        shared_max[simd_group_id] = sg_max;
        shared_sum[simd_group_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (block_dim + 31) >> 5;
    if (simd_group_id == 0) {
        float m_val = (simd_lane_id < num_simd_groups) ? shared_max[simd_lane_id] : -1e38f;
        float s_val = (simd_lane_id < num_simd_groups) ? shared_sum[simd_lane_id] : 0.0f;
        float row_max = simd_max(m_val);
        float row_sum = simd_sum(s_val * fast::exp(m_val - row_max));
        if (simd_lane_id == 0) {
            float lse = row_max + fast::log(row_sum);
            losses[row_idx] = lse - row_logits[target_class];
        }
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
    uint block_dim [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
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

    // Online single-pass max + sum over V
    float local_max = -1e38f;
    float local_sum = 0.0f;
    for (uint i = tid; i < V; i += block_dim) {
        float val = row_logits[i];
        if (val > local_max) {
            local_sum = local_sum * fast::exp(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += fast::exp(val - local_max);
        }
    }

    float sg_max = simd_max(local_max);
    float sg_sum = simd_sum(local_sum * fast::exp(local_max - sg_max));

    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];
    if (simd_lane_id == 0) {
        shared_max[simd_group_id] = sg_max;
        shared_sum[simd_group_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (block_dim + 31) >> 5;
    if (simd_group_id == 0) {
        float m_val = (simd_lane_id < num_simd_groups) ? shared_max[simd_lane_id] : -1e38f;
        float s_val = (simd_lane_id < num_simd_groups) ? shared_sum[simd_lane_id] : 0.0f;
        float r_max = simd_max(m_val);
        float r_sum = simd_sum(s_val * fast::exp(m_val - r_max));
        if (simd_lane_id == 0) {
            shared_max[0] = r_max;
            shared_sum[0] = 1.0f / r_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float row_max = shared_max[0];
    float inv_sum = shared_sum[0];
    for (uint i = tid; i < V; i += block_dim) {
        float prob = fast::exp(row_logits[i] - row_max) * inv_sum;
        float grad = prob - ((int)i == target_class ? 1.0f : 0.0f);
        row_grad[i] = grad * go;
    }
}

// ----------------------------------------------------------------------------
// 7. Flash Attention Forward (Tiled SRAM Online Softmax)
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
    uint2 tid_2d [[thread_position_in_threadgroup]],
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
    uint tid = tid_2d.x;
    uint q_idx = q_start + tid;

    bool valid_q = q_idx < seq_len_q;

    float m_i = -1e38f;
    float l_i = 0.0f;
    float O_i[128];
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

                float score = 0.0f;
                for (uint d = 0; d < head_dim; d++) {
                    score = fma(Q_i[d], K_shared[j * head_dim + d], score);
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

// ----------------------------------------------------------------------------
// 6A. Quantized Matrix-Vector Multiply Q4_G64 (FP16 Activations, 4-bit Weights)
// Input x: (1, K) half
// Weight w_packed: (N, K / 8) uint32 (8 x 4-bit nibbles per word)
// ScalesBiases: (N, K / 64) half2 (scale, bias) per 64 weights
// Output y: (1, N) half
// ----------------------------------------------------------------------------
kernel void qmv_q4_g64_half_kernel(
    device const half *x [[buffer(0)]],
    device const uint *w_packed [[buffer(1)]],
    device const half2 *scales_biases [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    // 2 SIMD-groups per threadgroup -> 8 output rows per threadgroup
    uint row_base = (tgpig.x * 2 + simd_group_id) * 4;
    if (row_base >= N) return;

    uint K_packed = K >> 3;       // K / 8 uint32 words per row
    uint K_groups = K >> 6;       // K / 64 groups per row

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Stride across K_packed in steps of 32 lanes
    for (uint idx = simd_lane_id; idx < K_packed; idx += 32) {
        uint k_elem = idx << 3;   // Starting element index (multiple of 8)
        uint g_idx = idx >> 3;    // Group index (every 8 uint32 words = 64 elements)

        // Load 8 FP16 activations as two half4 vectors into registers
        half4 x_lo = *(device const half4 *)(x + k_elem);
        half4 x_hi = *(device const half4 *)(x + k_elem + 4);
        float x_sum = float(x_lo.x + x_lo.y + x_lo.z + x_lo.w + x_hi.x + x_hi.y + x_hi.z + x_hi.w);

        // Unroll across 4 output rows sharing x_lo and x_hi in registers
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            uint row = row_base + r;
            if (row < N) {
                uint packed = w_packed[row * K_packed + idx];
                half2 sb = scales_biases[row * K_groups + g_idx];
                float scale = float(sb.x);
                float bias  = float(sb.y);

                float4 q_lo = float4(packed & 0xFu, (packed >> 4) & 0xFu, (packed >> 8) & 0xFu, (packed >> 12) & 0xFu);
                float4 q_hi = float4((packed >> 16) & 0xFu, (packed >> 20) & 0xFu, (packed >> 24) & 0xFu, (packed >> 28) & 0xFu);

                float d_prod = dot(float4(x_lo), q_lo) + dot(float4(x_hi), q_hi);
                float val = fma(d_prod, scale, bias * x_sum);
                if (r == 0) acc0 += val;
                else if (r == 1) acc1 += val;
                else if (r == 2) acc2 += val;
                else acc3 += val;
            }
        }
    }

    // Hardware single-cycle 32-lane SIMD reduction
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane_id == 0) {
        if (row_base + 0 < N) y[row_base + 0] = half(acc0);
        if (row_base + 1 < N) y[row_base + 1] = half(acc1);
        if (row_base + 2 < N) y[row_base + 2] = half(acc2);
        if (row_base + 3 < N) y[row_base + 3] = half(acc3);
    }
}

// ----------------------------------------------------------------------------
// 6B. Quantized Matrix-Vector Multiply Q8_G64 (FP16 Activations, 8-bit Weights)
// Input x: (1, K) half
// Weight w_packed: (N, K / 4) uint32 (4 x 8-bit bytes per word)
// ScalesBiases: (N, K / 64) half2 (scale, bias) per 64 weights
// Output y: (1, N) half
// ----------------------------------------------------------------------------
kernel void qmv_q8_g64_half_kernel(
    device const half *x [[buffer(0)]],
    device const uint *w_packed [[buffer(1)]],
    device const half2 *scales_biases [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = (tgpig.x * 2 + simd_group_id) * 4;
    if (row_base >= N) return;

    uint K_packed = K >> 2;       // K / 4 uint32 words per row
    uint K_groups = K >> 6;       // K / 64 groups per row

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint idx = simd_lane_id; idx < K_packed; idx += 32) {
        uint k_elem = idx << 2;   // Starting element index (multiple of 4)
        uint g_idx = idx >> 4;    // Group index (every 16 uint32 words = 64 elements)

        half4 x_vec = *(device const half4 *)(x + k_elem);
        float x_sum = float(x_vec.x + x_vec.y + x_vec.z + x_vec.w);

        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            uint row = row_base + r;
            if (row < N) {
                uint packed = w_packed[row * K_packed + idx];
                half2 sb = scales_biases[row * K_groups + g_idx];
                float scale = float(sb.x);
                float bias  = float(sb.y);

                float4 q = float4(packed & 0xFFu, (packed >> 8) & 0xFFu, (packed >> 16) & 0xFFu, (packed >> 24) & 0xFFu);
                float d_prod = dot(float4(x_vec), q);
                float val = fma(d_prod, scale, bias * x_sum);
                if (r == 0) acc0 += val;
                else if (r == 1) acc1 += val;
                else if (r == 2) acc2 += val;
                else acc3 += val;
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane_id == 0) {
        if (row_base + 0 < N) y[row_base + 0] = half(acc0);
        if (row_base + 1 < N) y[row_base + 1] = half(acc1);
        if (row_base + 2 < N) y[row_base + 2] = half(acc2);
        if (row_base + 3 < N) y[row_base + 3] = half(acc3);
    }
}
