#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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
// 1D-2. Batched Embedding Lookup Kernel (FP16)
// Copies D-dimensional embeddings for T tokens (T in 1..8) into x_out directly
// from unified shared memory buffer idx without PyTorch index_select dispatch.
// ----------------------------------------------------------------------------
kernel void embed_multi_half_kernel(
    device const long *idx [[buffer(0)]],
    device const half *wte [[buffer(1)]],
    device half *x_out [[buffer(2)]],
    constant uint &D [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]) {
  uint row_idx = tgpig.x;
  uint tid = tid_in_tg.x;
  uint block_dim = tg_size.x;

  long token_id = idx[row_idx];
  device const half *row_wte = wte + token_id * long(D);
  device half *row_x = x_out + row_idx * long(D);

  uint D_vec4 = D >> 2;
  device const half4 *wte4 = (device const half4 *)row_wte;
  device half4 *x4 = (device half4 *)row_x;
  for (uint i = tid; i < D_vec4; i += block_dim) {
    x4[i] = wte4[i];
  }
  for (uint i = (D_vec4 << 2) + tid; i < D; i += block_dim) {
    row_x[i] = row_wte[i];
  }
}

// ----------------------------------------------------------------------------
// 1D-3. On-The-Fly Q8_G64 Embedding Gather Kernel (Tied & Quantized Token Embeddings)
// Replaces 525 MB uncompressed FP16 wte table with 262 MB Q8_G64 table
// Unpacks 8-bit quantized token embeddings directly into FP16 row buffer
// ----------------------------------------------------------------------------
kernel void embed_q8_g64_gather_half_kernel(
    device const long *idx [[buffer(0)]],
    device const uint *w_packed [[buffer(1)]],
    device const half2 *scales_biases [[buffer(2)]],
    device half *x_out [[buffer(3)]],
    constant uint &D [[buffer(4)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]) {
  uint row_idx = tgpig.x;
  uint tid = tid_in_tg.x;
  uint block_dim = tg_size.x;

  long token_id = idx[row_idx];
  uint D_packed = D >> 2;
  uint D_groups = D >> 6;

  device const uint *row_w = w_packed + token_id * long(D_packed);
  device const half2 *row_sb = scales_biases + token_id * long(D_groups);
  device half *row_x = x_out + row_idx * long(D);
  device half4 *x4 = (device half4 *)row_x;

  for (uint i = tid; i < D_packed; i += block_dim) {
    uint g = i >> 4;
    half2 sb = row_sb[g];
    uchar4 q = as_type<uchar4>(row_w[i]);
    half4 h = half4(
        fma(half(q.x), sb.x, sb.y),
        fma(half(q.y), sb.x, sb.y),
        fma(half(q.z), sb.x, sb.y),
        fma(half(q.w), sb.x, sb.y)
    );
    x4[i] = h;
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
// 1E-2. Multi-Row In-Threadgroup Argmax Kernel over Vocabulary Logits (FP16)
// Reduces V vocabulary logits for T tokens (T in 1..8) in parallel on Metal GPU.
// Directly writes token IDs into unified shared memory out_idx without logit copying.
// ----------------------------------------------------------------------------
kernel void argmax_multi_half_kernel(
    device const half *logits [[buffer(0)]],
    device long *out_idx [[buffer(1)]],
    constant uint &V [[buffer(2)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]) {
  uint row_idx = tgpig.x;
  uint tid = tid_in_tg.x;
  uint block_dim = tg_size.x;

  device const half *row_logits = logits + row_idx * V;
  float best_val = -1e38f;
  uint best_idx = 0;

  uint V_vec4 = V >> 2;
  device const half4 *logits4 = (device const half4 *)row_logits;
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
    float v = float(row_logits[i]);
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
    out_idx[row_idx] = long(shared_idx[0]);
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
    device const half2 *qkv_bias [[buffer(9)]],
    constant uint &has_bias [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
    uint total_pairs = q_pairs + (kv_pairs << 1);
    if (tid >= total_pairs) return;

    half2 raw_val = qkv[tid];
    if (has_bias != 0u) {
        raw_val += qkv_bias[tid];
    }

    uint hd_idx = tid % half_head_dim;
    if (tid < q_pairs) {
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        qkv[tid] = half2(res);
    } else if (tid < q_pairs + kv_pairs) {
        uint k_tid = tid - q_pairs;
        uint kv_head = k_tid / half_head_dim;
        float2 x_val = float2(raw_val);
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
        v_cache[dst_idx] = raw_val;
    }
}

// ----------------------------------------------------------------------------
// Phase 3 Task 3.1: In-Flight FP8 & INT4 KV-Cache Quantization
// ----------------------------------------------------------------------------

// Converts float to 8-bit float (OCP FP8 E4M3FN: 1 sign, 4 exp, 3 mantissa, max finite 448.0)
inline uchar float_to_fp8_e4m3fn(float val) {
    uint u = as_type<uint>(val);
    uint sign = (u >> 24) & 0x80u;
    uint abs_u = u & 0x7FFFFFFFu;
    float abs_val = as_type<float>(abs_u);

    if (abs_val >= 448.0f) {
        return (uchar)(sign | 0x7Eu);
    }
    if (abs_val < 0.0009765625f) {
        return (uchar)(sign);
    }
    if (abs_val < 0.015625f) {
        uint mant = (uint)(abs_val * 512.0f + 0.5f);
        return (uchar)(sign | min(mant, 7u));
    }
    int exp = ((int)(abs_u >> 23)) - 120; // 127 - 7
    if (exp > 15) {
        return (uchar)(sign | 0x7Eu);
    }
    uint mant = (abs_u >> 20) & 7u;
    uint round_bit = (abs_u >> 19) & 1u;
    uint sticky = abs_u & 0x7FFFFu;
    if (round_bit && (sticky || (mant & 1u))) {
        mant++;
        if (mant == 8u) {
            mant = 0u;
            exp++;
            if (exp > 15) return (uchar)(sign | 0x7Eu);
        }
    }
    if (exp == 15 && mant > 6u) {
        mant = 6u;
    }
    return (uchar)(sign | ((uint)exp << 3) | mant);
}

// Converts 8-bit float (OCP FP8 E4M3FN) to half float via fast bitwise integer manipulation
inline half decode_fp8_e4m3fn(uchar b) {
    if ((b & 0x7Fu) == 0x7Fu) return 0.0h;
    uint exp = (b >> 3) & 0x0Fu;
    if (exp == 0u) {
        half sign = (b & 0x80u) ? -1.0h : 1.0h;
        return sign * (half(b & 0x07u) * 0.001953125h); // 2^-9
    }
    ushort sign_bit = ushort(b & 0x80u) << 8;
    ushort half_bits = sign_bit | (ushort((b & 0x78u) + (8u << 3)) << 7) | (ushort(b & 0x07u) << 7);
    return as_type<half>(half_bits);
}

// Packs a pair of half values into a single unsigned char with symmetric signed 4-bit [-8, 7]
inline uchar quant_pack_int4_pair(half2 val, half inv_scale) {
    float v0 = float(val.x) * float(inv_scale);
    float v1 = float(val.y) * float(inv_scale);
    int q0 = clamp(int(round(v0)), -8, 7);
    int q1 = clamp(int(round(v1)), -8, 7);
    return (uchar)((q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4));
}

// Unpacks an unsigned char containing 2 signed 4-bit values back to half2
inline half2 dequant_int4_packed(uchar packed, half scale) {
    int q0 = int((packed & 0x0Fu) ^ 0x08u) - 8;
    int q1 = int(((packed >> 4) & 0x0Fu) ^ 0x08u) - 8;
    return half2(half(q0) * scale, half(q1) * scale);
}

// In-Flight FP8 KV-Cache Direct Write Kernel (single token decode)
kernel void rope_kv_cache_write_fp8_kernel(
    device half2 *qkv [[buffer(0)]],
    device const float2 *freqs_cis [[buffer(1)]],
    device uchar2 *k_cache [[buffer(2)]],
    device uchar2 *v_cache [[buffer(3)]],
    device half *k_scales [[buffer(4)]],
    device half *v_scales [[buffer(5)]],
    constant uint &q_pairs [[buffer(6)]],
    constant uint &kv_pairs [[buffer(7)]],
    constant uint &half_head_dim [[buffer(8)]],
    constant uint &max_seq_len [[buffer(9)]],
    constant uint &write_pos [[buffer(10)]],
    device const half2 *qkv_bias [[buffer(11)]],
    constant uint &has_bias [[buffer(12)]],
    uint tid [[thread_position_in_grid]]) {
    uint total_pairs = q_pairs + (kv_pairs << 1);
    if (tid >= total_pairs) return;

    half2 raw_val = qkv[tid];
    if (has_bias != 0u) {
        raw_val += qkv_bias[tid];
    }

    uint hd_idx = tid % half_head_dim;
    if (tid < q_pairs) {
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        qkv[tid] = half2(res);
    } else if (tid < q_pairs + kv_pairs) {
        uint k_tid = tid - q_pairs;
        uint kv_head = k_tid / half_head_dim;
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        half2 res_h = half2(res);

        uint num_groups = max(1u, half_head_dim / 32u);
        uint group_idx = min(hd_idx / 32u, num_groups - 1u);
        uint lane_id = tid % 32u;
        float local_max = max(abs(float(res_h.x)), abs(float(res_h.y)));
        float group_max = simd_max(local_max);
        half scale = half(group_max / 448.0f);
        if (scale < 1e-8h) scale = 1.0h;
        half inv_scale = 1.0h / scale;

        uint scale_idx = (kv_head * max_seq_len + write_pos) * num_groups + group_idx;
        if (lane_id == 0u) {
            k_scales[scale_idx] = scale;
        }

        uchar2 packed = uchar2(
            float_to_fp8_e4m3fn(float(res_h.x) * float(inv_scale)),
            float_to_fp8_e4m3fn(float(res_h.y) * float(inv_scale))
        );
        uint dst_idx = (kv_head * max_seq_len + write_pos) * half_head_dim + hd_idx;
        k_cache[dst_idx] = packed;
    } else {
        uint v_tid = tid - (q_pairs + kv_pairs);
        uint kv_head = v_tid / half_head_dim;

        uint num_groups = max(1u, half_head_dim / 32u);
        uint group_idx = min(hd_idx / 32u, num_groups - 1u);
        uint lane_id = tid % 32u;
        float local_max = max(abs(float(raw_val.x)), abs(float(raw_val.y)));
        float group_max = simd_max(local_max);
        half scale = half(group_max / 448.0f);
        if (scale < 1e-8h) scale = 1.0h;
        half inv_scale = 1.0h / scale;

        uint scale_idx = (kv_head * max_seq_len + write_pos) * num_groups + group_idx;
        if (lane_id == 0u) {
            v_scales[scale_idx] = scale;
        }

        uchar2 packed = uchar2(
            float_to_fp8_e4m3fn(float(raw_val.x) * float(inv_scale)),
            float_to_fp8_e4m3fn(float(raw_val.y) * float(inv_scale))
        );
        uint dst_idx = (kv_head * max_seq_len + write_pos) * half_head_dim + hd_idx;
        v_cache[dst_idx] = packed;
    }
}

// In-Flight INT4 KV-Cache Direct Write Kernel (single token decode)
kernel void rope_kv_cache_write_int4_kernel(
    device half2 *qkv [[buffer(0)]],
    device const float2 *freqs_cis [[buffer(1)]],
    device uchar *k_cache [[buffer(2)]],
    device uchar *v_cache [[buffer(3)]],
    device half *k_scales [[buffer(4)]],
    device half *v_scales [[buffer(5)]],
    constant uint &q_pairs [[buffer(6)]],
    constant uint &kv_pairs [[buffer(7)]],
    constant uint &half_head_dim [[buffer(8)]],
    constant uint &max_seq_len [[buffer(9)]],
    constant uint &write_pos [[buffer(10)]],
    device const half2 *qkv_bias [[buffer(11)]],
    constant uint &has_bias [[buffer(12)]],
    uint tid [[thread_position_in_grid]]) {
    uint total_pairs = q_pairs + (kv_pairs << 1);
    if (tid >= total_pairs) return;

    half2 raw_val = qkv[tid];
    if (has_bias != 0u) {
        raw_val += qkv_bias[tid];
    }

    uint hd_idx = tid % half_head_dim;
    if (tid < q_pairs) {
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        qkv[tid] = half2(res);
    } else if (tid < q_pairs + kv_pairs) {
        uint k_tid = tid - q_pairs;
        uint kv_head = k_tid / half_head_dim;
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        half2 res_h = half2(res);

        uint num_groups = max(1u, half_head_dim / 32u);
        uint group_idx = min(hd_idx / 32u, num_groups - 1u);
        uint lane_id = tid % 32u;
        float local_max = max(abs(float(res_h.x)), abs(float(res_h.y)));
        float group_max = simd_max(local_max);
        half scale = half(group_max / 7.0f);
        if (scale < 1e-8h) scale = 1.0h;
        half inv_scale = 1.0h / scale;

        uint scale_idx = (kv_head * max_seq_len + write_pos) * num_groups + group_idx;
        if (lane_id == 0u) {
            k_scales[scale_idx] = scale;
        }

        uchar packed = quant_pack_int4_pair(res_h, inv_scale);
        uint dst_idx = (kv_head * max_seq_len + write_pos) * half_head_dim + hd_idx;
        k_cache[dst_idx] = packed;
    } else {
        uint v_tid = tid - (q_pairs + kv_pairs);
        uint kv_head = v_tid / half_head_dim;

        uint num_groups = max(1u, half_head_dim / 32u);
        uint group_idx = min(hd_idx / 32u, num_groups - 1u);
        uint lane_id = tid % 32u;
        float local_max = max(abs(float(raw_val.x)), abs(float(raw_val.y)));
        float group_max = simd_max(local_max);
        half scale = half(group_max / 7.0f);
        if (scale < 1e-8h) scale = 1.0h;
        half inv_scale = 1.0h / scale;

        uint scale_idx = (kv_head * max_seq_len + write_pos) * num_groups + group_idx;
        if (lane_id == 0u) {
            v_scales[scale_idx] = scale;
        }

        uchar packed = quant_pack_int4_pair(raw_val, inv_scale);
        uint dst_idx = (kv_head * max_seq_len + write_pos) * half_head_dim + hd_idx;
        v_cache[dst_idx] = packed;
    }
}

// In-Flight FP8 KV-Cache Direct Write Kernel (batched tree speculative verification)
kernel void rope_tree_kv_cache_write_fp8_kernel(
    device half2 *qkv [[buffer(0)]],
    device const float2 *freqs_cis [[buffer(1)]],
    device uchar2 *k_cache [[buffer(2)]],
    device uchar2 *v_cache [[buffer(3)]],
    device half *k_scales [[buffer(4)]],
    device half *v_scales [[buffer(5)]],
    constant const int *tree_depths [[buffer(6)]],
    constant uint &q_pairs [[buffer(7)]],
    constant uint &kv_pairs [[buffer(8)]],
    constant uint &half_head_dim [[buffer(9)]],
    constant uint &max_seq_len [[buffer(10)]],
    constant uint &max_rope_len [[buffer(11)]],
    constant uint &start_pos [[buffer(12)]],
    constant uint &write_pos [[buffer(13)]],
    constant uint &T [[buffer(14)]],
    device const half2 *qkv_bias [[buffer(15)]],
    constant uint &has_bias [[buffer(16)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint tid = gid.x;
    uint node_idx = gid.y;
    uint total_pairs = q_pairs + (kv_pairs << 1);
    if (tid >= total_pairs || node_idx >= T) return;

    device half2 *qkv_node = qkv + node_idx * total_pairs;
    half2 raw_val = qkv_node[tid];
    if (has_bias != 0u) {
        raw_val += qkv_bias[tid];
    }

    uint rope_pos = (start_pos + uint(tree_depths[node_idx])) % max_rope_len;
    uint slot = (write_pos + node_idx) % max_seq_len;
    uint hd_idx = tid % half_head_dim;

    if (tid < q_pairs) {
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[rope_pos * half_head_dim + hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        qkv_node[tid] = half2(res);
    } else if (tid < q_pairs + kv_pairs) {
        uint k_tid = tid - q_pairs;
        uint kv_head = k_tid / half_head_dim;
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[rope_pos * half_head_dim + hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        half2 res_h = half2(res);

        uint num_groups = max(1u, half_head_dim / 32u);
        uint group_idx = min(hd_idx / 32u, num_groups - 1u);
        uint lane_id = tid % 32u;
        float local_max = max(abs(float(res_h.x)), abs(float(res_h.y)));
        float group_max = simd_max(local_max);
        half scale = half(group_max / 448.0f);
        if (scale < 1e-8h) scale = 1.0h;
        half inv_scale = 1.0h / scale;

        uint scale_idx = (kv_head * max_seq_len + slot) * num_groups + group_idx;
        if (lane_id == 0u) {
            k_scales[scale_idx] = scale;
        }

        uchar2 packed = uchar2(
            float_to_fp8_e4m3fn(float(res_h.x) * float(inv_scale)),
            float_to_fp8_e4m3fn(float(res_h.y) * float(inv_scale))
        );
        uint dst_idx = (kv_head * max_seq_len + slot) * half_head_dim + hd_idx;
        k_cache[dst_idx] = packed;
    } else {
        uint v_tid = tid - (q_pairs + kv_pairs);
        uint kv_head = v_tid / half_head_dim;

        uint num_groups = max(1u, half_head_dim / 32u);
        uint group_idx = min(hd_idx / 32u, num_groups - 1u);
        uint lane_id = tid % 32u;
        float local_max = max(abs(float(raw_val.x)), abs(float(raw_val.y)));
        float group_max = simd_max(local_max);
        half scale = half(group_max / 448.0f);
        if (scale < 1e-8h) scale = 1.0h;
        half inv_scale = 1.0h / scale;

        uint scale_idx = (kv_head * max_seq_len + slot) * num_groups + group_idx;
        if (lane_id == 0u) {
            v_scales[scale_idx] = scale;
        }

        uchar2 packed = uchar2(
            float_to_fp8_e4m3fn(float(raw_val.x) * float(inv_scale)),
            float_to_fp8_e4m3fn(float(raw_val.y) * float(inv_scale))
        );
        uint dst_idx = (kv_head * max_seq_len + slot) * half_head_dim + hd_idx;
        v_cache[dst_idx] = packed;
    }
}

// In-Flight INT4 KV-Cache Direct Write Kernel (batched tree speculative verification)
kernel void rope_tree_kv_cache_write_int4_kernel(
    device half2 *qkv [[buffer(0)]],
    device const float2 *freqs_cis [[buffer(1)]],
    device uchar *k_cache [[buffer(2)]],
    device uchar *v_cache [[buffer(3)]],
    device half *k_scales [[buffer(4)]],
    device half *v_scales [[buffer(5)]],
    constant const int *tree_depths [[buffer(6)]],
    constant uint &q_pairs [[buffer(7)]],
    constant uint &kv_pairs [[buffer(8)]],
    constant uint &half_head_dim [[buffer(9)]],
    constant uint &max_seq_len [[buffer(10)]],
    constant uint &max_rope_len [[buffer(11)]],
    constant uint &start_pos [[buffer(12)]],
    constant uint &write_pos [[buffer(13)]],
    constant uint &T [[buffer(14)]],
    device const half2 *qkv_bias [[buffer(15)]],
    constant uint &has_bias [[buffer(16)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint tid = gid.x;
    uint node_idx = gid.y;
    uint total_pairs = q_pairs + (kv_pairs << 1);
    if (tid >= total_pairs || node_idx >= T) return;

    device half2 *qkv_node = qkv + node_idx * total_pairs;
    half2 raw_val = qkv_node[tid];
    if (has_bias != 0u) {
        raw_val += qkv_bias[tid];
    }

    uint rope_pos = (start_pos + uint(tree_depths[node_idx])) % max_rope_len;
    uint slot = (write_pos + node_idx) % max_seq_len;
    uint hd_idx = tid % half_head_dim;

    if (tid < q_pairs) {
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[rope_pos * half_head_dim + hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        qkv_node[tid] = half2(res);
    } else if (tid < q_pairs + kv_pairs) {
        uint k_tid = tid - q_pairs;
        uint kv_head = k_tid / half_head_dim;
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[rope_pos * half_head_dim + hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        half2 res_h = half2(res);

        uint num_groups = max(1u, half_head_dim / 32u);
        uint group_idx = min(hd_idx / 32u, num_groups - 1u);
        uint lane_id = tid % 32u;
        float local_max = max(abs(float(res_h.x)), abs(float(res_h.y)));
        float group_max = simd_max(local_max);
        half scale = half(group_max / 7.0f);
        if (scale < 1e-8h) scale = 1.0h;
        half inv_scale = 1.0h / scale;

        uint scale_idx = (kv_head * max_seq_len + slot) * num_groups + group_idx;
        if (lane_id == 0u) {
            k_scales[scale_idx] = scale;
        }

        uchar packed = quant_pack_int4_pair(res_h, inv_scale);
        uint dst_idx = (kv_head * max_seq_len + slot) * half_head_dim + hd_idx;
        k_cache[dst_idx] = packed;
    } else {
        uint v_tid = tid - (q_pairs + kv_pairs);
        uint kv_head = v_tid / half_head_dim;

        uint num_groups = max(1u, half_head_dim / 32u);
        uint group_idx = min(hd_idx / 32u, num_groups - 1u);
        uint lane_id = tid % 32u;
        float local_max = max(abs(float(raw_val.x)), abs(float(raw_val.y)));
        float group_max = simd_max(local_max);
        half scale = half(group_max / 7.0f);
        if (scale < 1e-8h) scale = 1.0h;
        half inv_scale = 1.0h / scale;

        uint scale_idx = (kv_head * max_seq_len + slot) * num_groups + group_idx;
        if (lane_id == 0u) {
            v_scales[scale_idx] = scale;
        }

        uchar packed = quant_pack_int4_pair(raw_val, inv_scale);
        uint dst_idx = (kv_head * max_seq_len + slot) * half_head_dim + hd_idx;
        v_cache[dst_idx] = packed;
    }
}

// Fast FP8 KV Cache Dequantization Kernel
kernel void dequantize_kv_cache_fp8_kernel(
    device const uchar2 *cache_fp8 [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device half2 *out_fp16 [[buffer(2)]],
    constant uint &half_head_dim [[buffer(3)]],
    constant uint &seq_len [[buffer(4)]],
    constant uint &total_elements [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= total_elements) return;

    uint hd_idx = tid % half_head_dim;
    uint t_idx = (tid / half_head_dim) % seq_len;
    uint kv_head = tid / (half_head_dim * seq_len);

    uint num_groups = max(1u, half_head_dim / 32u);
    uint group_idx = min(hd_idx / 32u, num_groups - 1u);
    uint scale_idx = (kv_head * seq_len + t_idx) * num_groups + group_idx;
    half scale = scales[scale_idx];

    uchar2 packed = cache_fp8[tid];
    out_fp16[tid] = half2(
        decode_fp8_e4m3fn(packed.x) * scale,
        decode_fp8_e4m3fn(packed.y) * scale
    );
}

// Fast INT4 KV Cache Dequantization Kernel
kernel void dequantize_kv_cache_int4_kernel(
    device const uchar *cache_int4 [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device half2 *out_fp16 [[buffer(2)]],
    constant uint &half_head_dim [[buffer(3)]],
    constant uint &seq_len [[buffer(4)]],
    constant uint &total_elements [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= total_elements) return;

    uint hd_idx = tid % half_head_dim;
    uint t_idx = (tid / half_head_dim) % seq_len;
    uint kv_head = tid / (half_head_dim * seq_len);

    uint num_groups = max(1u, half_head_dim / 32u);
    uint group_idx = min(hd_idx / 32u, num_groups - 1u);
    uint scale_idx = (kv_head * seq_len + t_idx) * num_groups + group_idx;
    half scale = scales[scale_idx];

    uchar packed = cache_int4[tid];
    out_fp16[tid] = dequant_int4_packed(packed, scale);
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
// 6C. Quantized Batched Matrix-Vector Multiply Q4_G64 (Tall-and-Skinny GEMM)
// Evaluates M tokens (M in [1, 8]) simultaneously against 4-bit weights
// Weights are read from DRAM ONCE and reused across all M tokens in registers.
// Input x: (M, K) half
// Weight w_packed: (N, K / 8) uint32 (8 x 4-bit nibbles per word)
// ScalesBiases: (N, K / 64) half2 (scale, bias) per 64 weights
// Output y: (M, N) half
// ----------------------------------------------------------------------------
kernel void qmm_q4_g64_half_kernel(
    device const half *x [[buffer(0)]],
    device const uint *w_packed [[buffer(1)]],
    device const half2 *scales_biases [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &N [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    // 2 SIMD-groups per threadgroup, 2 output rows per SIMD-group -> 4 output rows per threadgroup
    uint row_base = (tgpig.x * 2 + simd_group_id) * 2;
    uint m_base = tgpig.y * 8u;
    if (row_base >= N || m_base >= M) return;

    uint K_packed = K >> 3;       // K / 8 uint32 words per row
    uint K_groups = K >> 6;       // K / 64 groups per row

    // Accumulators for up to 8 tokens x 2 output rows
    float acc[8][2] = {{0.0f}};

    uint actual_M = min(M - m_base, 8u);

    for (uint idx = simd_lane_id; idx < K_packed; idx += 32) {
        uint k_elem = idx << 3;   // Starting element index (multiple of 8)
        uint g_idx = idx >> 3;    // Group index (every 8 uint32 words = 64 elements)

        // Load 8 FP16 activations for each of the M tokens
        half4 x_lo[8];
        half4 x_hi[8];
        float x_sum[8];

        #pragma unroll
        for (uint m = 0; m < 8; ++m) {
            if (m < actual_M) {
                device const half *x_m = x + (m_base + m) * K + k_elem;
                x_lo[m] = *(device const half4 *)x_m;
                x_hi[m] = *(device const half4 *)(x_m + 4);
                x_sum[m] = float(x_lo[m].x + x_lo[m].y + x_lo[m].z + x_lo[m].w +
                                 x_hi[m].x + x_hi[m].y + x_hi[m].z + x_hi[m].w);
            }
        }

        // Process up to 2 output rows
        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            uint row = row_base + r;
            if (row < N) {
                uint packed = w_packed[row * K_packed + idx];
                half2 sb = scales_biases[row * K_groups + g_idx];
                float scale = float(sb.x);
                float bias  = float(sb.y);

                float4 q_lo = float4(packed & 0xFu, (packed >> 4) & 0xFu, (packed >> 8) & 0xFu, (packed >> 12) & 0xFu);
                float4 q_hi = float4((packed >> 16) & 0xFu, (packed >> 20) & 0xFu, (packed >> 24) & 0xFu, (packed >> 28) & 0xFu);

                #pragma unroll
                for (uint m = 0; m < 8; ++m) {
                    if (m < actual_M) {
                        float d_prod = dot(float4(x_lo[m]), q_lo) + dot(float4(x_hi[m]), q_hi);
                        acc[m][r] += fma(d_prod, scale, bias * x_sum[m]);
                    }
                }
            }
        }
    }

    // Hardware single-cycle 32-lane SIMD reduction for each token
    #pragma unroll
    for (uint m = 0; m < 8; ++m) {
        if (m < actual_M) {
            acc[m][0] = simd_sum(acc[m][0]);
            acc[m][1] = simd_sum(acc[m][1]);
        }
    }

    if (simd_lane_id == 0) {
        for (uint m = 0; m < actual_M; ++m) {
            if (row_base + 0 < N) y[(m_base + m) * N + row_base + 0] = half(acc[m][0]);
            if (row_base + 1 < N) y[(m_base + m) * N + row_base + 1] = half(acc[m][1]);
        }
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
    // 4 SIMD-groups (128 threads) = 8 sub-warps of 16 lanes; 2 rows per sub-warp = 16 rows/TG.
    // 16 lanes * 4 INT8 bytes/lane = 64 weights = 1 exact quantization group per step.
    uint sg_row_base = tgpig.x * 16u + (simd_group_id << 2);
    if (sg_row_base >= N) return;

    uint sub_id = (simd_group_id << 1) | (simd_lane_id >> 4);
    uint lane16 = simd_lane_id & 15u;
    uint row_base = tgpig.x * 16u + (sub_id << 1);

    uint K_packed = K >> 2;       // K / 4 uint32 words per row
    uint K_groups = K >> 6;       // K / 64 groups per row

    bool valid0 = (row_base < N);
    bool valid1 = (row_base + 1u < N);
    float acc0 = 0.0f, acc1 = 0.0f;

    device const uint *w0 = w_packed + row_base * K_packed;
    device const uint *w1 = w0 + K_packed;
    device const half2 *sb0 = scales_biases + row_base * K_groups;
    device const half2 *sb1 = sb0 + K_groups;

    for (uint g = 0; g < K_groups; ++g) {
        uint idx = (g << 4) + lane16;
        half4 x_vec = *(device const half4 *)(x + (idx << 2));
        float4 fx = float4(x_vec);
        float x_sum = (fx.x + fx.y) + (fx.z + fx.w);

        if (valid0) {
            uchar4 q0 = as_type<uchar4>(w0[idx]);
            half2 s0 = sb0[g];
            acc0 = fma(dot(fx, float4(q0)), float(s0.x), fma(float(s0.y), x_sum, acc0));
        }
        if (valid1) {
            uchar4 q1 = as_type<uchar4>(w1[idx]);
            half2 s1 = sb1[g];
            acc1 = fma(dot(fx, float4(q1)), float(s1.x), fma(float(s1.y), x_sum, acc1));
        }
    }

    acc0 += simd_shuffle_xor(acc0, 8u);
    acc0 += simd_shuffle_xor(acc0, 4u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc0 += simd_shuffle_xor(acc0, 1u);

    acc1 += simd_shuffle_xor(acc1, 8u);
    acc1 += simd_shuffle_xor(acc1, 4u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);

    if (lane16 == 0u) {
        if (valid0) y[row_base] = half(acc0);
        if (valid1) y[row_base + 1u] = half(acc1);
    }
}

// ----------------------------------------------------------------------------
// 6D. Quantized Batched Matrix-Vector Multiply Q8_G64 (Tall-and-Skinny GEMM)
// Evaluates M tokens (tiled by 8) simultaneously against 8-bit weights.
// Cooperative SRAM x-tiling + 16-lane group-aligned hardware uchar4 unpack.
// ----------------------------------------------------------------------------
kernel void qmm_q8_g64_half_kernel(
    device const half *x [[buffer(0)]],
    device const uint *w_packed [[buffer(1)]],
    device const half2 *scales_biases [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &N [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    // 4 SIMD-groups (128 threads) = 8 sub-warps of 16 lanes; 4 rows/sub-warp = 32 rows/TG
    uint tg_row_base = tgpig.x * 32u;
    uint m_base = tgpig.y * 8u;
    if (tg_row_base >= N || m_base >= M) return;

    uint tid = (simd_group_id << 5) | simd_lane_id; // 0..127
    uint m_ld = tid >> 4;                           // 0..7
    uint c_ld = tid & 15u;                          // 0..15

    uint sub_id = (simd_group_id << 1) | (simd_lane_id >> 4); // 0..7
    uint lane16 = simd_lane_id & 15u;                         // 0..15
    uint row_base = tg_row_base + (sub_id << 2);

    uint K_packed = K >> 2;
    uint K_groups = K >> 6;
    uint actual_M = min(M - m_base, 8u);

    threadgroup float4 s_x[8][16];
    threadgroup float s_xsum[8][16];

    float acc[8][4] = {{0.0f}};

    for (uint g = 0; g < K_groups; ++g) {
        if (m_ld < actual_M) {
            half4 xv = *(device const half4 *)(x + (m_base + m_ld) * K + (g << 6) + (c_ld << 2));
            float4 fxv = float4(xv);
            s_x[m_ld][c_ld] = fxv;
            s_xsum[m_ld][c_ld] = (fxv.x + fxv.y) + (fxv.z + fxv.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float4 fx_reg[8];
        float xsum_reg[8];
        #pragma unroll
        for (uint m = 0; m < 8; ++m) {
            if (m < actual_M) {
                fx_reg[m] = s_x[m][lane16];
                xsum_reg[m] = s_xsum[m][lane16];
            }
        }

        uint idx = (g << 4) + lane16;
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            uint row = row_base + r;
            if (row < N) {
                uchar4 qb = as_type<uchar4>(w_packed[row * K_packed + idx]);
                float4 qf = float4(qb);
                half2 sb = scales_biases[row * K_groups + g];
                float scale = float(sb.x);
                float bias  = float(sb.y);

                #pragma unroll
                for (uint m = 0; m < 8; ++m) {
                    if (m < actual_M) {
                        acc[m][r] = fma(dot(fx_reg[m], qf), scale, fma(bias, xsum_reg[m], acc[m][r]));
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint m = 0; m < 8; ++m) {
        if (m < actual_M) {
            #pragma unroll
            for (uint r = 0; r < 4; ++r) {
                acc[m][r] += simd_shuffle_xor(acc[m][r], 8u);
                acc[m][r] += simd_shuffle_xor(acc[m][r], 4u);
                acc[m][r] += simd_shuffle_xor(acc[m][r], 2u);
                acc[m][r] += simd_shuffle_xor(acc[m][r], 1u);
            }
        }
    }

    if (lane16 == 0u) {
        for (uint m = 0; m < actual_M; ++m) {
            #pragma unroll
            for (uint r = 0; r < 4; ++r) {
                if (row_base + r < N) {
                    y[(m_base + m) * N + row_base + r] = half(acc[m][r]);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 6E. MARLIN 64207531 + Ladder tTile Register-Direct Dequantization Kernels
// Eliminates serial ALU shift/mask stalls via IEEE-754 Binary16 magic-exponent
// parallel unpacking (1 shift, 0 uitofp) and eliminates 100% of Threadgroup
// shared memory round-trips by writing directly to simdgroup_half8x8 registers.
// Unified Weight Layout:
//   w_marlin:  (N_tiles, K / 64, 32, 2) uint32 / uint2  (N_tiles = ceil(N / 8))
//   sb_marlin: (N_tiles, K / 64, 8, 2)  half   / half2  (scale, bias)
// ----------------------------------------------------------------------------
inline void marlin_unpack_4x_half2(
    uint w,
    thread half2 &q0,
    thread half2 &q1,
    thread half2 &q2,
    thread half2 &q3)
{
    uint w_hi = w >> 8;
    q0 = as_type<half2>((w    & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
    q1 = as_type<half2>((w    & 0x00F000F0u) | 0x54005400u) - half2(64.0h);
    q2 = as_type<half2>((w_hi & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
    q3 = as_type<half2>((w_hi & 0x00F000F0u) | 0x54005400u) - half2(64.0h);
}

kernel void qmv_q4_marlin_apple_kernel(
    device const half *x [[buffer(0)]],
    device const uint2 *w_marlin [[buffer(1)]],
    device const half2 *sb_marlin [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &split_k [[buffer(6)]],
    device const half *lin_bias [[buffer(7)]],
    constant uint &has_bias [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    uint num_sg = tg_size.x >> 5;
    uint tiles_per_tg = num_sg / split_k;
    uint local_tile = sg_id / split_k;
    uint sk_id = sg_id % split_k;

    uint tile_n = tgpig.x * tiles_per_tg + local_tile;
    uint N_tiles = (N + 7u) >> 3;
    uint K_groups = K >> 6;

    uint r  = ((lane & 16u) >> 2) | ((lane & 6u) >> 1);
    uint c0 = ((lane & 8u)  >> 1) | ((lane & 1u) << 1);
    uint c1 = c0 + 1u;

    float2 acc = float2(0.0f);
    if (tile_n < N_tiles) {
        device const uint2 *w_tile  = w_marlin  + tile_n * (K_groups * 32u);
        device const half2 *sb_tile = sb_marlin + tile_n * (K_groups * 8u);

        uint gk = sk_id;
        if (gk < K_groups) {
            uint2 wp_curr = w_tile[gk * 32u + lane];
            half2 sb_lane = (lane < 8u) ? sb_tile[gk * 8u + lane] : half2(0.0h);

            for (; gk + split_k < K_groups; gk += split_k) {
                uint next_gk = gk + split_k;
                uint2 wp_next = w_tile[next_gk * 32u + lane];
                half2 sb_next = (lane < 8u) ? sb_tile[next_gk * 8u + lane] : half2(0.0h);

                half2 sb0 = simd_shuffle(sb_lane, c0);
                half2 sb1 = simd_shuffle(sb_lane, c1);
                float2 sc = float2(float(sb0.x), float(sb1.x));
                float2 bi = float2(float(sb0.y), float(sb1.y));

                device const half *xk = x + (gk << 6) + r;
                float fx0 = float(xk[0]),  fx1 = float(xk[8]),  fx2 = float(xk[16]), fx3 = float(xk[24]);
                float fx4 = float(xk[32]), fx5 = float(xk[40]), fx6 = float(xk[48]), fx7 = float(xk[56]);
                float x_sum = (fx0 + fx1 + fx2 + fx3) + (fx4 + fx5 + fx6 + fx7);

                half2 q0, q1, q2, q3;
                marlin_unpack_4x_half2(wp_curr.x, q0, q1, q2, q3);
                float2 q_dot = fma(float2(q0), fx0, float2(q1) * fx1);
                q_dot = fma(float2(q2), fx2, fma(float2(q3), fx3, q_dot));

                marlin_unpack_4x_half2(wp_curr.y, q0, q1, q2, q3);
                q_dot = fma(float2(q0), fx4, fma(float2(q1), fx5, q_dot));
                q_dot = fma(float2(q2), fx6, fma(float2(q3), fx7, q_dot));

                acc = fma(q_dot, sc, fma(bi, x_sum, acc));

                wp_curr = wp_next;
                sb_lane = sb_next;
            }

            half2 sb0 = simd_shuffle(sb_lane, c0);
            half2 sb1 = simd_shuffle(sb_lane, c1);
            float2 sc = float2(float(sb0.x), float(sb1.x));
            float2 bi = float2(float(sb0.y), float(sb1.y));

            device const half *xk = x + (gk << 6) + r;
            float fx0 = float(xk[0]),  fx1 = float(xk[8]),  fx2 = float(xk[16]), fx3 = float(xk[24]);
            float fx4 = float(xk[32]), fx5 = float(xk[40]), fx6 = float(xk[48]), fx7 = float(xk[56]);
            float x_sum = (fx0 + fx1 + fx2 + fx3) + (fx4 + fx5 + fx6 + fx7);

            half2 q0, q1, q2, q3;
            marlin_unpack_4x_half2(wp_curr.x, q0, q1, q2, q3);
            float2 q_dot = fma(float2(q0), fx0, float2(q1) * fx1);
            q_dot = fma(float2(q2), fx2, fma(float2(q3), fx3, q_dot));

            marlin_unpack_4x_half2(wp_curr.y, q0, q1, q2, q3);
            q_dot = fma(float2(q0), fx4, fma(float2(q1), fx5, q_dot));
            q_dot = fma(float2(q2), fx6, fma(float2(q3), fx7, q_dot));

            acc = fma(q_dot, sc, fma(bi, x_sum, acc));
        }

        acc += simd_shuffle_xor(acc, 2u);
        acc += simd_shuffle_xor(acc, 4u);
        acc += simd_shuffle_xor(acc, 16u);
    }

    uint row_base = tile_n << 3;
    if (split_k == 1u) {
        if (tile_n < N_tiles && r == 0u) {
            if (row_base + c0 < N) {
                float v0 = acc.x + (has_bias ? float(lin_bias[row_base + c0]) : 0.0f);
                y[row_base + c0] = half(v0);
            }
            if (row_base + c1 < N) {
                float v1 = acc.y + (has_bias ? float(lin_bias[row_base + c1]) : 0.0f);
                y[row_base + c1] = half(v1);
            }
        }
    } else {
        threadgroup float sg_red[8][8];
        if (r == 0u) {
            sg_red[sg_id][c0] = acc.x;
            sg_red[sg_id][c1] = acc.y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tile_n < N_tiles && sk_id == 0u && lane < 8u) {
            float sum = 0.0f;
            for (uint s = 0; s < split_k; ++s) {
                sum += sg_red[local_tile * split_k + s][lane];
            }
            if (row_base + lane < N) {
                if (has_bias) sum += float(lin_bias[row_base + lane]);
                y[row_base + lane] = half(sum);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 6F. MARLIN + Ladder Register-Direct Split-K QMM for M in [2, 8]
// Tile shape: BM = 8, BN = 8 per SIMD-group (1 simdgroup_float8x8 accumulator),
// P=2 software-pipelined weight prefetching + Split-K for full M3 GPU occupancy.
// Zero inner-loop shared memory or barriers.
// ----------------------------------------------------------------------------
kernel void qmm_q4_marlin_m8_apple_kernel(
    device const half *x [[buffer(0)]],
    device const uint2 *w_marlin [[buffer(1)]],
    device const half2 *sb_marlin [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &N [[buffer(6)]],
    constant uint &split_k [[buffer(7)]],
    device const half *lin_bias [[buffer(8)]],
    constant uint &has_bias [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    uint num_sg = tg_size.x >> 5;
    uint tiles_per_tg = num_sg / split_k;
    uint local_tile = sg_id / split_k;
    uint sk_id = sg_id % split_k;

    uint tile_n = tgpig.x * tiles_per_tg + local_tile;
    uint N_tiles = (N + 7u) >> 3;
    uint K_groups = K >> 6;

    uint r  = ((lane & 16u) >> 2) | ((lane & 6u) >> 1);
    uint c0 = ((lane & 8u)  >> 1) | ((lane & 1u) << 1);
    uint c1 = c0 + 1u;

    bool valid_m = (r < M);
    device const half *x_row = valid_m ? (x + r * K) : x;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    if (tile_n < N_tiles) {
        device const uint2 *wt  = w_marlin  + tile_n * (K_groups * 32u);
        device const half2 *sbt = sb_marlin + tile_n * (K_groups * 8u);

        uint gk = sk_id;
        if (gk < K_groups) {
            uint2 wp_curr = wt[gk * 32u + lane];
            half2 sb_curr = (lane < 8u) ? sbt[gk * 8u + lane] : half2(0.0h);

            for (; gk + split_k < K_groups; gk += split_k) {
                uint next_gk = gk + split_k;
                uint2 wp_next = wt[next_gk * 32u + lane];
                half2 sb_next = (lane < 8u) ? sbt[next_gk * 8u + lane] : half2(0.0h);

                half2 s0 = simd_shuffle(sb_curr, c0), s1 = simd_shuffle(sb_curr, c1);
                half2 sc = half2(s0.x, s1.x), bi = half2(s0.y, s1.y);
                uint k_base = gk << 6;
                half2 q[4];

                marlin_unpack_4x_half2(wp_curr.x, q[0], q[1], q[2], q[3]);
                #pragma unroll
                for (uint s = 0; s < 4; ++s) {
                    half2 a = valid_m ? *(device const half2 *)(x_row + k_base + (s << 3) + c0) : half2(0.0h);
                    half2 w = fma(q[s], sc, bi);
                    simdgroup_half8x8 A, B;
                    A.thread_elements()[0] = a.x; A.thread_elements()[1] = a.y;
                    B.thread_elements()[0] = w.x; B.thread_elements()[1] = w.y;
                    simdgroup_multiply_accumulate(acc, A, B, acc);
                }

                marlin_unpack_4x_half2(wp_curr.y, q[0], q[1], q[2], q[3]);
                #pragma unroll
                for (uint s = 0; s < 4; ++s) {
                    half2 a = valid_m ? *(device const half2 *)(x_row + k_base + 32u + (s << 3) + c0) : half2(0.0h);
                    half2 w = fma(q[s], sc, bi);
                    simdgroup_half8x8 A, B;
                    A.thread_elements()[0] = a.x; A.thread_elements()[1] = a.y;
                    B.thread_elements()[0] = w.x; B.thread_elements()[1] = w.y;
                    simdgroup_multiply_accumulate(acc, A, B, acc);
                }

                wp_curr = wp_next;
                sb_curr = sb_next;
            }

            half2 s0 = simd_shuffle(sb_curr, c0), s1 = simd_shuffle(sb_curr, c1);
            half2 sc = half2(s0.x, s1.x), bi = half2(s0.y, s1.y);
            uint k_base = gk << 6;
            half2 q[4];

            marlin_unpack_4x_half2(wp_curr.x, q[0], q[1], q[2], q[3]);
            #pragma unroll
            for (uint s = 0; s < 4; ++s) {
                half2 a = valid_m ? *(device const half2 *)(x_row + k_base + (s << 3) + c0) : half2(0.0h);
                half2 w = fma(q[s], sc, bi);
                simdgroup_half8x8 A, B;
                A.thread_elements()[0] = a.x; A.thread_elements()[1] = a.y;
                B.thread_elements()[0] = w.x; B.thread_elements()[1] = w.y;
                simdgroup_multiply_accumulate(acc, A, B, acc);
            }

            marlin_unpack_4x_half2(wp_curr.y, q[0], q[1], q[2], q[3]);
            #pragma unroll
            for (uint s = 0; s < 4; ++s) {
                half2 a = valid_m ? *(device const half2 *)(x_row + k_base + 32u + (s << 3) + c0) : half2(0.0h);
                half2 w = fma(q[s], sc, bi);
                simdgroup_half8x8 A, B;
                A.thread_elements()[0] = a.x; A.thread_elements()[1] = a.y;
                B.thread_elements()[0] = w.x; B.thread_elements()[1] = w.y;
                simdgroup_multiply_accumulate(acc, A, B, acc);
            }
        }
    }

    auto e = acc.thread_elements();
    uint col0 = (tile_n << 3) + c0;
    if (split_k == 1u) {
        if (tile_n < N_tiles && valid_m) {
            if (col0 < N)      y[r * N + col0]      = half(e[0] + (has_bias ? float(lin_bias[col0]) : 0.0f));
            if (col0 + 1u < N) y[r * N + col0 + 1u] = half(e[1] + (has_bias ? float(lin_bias[col0 + 1u]) : 0.0f));
        }
    } else {
        threadgroup float2 sg_red[4][32];
        sg_red[sg_id][lane] = float2(e[0], e[1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tile_n < N_tiles && sk_id == 0u && valid_m) {
            float2 sum = float2(0.0f);
            for (uint s = 0; s < split_k; ++s) {
                sum += sg_red[local_tile * split_k + s][lane];
            }
            if (col0 < N)      y[r * N + col0]      = half(sum.x + (has_bias ? float(lin_bias[col0]) : 0.0f));
            if (col0 + 1u < N) y[r * N + col0 + 1u] = half(sum.y + (has_bias ? float(lin_bias[col0 + 1u]) : 0.0f));
        }
    }
}

// ----------------------------------------------------------------------------
// 6G. MARLIN + Ladder Register-Direct Split-K QMM for M > 8 (M = 9..2048+)
// Tile shape: BM = 32, BN = 16 per SIMD-group (4x2 simdgroup_float8x8 accumulators)
// Uniformly gates M-subtiles (m0..m3) with zero SIMD divergence so M=16 executes
// 2x2 and M>=32 executes full 4x2 (reusing each unpacked weight across 32 tokens!),
// paired with P=2 software prefetching and Split-K for full M3 GPU occupancy.
// ----------------------------------------------------------------------------
kernel void qmm_q4_marlin_apple_kernel(
    device const half *x [[buffer(0)]],
    device const uint2 *w_marlin [[buffer(1)]],
    device const half2 *sb_marlin [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &N [[buffer(6)]],
    constant uint &split_k [[buffer(7)]],
    device const half *lin_bias [[buffer(8)]],
    constant uint &has_bias [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    uint num_sg = tg_size.x >> 5;
    uint pairs_per_tg = num_sg / split_k;
    uint local_pair = sg_id / split_k;
    uint sk_id = sg_id % split_k;

    uint tile_n0 = (tgpig.x * pairs_per_tg + local_pair) * 2u;
    uint tile_m0 = tgpig.y * 32u;
    uint N_tiles = (N + 7u) >> 3;
    if (tile_m0 >= M) return;

    bool v0 = (tile_n0 < N_tiles);
    bool v1 = (tile_n0 + 1u < N_tiles);
    uint K_groups = K >> 6;
    uint r  = ((lane & 16u) >> 2) | ((lane & 6u) >> 1);
    uint c0 = ((lane & 8u)  >> 1) | ((lane & 1u) << 1);
    uint c1 = c0 + 1u;

    bool has_m1 = (tile_m0 + 8u  < M);
    bool has_m2 = (tile_m0 + 16u < M);
    bool has_m3 = (tile_m0 + 24u < M);

    bool vm0 = (tile_m0 + r < M);
    bool vm1 = (tile_m0 + 8u  + r < M);
    bool vm2 = (tile_m0 + 16u + r < M);
    bool vm3 = (tile_m0 + 24u + r < M);

    device const half *xr0 = vm0 ? (x + (tile_m0 + r) * K) : x;
    device const half *xr1 = vm1 ? (x + (tile_m0 + 8u  + r) * K) : x;
    device const half *xr2 = vm2 ? (x + (tile_m0 + 16u + r) * K) : x;
    device const half *xr3 = vm3 ? (x + (tile_m0 + 24u + r) * K) : x;

    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc20 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc21 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc30 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc31 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    if (v0) {
        uint stride_w  = K_groups * 32u;
        uint stride_sb = K_groups * 8u;
        device const uint2 *wt0 = w_marlin + tile_n0 * stride_w;
        device const uint2 *wt1 = v1 ? (wt0 + stride_w) : wt0;
        device const half2 *sbt0 = sb_marlin + tile_n0 * stride_sb;
        device const half2 *sbt1 = v1 ? (sbt0 + stride_sb) : sbt0;

        uint gk = sk_id;
        if (gk < K_groups) {
            uint2 wp0_c = wt0[gk * 32u + lane];
            uint2 wp1_c = v1 ? wt1[gk * 32u + lane] : uint2(0u);
            half2 sb_c  = (lane < 8u) ? sbt0[gk * 8u + lane] : ((v1 && lane < 16u) ? sbt1[gk * 8u + (lane - 8u)] : half2(0.0h));

            for (; gk < K_groups; gk += split_k) {
                uint next_gk = gk + split_k;
                uint2 wp0_n = uint2(0u), wp1_n = uint2(0u);
                half2 sb_n = half2(0.0h);
                if (next_gk < K_groups) {
                    uint noff = next_gk * 32u + lane;
                    wp0_n = wt0[noff];
                    if (v1) wp1_n = wt1[noff];
                    if (lane < 8u) sb_n = sbt0[next_gk * 8u + lane];
                    else if (v1 && lane < 16u) sb_n = sbt1[next_gk * 8u + (lane - 8u)];
                }

                half2 s00 = simd_shuffle(sb_c, c0), s01 = simd_shuffle(sb_c, c1);
                half2 sc0 = half2(s00.x, s01.x), bi0 = half2(s00.y, s01.y);
                half2 s10 = simd_shuffle(sb_c, 8u + c0), s11 = simd_shuffle(sb_c, 8u + c1);
                half2 sc1 = half2(s10.x, s11.x), bi1 = half2(s10.y, s11.y);

                uint k_base = gk << 6;
                half2 q0[4], q1[4];

                marlin_unpack_4x_half2(wp0_c.x, q0[0], q0[1], q0[2], q0[3]);
                if (v1) marlin_unpack_4x_half2(wp1_c.x, q1[0], q1[1], q1[2], q1[3]);
                #pragma unroll
                for (uint s = 0; s < 4; ++s) {
                    uint kc = k_base + (s << 3) + c0;
                    simdgroup_half8x8 B0, B1, A0, A1, A2, A3;
                    half2 w0 = fma(q0[s], sc0, bi0);
                    B0.thread_elements()[0] = w0.x; B0.thread_elements()[1] = w0.y;
                    if (v1) { half2 w1 = fma(q1[s], sc1, bi1); B1.thread_elements()[0] = w1.x; B1.thread_elements()[1] = w1.y; }

                    half2 a0 = vm0 ? *(device const half2 *)(xr0 + kc) : half2(0.0h);
                    A0.thread_elements()[0] = a0.x; A0.thread_elements()[1] = a0.y;
                    simdgroup_multiply_accumulate(acc00, A0, B0, acc00);
                    if (v1) simdgroup_multiply_accumulate(acc01, A0, B1, acc01);

                    if (has_m1) {
                        half2 a1 = vm1 ? *(device const half2 *)(xr1 + kc) : half2(0.0h);
                        A1.thread_elements()[0] = a1.x; A1.thread_elements()[1] = a1.y;
                        simdgroup_multiply_accumulate(acc10, A1, B0, acc10);
                        if (v1) simdgroup_multiply_accumulate(acc11, A1, B1, acc11);
                    }
                    if (has_m2) {
                        half2 a2 = vm2 ? *(device const half2 *)(xr2 + kc) : half2(0.0h);
                        A2.thread_elements()[0] = a2.x; A2.thread_elements()[1] = a2.y;
                        simdgroup_multiply_accumulate(acc20, A2, B0, acc20);
                        if (v1) simdgroup_multiply_accumulate(acc21, A2, B1, acc21);
                    }
                    if (has_m3) {
                        half2 a3 = vm3 ? *(device const half2 *)(xr3 + kc) : half2(0.0h);
                        A3.thread_elements()[0] = a3.x; A3.thread_elements()[1] = a3.y;
                        simdgroup_multiply_accumulate(acc30, A3, B0, acc30);
                        if (v1) simdgroup_multiply_accumulate(acc31, A3, B1, acc31);
                    }
                }

                marlin_unpack_4x_half2(wp0_c.y, q0[0], q0[1], q0[2], q0[3]);
                if (v1) marlin_unpack_4x_half2(wp1_c.y, q1[0], q1[1], q1[2], q1[3]);
                #pragma unroll
                for (uint s = 0; s < 4; ++s) {
                    uint kc = k_base + 32u + (s << 3) + c0;
                    simdgroup_half8x8 B0, B1, A0, A1, A2, A3;
                    half2 w0 = fma(q0[s], sc0, bi0);
                    B0.thread_elements()[0] = w0.x; B0.thread_elements()[1] = w0.y;
                    if (v1) { half2 w1 = fma(q1[s], sc1, bi1); B1.thread_elements()[0] = w1.x; B1.thread_elements()[1] = w1.y; }

                    half2 a0 = vm0 ? *(device const half2 *)(xr0 + kc) : half2(0.0h);
                    A0.thread_elements()[0] = a0.x; A0.thread_elements()[1] = a0.y;
                    simdgroup_multiply_accumulate(acc00, A0, B0, acc00);
                    if (v1) simdgroup_multiply_accumulate(acc01, A0, B1, acc01);

                    if (has_m1) {
                        half2 a1 = vm1 ? *(device const half2 *)(xr1 + kc) : half2(0.0h);
                        A1.thread_elements()[0] = a1.x; A1.thread_elements()[1] = a1.y;
                        simdgroup_multiply_accumulate(acc10, A1, B0, acc10);
                        if (v1) simdgroup_multiply_accumulate(acc11, A1, B1, acc11);
                    }
                    if (has_m2) {
                        half2 a2 = vm2 ? *(device const half2 *)(xr2 + kc) : half2(0.0h);
                        A2.thread_elements()[0] = a2.x; A2.thread_elements()[1] = a2.y;
                        simdgroup_multiply_accumulate(acc20, A2, B0, acc20);
                        if (v1) simdgroup_multiply_accumulate(acc21, A2, B1, acc21);
                    }
                    if (has_m3) {
                        half2 a3 = vm3 ? *(device const half2 *)(xr3 + kc) : half2(0.0h);
                        A3.thread_elements()[0] = a3.x; A3.thread_elements()[1] = a3.y;
                        simdgroup_multiply_accumulate(acc30, A3, B0, acc30);
                        if (v1) simdgroup_multiply_accumulate(acc31, A3, B1, acc31);
                    }
                }

                wp0_c = wp0_n; wp1_c = wp1_n; sb_c = sb_n;
            }
        }
    }

    uint col0 = (tile_n0 << 3) + c0;
    float b00 = has_bias && (col0 + 0u < N) ? float(lin_bias[col0 + 0u]) : 0.0f;
    float b01 = has_bias && (col0 + 1u < N) ? float(lin_bias[col0 + 1u]) : 0.0f;
    float b10 = has_bias && (col0 + 8u < N) ? float(lin_bias[col0 + 8u]) : 0.0f;
    float b11 = has_bias && (col0 + 9u < N) ? float(lin_bias[col0 + 9u]) : 0.0f;

    auto e00 = acc00.thread_elements(), e01 = acc01.thread_elements();
    auto e10 = acc10.thread_elements(), e11 = acc11.thread_elements();
    auto e20 = acc20.thread_elements(), e21 = acc21.thread_elements();
    auto e30 = acc30.thread_elements(), e31 = acc31.thread_elements();

    if (split_k == 1u) {
        if (v0) {
            if (vm0) {
                uint gm = tile_m0 + r;
                if (col0 + 0u < N) y[gm * N + col0 + 0u] = half(e00[0] + b00);
                if (col0 + 1u < N) y[gm * N + col0 + 1u] = half(e00[1] + b01);
                if (v1) {
                    if (col0 + 8u < N) y[gm * N + col0 + 8u] = half(e01[0] + b10);
                    if (col0 + 9u < N) y[gm * N + col0 + 9u] = half(e01[1] + b11);
                }
            }
            if (vm1) {
                uint gm = tile_m0 + 8u + r;
                if (col0 + 0u < N) y[gm * N + col0 + 0u] = half(e10[0] + b00);
                if (col0 + 1u < N) y[gm * N + col0 + 1u] = half(e10[1] + b01);
                if (v1) {
                    if (col0 + 8u < N) y[gm * N + col0 + 8u] = half(e11[0] + b10);
                    if (col0 + 9u < N) y[gm * N + col0 + 9u] = half(e11[1] + b11);
                }
            }
            if (vm2) {
                uint gm = tile_m0 + 16u + r;
                if (col0 + 0u < N) y[gm * N + col0 + 0u] = half(e20[0] + b00);
                if (col0 + 1u < N) y[gm * N + col0 + 1u] = half(e20[1] + b01);
                if (v1) {
                    if (col0 + 8u < N) y[gm * N + col0 + 8u] = half(e21[0] + b10);
                    if (col0 + 9u < N) y[gm * N + col0 + 9u] = half(e21[1] + b11);
                }
            }
            if (vm3) {
                uint gm = tile_m0 + 24u + r;
                if (col0 + 0u < N) y[gm * N + col0 + 0u] = half(e30[0] + b00);
                if (col0 + 1u < N) y[gm * N + col0 + 1u] = half(e30[1] + b01);
                if (v1) {
                    if (col0 + 8u < N) y[gm * N + col0 + 8u] = half(e31[0] + b10);
                    if (col0 + 9u < N) y[gm * N + col0 + 9u] = half(e31[1] + b11);
                }
            }
        }
    } else {
        threadgroup float4 sg_red[4][32];
        // Pass 1: reduce m0 and m1 for tile_n0 and tile_n0 + 1
        sg_red[sg_id][lane] = float4(e00[0], e00[1], e01[0], e01[1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (v0 && sk_id == 0u && vm0) {
            float4 sum = float4(0.0f);
            for (uint s = 0; s < split_k; ++s) sum += sg_red[local_pair * split_k + s][lane];
            uint gm = tile_m0 + r;
            if (col0 + 0u < N) y[gm * N + col0 + 0u] = half(sum.x + b00);
            if (col0 + 1u < N) y[gm * N + col0 + 1u] = half(sum.y + b01);
            if (v1) {
                if (col0 + 8u < N) y[gm * N + col0 + 8u] = half(sum.z + b10);
                if (col0 + 9u < N) y[gm * N + col0 + 9u] = half(sum.w + b11);
            }
        }
        if (has_m1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sg_red[sg_id][lane] = float4(e10[0], e10[1], e11[0], e11[1]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (v0 && sk_id == 0u && vm1) {
                float4 sum = float4(0.0f);
                for (uint s = 0; s < split_k; ++s) sum += sg_red[local_pair * split_k + s][lane];
                uint gm = tile_m0 + 8u + r;
                if (col0 + 0u < N) y[gm * N + col0 + 0u] = half(sum.x + b00);
                if (col0 + 1u < N) y[gm * N + col0 + 1u] = half(sum.y + b01);
                if (v1) {
                    if (col0 + 8u < N) y[gm * N + col0 + 8u] = half(sum.z + b10);
                    if (col0 + 9u < N) y[gm * N + col0 + 9u] = half(sum.w + b11);
                }
            }
        }
        if (has_m2) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sg_red[sg_id][lane] = float4(e20[0], e20[1], e21[0], e21[1]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (v0 && sk_id == 0u && vm2) {
                float4 sum = float4(0.0f);
                for (uint s = 0; s < split_k; ++s) sum += sg_red[local_pair * split_k + s][lane];
                uint gm = tile_m0 + 16u + r;
                if (col0 + 0u < N) y[gm * N + col0 + 0u] = half(sum.x + b00);
                if (col0 + 1u < N) y[gm * N + col0 + 1u] = half(sum.y + b01);
                if (v1) {
                    if (col0 + 8u < N) y[gm * N + col0 + 8u] = half(sum.z + b10);
                    if (col0 + 9u < N) y[gm * N + col0 + 9u] = half(sum.w + b11);
                }
            }
        }
        if (has_m3) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sg_red[sg_id][lane] = float4(e30[0], e30[1], e31[0], e31[1]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (v0 && sk_id == 0u && vm3) {
                float4 sum = float4(0.0f);
                for (uint s = 0; s < split_k; ++s) sum += sg_red[local_pair * split_k + s][lane];
                uint gm = tile_m0 + 24u + r;
                if (col0 + 0u < N) y[gm * N + col0 + 0u] = half(sum.x + b00);
                if (col0 + 1u < N) y[gm * N + col0 + 1u] = half(sum.y + b01);
                if (v1) {
                    if (col0 + 8u < N) y[gm * N + col0 + 8u] = half(sum.z + b10);
                    if (col0 + 9u < N) y[gm * N + col0 + 9u] = half(sum.w + b11);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 7A. Generalized GQA Vector Flash-Decode Split-KV Kernel (T_q = 1, FP16)
// Supports arbitrary (n_head, n_kv_head) GQA ratios (1:1 MHA, 3:1 SmolLM2,
// 4:1 Llama-3.2, 7:1 Qwen2.5, 8:1+) and arbitrary even head_dim <= 128.
// Evaluates up to 8 Query heads per 1 KV head in registers per SIMD-group.
// ----------------------------------------------------------------------------
kernel void gqa_flash_decode_split_kv_half_kernel(
    device const half *q [[buffer(0)]],
    device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]],
    device half *out [[buffer(3)]],
    device half2 *partial_O [[buffer(4)]],
    device float *partial_m [[buffer(5)]],
    device float *partial_l [[buffer(6)]],
    constant uint &max_seq_len [[buffer(7)]],
    constant uint &active_len [[buffer(8)]],
    constant uint &num_splits [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    constant uint &n_head [[buffer(11)]],
    constant uint &n_kv_head [[buffer(12)]],
    constant uint &head_dim [[buffer(13)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    uint kv_h = tgpig.x;
    uint split_id = tgpig.y;
    uint rep_tile = tgpig.z;
    uint lane_id = simd_lane_id;

    uint n_rep = n_head / n_kv_head;
    uint q_sub_base = rep_tile * 8u;
    if (kv_h >= n_kv_head || q_sub_base >= n_rep) return;
    uint q_cnt = min(n_rep - q_sub_base, 8u);
    uint half_hd = head_dim >> 1;

    bool has_lo = (lane_id < half_hd);
    bool has_hi = ((lane_id + 32u) < half_hd);

    half2 q_lo[8];
    half2 q_hi[8];
    float m_arr[8];
    float l_arr[8];
    float2 o_lo[8];
    float2 o_hi[8];

    #pragma unroll
    for (uint r = 0; r < 8; ++r) {
        m_arr[r] = -1e38f;
        l_arr[r] = 0.0f;
        o_lo[r] = float2(0.0f);
        o_hi[r] = float2(0.0f);
        if (r < q_cnt) {
            uint qh = kv_h * n_rep + q_sub_base + r;
            device const half2 *q_ptr = (device const half2 *)(q + qh * head_dim);
            q_lo[r] = has_lo ? q_ptr[lane_id] : half2(0.0h);
            q_hi[r] = has_hi ? q_ptr[lane_id + 32u] : half2(0.0h);
        } else {
            q_lo[r] = half2(0.0h);
            q_hi[r] = half2(0.0h);
        }
    }

    uint tokens_per_split = (active_len + num_splits - 1u) / num_splits;
    uint t_start = min(split_id * tokens_per_split, active_len);
    uint t_end = min(t_start + tokens_per_split, active_len);

    device const half *k_base = k_cache + kv_h * max_seq_len * head_dim;
    device const half *v_base = v_cache + kv_h * max_seq_len * head_dim;

    for (uint t = t_start; t < t_end; ++t) {
        device const half2 *k_ptr = (device const half2 *)(k_base + t * head_dim);
        device const half2 *v_ptr = (device const half2 *)(v_base + t * head_dim);
        float2 k_lo = has_lo ? float2(k_ptr[lane_id]) : float2(0.0f);
        float2 k_hi = has_hi ? float2(k_ptr[lane_id + 32u]) : float2(0.0f);
        float2 v_lo = has_lo ? float2(v_ptr[lane_id]) : float2(0.0f);
        float2 v_hi = has_hi ? float2(v_ptr[lane_id + 32u]) : float2(0.0f);

        for (uint r = 0; r < q_cnt; ++r) {
            float p = fma(float(q_lo[r].x), k_lo.x, float(q_lo[r].y) * k_lo.y);
            if (has_hi) {
                p += fma(float(q_hi[r].x), k_hi.x, float(q_hi[r].y) * k_hi.y);
            }
            float score = simd_sum(p) * scale;
            float m_new = max(m_arr[r], score);
            float a = fast::exp(m_arr[r] - m_new);
            float b = fast::exp(score - m_new);
            l_arr[r] = fma(l_arr[r], a, b);
            m_arr[r] = m_new;
            o_lo[r] = o_lo[r] * a + v_lo * b;
            if (has_hi) {
                o_hi[r] = o_hi[r] * a + v_hi * b;
            }
        }
    }

    if (num_splits == 1u) {
        for (uint r = 0; r < q_cnt; ++r) {
            uint qh = kv_h * n_rep + q_sub_base + r;
            device half2 *out_ptr = (device half2 *)(out + qh * head_dim);
            float inv_l = (l_arr[r] > 0.0f) ? (1.0f / l_arr[r]) : 0.0f;
            if (has_lo) out_ptr[lane_id] = half2(o_lo[r] * inv_l);
            if (has_hi) out_ptr[lane_id + 32u] = half2(o_hi[r] * inv_l);
        }
    } else {
        for (uint r = 0; r < q_cnt; ++r) {
            uint qh = kv_h * n_rep + q_sub_base + r;
            uint p_base = qh * (num_splits * half_hd) + split_id * half_hd;
            if (has_lo) partial_O[p_base + lane_id] = half2(o_lo[r]);
            if (has_hi) partial_O[p_base + lane_id + 32u] = half2(o_hi[r]);
            if (lane_id == 0u) {
                uint ml_idx = qh * num_splits + split_id;
                partial_m[ml_idx] = m_arr[r];
                partial_l[ml_idx] = l_arr[r];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 7B. Generalized GQA Split-KV Reduction Kernel (FP16)
// ----------------------------------------------------------------------------
kernel void gqa_flash_decode_reduce_half_kernel(
    device const half2 *partial_O [[buffer(0)]],
    device const float *partial_m [[buffer(1)]],
    device const float *partial_l [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant uint &num_splits [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    uint qh = tgpig.x;
    uint lane_id = simd_lane_id;
    uint half_hd = head_dim >> 1;

    threadgroup float shared_buf[2];

    if (lane_id == 0u) {
        float local_max = -1e38f;
        for (uint s = 0; s < num_splits; ++s) {
            float m_val = partial_m[qh * num_splits + s];
            if (m_val > local_max) local_max = m_val;
        }
        float global_l = 0.0f;
        for (uint s = 0; s < num_splits; ++s) {
            float m_val = partial_m[qh * num_splits + s];
            float l_val = partial_l[qh * num_splits + s];
            global_l += l_val * fast::exp(m_val - local_max);
        }
        shared_buf[0] = local_max;
        shared_buf[1] = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_m = shared_buf[0];
    float inv_l = shared_buf[1];
    device half2 *out_head = (device half2 *)(out + qh * head_dim);

    for (uint c = lane_id; c < half_hd; c += 32u) {
        float2 o_acc = float2(0.0f);
        for (uint s = 0; s < num_splits; ++s) {
            float m_val = partial_m[qh * num_splits + s];
            float weight = fast::exp(m_val - global_m);
            half2 o_part = partial_O[qh * (num_splits * half_hd) + s * half_hd + c];
            o_acc += float2(o_part) * weight;
        }
        out_head[c] = half2(o_acc * inv_l);
    }
}

// ----------------------------------------------------------------------------
// 7C. Topology-Aware Batched Tree RoPE + KV Cache Write Kernel (T <= 32, FP16)
// Rotates Q and K for each candidate tree node u in [0, T) using its exact
// topological tree depth (start_pos + tree_depths[u]), fuses optional QKV bias,
// and writes K, V into KV cache slot (write_pos + u) % max_seq_len.
// ----------------------------------------------------------------------------
kernel void rope_tree_kv_cache_write_half_kernel(
    device half2 *qkv [[buffer(0)]],
    device const float2 *freqs_cis [[buffer(1)]],
    device half2 *k_cache [[buffer(2)]],
    device half2 *v_cache [[buffer(3)]],
    constant const int *tree_depths [[buffer(4)]],
    constant uint &q_pairs [[buffer(5)]],
    constant uint &kv_pairs [[buffer(6)]],
    constant uint &half_head_dim [[buffer(7)]],
    constant uint &max_seq_len [[buffer(8)]],
    constant uint &max_rope_len [[buffer(9)]],
    constant uint &start_pos [[buffer(10)]],
    constant uint &write_pos [[buffer(11)]],
    constant uint &T [[buffer(12)]],
    device const half2 *qkv_bias [[buffer(13)]],
    constant uint &has_bias [[buffer(14)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint tid = gid.x;
    uint node_idx = gid.y;
    uint total_pairs = q_pairs + (kv_pairs << 1);
    if (tid >= total_pairs || node_idx >= T) return;

    device half2 *qkv_node = qkv + node_idx * total_pairs;
    half2 raw_val = qkv_node[tid];
    if (has_bias != 0u) {
        raw_val += qkv_bias[tid];
    }

    uint rope_pos = (start_pos + uint(tree_depths[node_idx])) % max_rope_len;
    uint slot = (write_pos + node_idx) % max_seq_len;
    uint hd_idx = tid % half_head_dim;

    if (tid < q_pairs) {
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[rope_pos * half_head_dim + hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        qkv_node[tid] = half2(res);
    } else if (tid < q_pairs + kv_pairs) {
        uint k_tid = tid - q_pairs;
        uint kv_head = k_tid / half_head_dim;
        float2 x_val = float2(raw_val);
        float2 f_val = freqs_cis[rope_pos * half_head_dim + hd_idx];
        float2 res;
        res.x = fma(x_val.x, f_val.x, -x_val.y * f_val.y);
        res.y = fma(x_val.x, f_val.y, x_val.y * f_val.x);
        uint dst_idx = (kv_head * max_seq_len + slot) * half_head_dim + hd_idx;
        k_cache[dst_idx] = half2(res);
    } else {
        uint v_tid = tid - (q_pairs + kv_pairs);
        uint kv_head = v_tid / half_head_dim;
        uint dst_idx = (kv_head * max_seq_len + slot) * half_head_dim + hd_idx;
        v_cache[dst_idx] = raw_val;
    }
}

// ----------------------------------------------------------------------------
// 7D. SpecInfer Topology-Aware Tree Attention Split-KV Kernel (T <= 32, FP16)
// Evaluates 4 tree nodes simultaneously per SIMD-group, sharing prefix KV loads
// 4x across registers and enforcing exact branching topology via 32-bit
// ancestor bitmasks (tree_mask[u] bit j == 1 iff node u attends to tree node j).
// ----------------------------------------------------------------------------
kernel void gqa_tree_verify_split_kv_half_kernel(
    device const half *q [[buffer(0)]],
    device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]],
    device half *out [[buffer(3)]],
    device half2 *partial_O [[buffer(4)]],
    device float *partial_m [[buffer(5)]],
    device float *partial_l [[buffer(6)]],
    constant const uint *tree_mask [[buffer(7)]],
    constant uint &max_seq_len [[buffer(8)]],
    constant uint &prefix_len [[buffer(9)]],
    constant uint &write_pos [[buffer(10)]],
    constant uint &T [[buffer(11)]],
    constant uint &num_splits [[buffer(12)]],
    constant float &scale [[buffer(13)]],
    constant uint &n_head [[buffer(14)]],
    constant uint &n_kv_head [[buffer(15)]],
    constant uint &head_dim [[buffer(16)]],
    constant uint &q_stride [[buffer(17)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    uint qh = tgpig.x;
    uint node_base = tgpig.y * 4u;
    uint split_id = tgpig.z;
    uint lane_id = simd_lane_id;

    if (qh >= n_head || node_base >= T) return;
    uint u_cnt = min(T - node_base, 4u);
    uint n_rep = n_head / n_kv_head;
    uint kv_h = qh / n_rep;
    uint half_hd = head_dim >> 1;

    bool has_lo = (lane_id < half_hd);
    bool has_hi = ((lane_id + 32u) < half_hd);

    half2 q_lo[4];
    half2 q_hi[4];
    float m_arr[4];
    float l_arr[4];
    float2 o_lo[4];
    float2 o_hi[4];

    #pragma unroll
    for (uint u = 0; u < 4; ++u) {
        m_arr[u] = -1e38f;
        l_arr[u] = 0.0f;
        o_lo[u] = float2(0.0f);
        o_hi[u] = float2(0.0f);
        if (u < u_cnt) {
            device const half2 *q_ptr = (device const half2 *)(q + (node_base + u) * q_stride + qh * head_dim);
            q_lo[u] = has_lo ? q_ptr[lane_id] : half2(0.0h);
            q_hi[u] = has_hi ? q_ptr[lane_id + 32u] : half2(0.0h);
        } else {
            q_lo[u] = half2(0.0h);
            q_hi[u] = half2(0.0h);
        }
    }

    device const half *k_base = k_cache + kv_h * max_seq_len * head_dim;
    device const half *v_base = v_cache + kv_h * max_seq_len * head_dim;

    // 1. Committed prefix tokens [t_start, t_end): shared across all 4 tree nodes!
    uint tokens_per_split = (prefix_len > 0u) ? ((prefix_len + num_splits - 1u) / num_splits) : 0u;
    uint t_start = min(split_id * tokens_per_split, prefix_len);
    uint t_end = min(t_start + tokens_per_split, prefix_len);

    for (uint t = t_start; t < t_end; ++t) {
        device const half2 *k_ptr = (device const half2 *)(k_base + t * head_dim);
        device const half2 *v_ptr = (device const half2 *)(v_base + t * head_dim);
        float2 k_lo = has_lo ? float2(k_ptr[lane_id]) : float2(0.0f);
        float2 k_hi = has_hi ? float2(k_ptr[lane_id + 32u]) : float2(0.0f);
        float2 v_lo = has_lo ? float2(v_ptr[lane_id]) : float2(0.0f);
        float2 v_hi = has_hi ? float2(v_ptr[lane_id + 32u]) : float2(0.0f);

        for (uint u = 0; u < u_cnt; ++u) {
            float p = fma(float(q_lo[u].x), k_lo.x, float(q_lo[u].y) * k_lo.y);
            if (has_hi) {
                p += fma(float(q_hi[u].x), k_hi.x, float(q_hi[u].y) * k_hi.y);
            }
            float score = simd_sum(p) * scale;
            float m_new = max(m_arr[u], score);
            float a = fast::exp(m_arr[u] - m_new);
            float b = fast::exp(score - m_new);
            l_arr[u] = fma(l_arr[u], a, b);
            m_arr[u] = m_new;
            o_lo[u] = o_lo[u] * a + v_lo * b;
            if (has_hi) {
                o_hi[u] = o_hi[u] * a + v_hi * b;
            }
        }
    }

    // 2. Speculative tree tokens [0, T): evaluated once on split_id == 0 via bitmask!
    if (split_id == 0u) {
        uint masks[4];
        #pragma unroll
        for (uint u = 0; u < 4; ++u) {
            masks[u] = (u < u_cnt) ? tree_mask[node_base + u] : 0u;
        }
        uint union_mask = masks[0] | masks[1] | masks[2] | masks[3];

        for (uint j = 0; j < T; ++j) {
            uint bit = 1u << j;
            if ((union_mask & bit) == 0u) continue;

            uint slot = (write_pos + j) % max_seq_len;
            device const half2 *k_ptr = (device const half2 *)(k_base + slot * head_dim);
            device const half2 *v_ptr = (device const half2 *)(v_base + slot * head_dim);
            float2 k_lo = has_lo ? float2(k_ptr[lane_id]) : float2(0.0f);
            float2 k_hi = has_hi ? float2(k_ptr[lane_id + 32u]) : float2(0.0f);
            float2 v_lo = has_lo ? float2(v_ptr[lane_id]) : float2(0.0f);
            float2 v_hi = has_hi ? float2(v_ptr[lane_id + 32u]) : float2(0.0f);

            for (uint u = 0; u < u_cnt; ++u) {
                if ((masks[u] & bit) != 0u) {
                    float p = fma(float(q_lo[u].x), k_lo.x, float(q_lo[u].y) * k_lo.y);
                    if (has_hi) {
                        p += fma(float(q_hi[u].x), k_hi.x, float(q_hi[u].y) * k_hi.y);
                    }
                    float score = simd_sum(p) * scale;
                    float m_new = max(m_arr[u], score);
                    float a = fast::exp(m_arr[u] - m_new);
                    float b = fast::exp(score - m_new);
                    l_arr[u] = fma(l_arr[u], a, b);
                    m_arr[u] = m_new;
                    o_lo[u] = o_lo[u] * a + v_lo * b;
                    if (has_hi) {
                        o_hi[u] = o_hi[u] * a + v_hi * b;
                    }
                }
            }
        }
    }

    uint out_stride = n_head * head_dim;
    if (num_splits == 1u) {
        for (uint u = 0; u < u_cnt; ++u) {
            device half2 *out_ptr = (device half2 *)(out + (node_base + u) * out_stride + qh * head_dim);
            float inv_l = (l_arr[u] > 0.0f) ? (1.0f / l_arr[u]) : 0.0f;
            if (has_lo) out_ptr[lane_id] = half2(o_lo[u] * inv_l);
            if (has_hi) out_ptr[lane_id + 32u] = half2(o_hi[u] * inv_l);
        }
    } else {
        for (uint u = 0; u < u_cnt; ++u) {
            uint row_id = (node_base + u) * n_head + qh;
            uint p_base = row_id * (num_splits * half_hd) + split_id * half_hd;
            if (has_lo) partial_O[p_base + lane_id] = half2(o_lo[u]);
            if (has_hi) partial_O[p_base + lane_id + 32u] = half2(o_hi[u]);
            if (lane_id == 0u) {
                uint ml_idx = row_id * num_splits + split_id;
                partial_m[ml_idx] = m_arr[u];
                partial_l[ml_idx] = l_arr[u];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 7E. SpecInfer Tree Attention Split-KV Reduction Kernel (FP16)
// ----------------------------------------------------------------------------
kernel void gqa_tree_verify_reduce_half_kernel(
    device const half2 *partial_O [[buffer(0)]],
    device const float *partial_m [[buffer(1)]],
    device const float *partial_l [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant uint &num_splits [[buffer(4)]],
    constant uint &n_head [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    uint qh = tgpig.x;
    uint node_idx = tgpig.y;
    uint lane_id = simd_lane_id;
    uint half_hd = head_dim >> 1;
    uint row_id = node_idx * n_head + qh;

    threadgroup float shared_buf[2];

    if (lane_id == 0u) {
        float local_max = -1e38f;
        for (uint s = 0; s < num_splits; ++s) {
            float m_val = partial_m[row_id * num_splits + s];
            if (m_val > local_max) local_max = m_val;
        }
        float global_l = 0.0f;
        for (uint s = 0; s < num_splits; ++s) {
            float m_val = partial_m[row_id * num_splits + s];
            float l_val = partial_l[row_id * num_splits + s];
            global_l += l_val * fast::exp(m_val - local_max);
        }
        shared_buf[0] = local_max;
        shared_buf[1] = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_m = shared_buf[0];
    float inv_l = shared_buf[1];
    device half2 *out_head = (device half2 *)(out + (node_idx * n_head + qh) * head_dim);

    for (uint c = lane_id; c < half_hd; c += 32u) {
        float2 o_acc = float2(0.0f);
        for (uint s = 0; s < num_splits; ++s) {
            float m_val = partial_m[row_id * num_splits + s];
            float weight = fast::exp(m_val - global_m);
            half2 o_part = partial_O[row_id * (num_splits * half_hd) + s * half_hd + c];
            o_acc += float2(o_part) * weight;
        }
        out_head[c] = half2(o_acc * inv_l);
    }
}

// ----------------------------------------------------------------------------
// 7F. In-Place DFS / Topological KV-Cache Compaction Kernel (FP16)
// Compacts accepted tree node KV slots (write_pos + accepted_indices[m]) into
// contiguous prefix positions (write_pos + m) across ALL layers in 1 dispatch!
// Race-free via private thread register staging (num_accepted <= 32).
// ----------------------------------------------------------------------------
kernel void compact_tree_kv_half_kernel(
    device half2 *cache_ptr [[buffer(0)]],
    constant const int *accepted_indices [[buffer(1)]],
    constant uint &n_layer [[buffer(2)]],
    constant uint &max_batch_size [[buffer(3)]],
    constant uint &n_kv_head [[buffer(4)]],
    constant uint &max_seq_len [[buffer(5)]],
    constant uint &half_head_dim [[buffer(6)]],
    constant uint &write_pos [[buffer(7)]],
    constant uint &num_accepted [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    uint total_streams = n_layer * 2u * n_kv_head * half_head_dim;
    if (tid >= total_streams || num_accepted == 0u) return;

    uint d_pair = tid % half_head_dim;
    uint rem = tid / half_head_dim;
    uint kv_h = rem % n_kv_head;
    rem /= n_kv_head;
    uint kv_type = rem & 1u;
    uint layer = rem >> 1;

    uint base_idx = (((layer * 2u + kv_type) * max_batch_size) * n_kv_head + kv_h) * max_seq_len * half_head_dim;

    half2 vals[32];
    uint cnt = min(num_accepted, 32u);
    for (uint m = 0; m < cnt; ++m) {
        uint src_slot = (write_pos + uint(accepted_indices[m])) % max_seq_len;
        vals[m] = cache_ptr[base_idx + src_slot * half_head_dim + d_pair];
    }
    for (uint m = 0; m < cnt; ++m) {
        uint dst_slot = (write_pos + m) % max_seq_len;
        cache_ptr[base_idx + dst_slot * half_head_dim + d_pair] = vals[m];
    }
}

// ============================================================================
// Phase 3 Task 3.2: In-Register Fused Quantized Flash-Decode Kernels
// ============================================================================

// ----------------------------------------------------------------------------
// 7A-FP8. Generalized GQA Vector Flash-Decode Split-KV Kernel (T_q = 1, FP8 KV Cache)
// Fuses in-flight OCP FP8 E4M3FN dequantization directly inside private SIMD registers
// during the QK^T dot product and V accumulation, completely eliminating O(N^2)
// DRAM reconstruction buffers during long-context decode up to 32K.
// ----------------------------------------------------------------------------
kernel void gqa_flash_decode_split_kv_fp8_kernel(
    device const half *q [[buffer(0)]],
    device const uchar2 *k_cache [[buffer(1)]],
    device const uchar2 *v_cache [[buffer(2)]],
    device half *out [[buffer(3)]],
    device half2 *partial_O [[buffer(4)]],
    device float *partial_m [[buffer(5)]],
    device float *partial_l [[buffer(6)]],
    device const half *k_scales [[buffer(7)]],
    device const half *v_scales [[buffer(8)]],
    constant uint &max_seq_len [[buffer(9)]],
    constant uint &active_len [[buffer(10)]],
    constant uint &num_splits [[buffer(11)]],
    constant float &scale [[buffer(12)]],
    constant uint &n_head [[buffer(13)]],
    constant uint &n_kv_head [[buffer(14)]],
    constant uint &head_dim [[buffer(15)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    uint kv_h = tgpig.x;
    uint split_id = tgpig.y;
    uint rep_tile = tgpig.z;
    uint lane_id = simd_lane_id;

    uint n_rep = n_head / n_kv_head;
    uint q_sub_base = rep_tile * 8u;
    if (kv_h >= n_kv_head || q_sub_base >= n_rep) return;
    uint q_cnt = min(n_rep - q_sub_base, 8u);
    uint half_hd = head_dim >> 1;

    bool has_lo = (lane_id < half_hd);
    bool has_hi = ((lane_id + 32u) < half_hd);

    half2 q_lo[8];
    half2 q_hi[8];
    float m_arr[8];
    float l_arr[8];
    float2 o_lo[8];
    float2 o_hi[8];

    #pragma unroll
    for (uint r = 0; r < 8; ++r) {
        m_arr[r] = -1e38f;
        l_arr[r] = 0.0f;
        o_lo[r] = float2(0.0f);
        o_hi[r] = float2(0.0f);
        if (r < q_cnt) {
            uint qh = kv_h * n_rep + q_sub_base + r;
            device const half2 *q_ptr = (device const half2 *)(q + qh * head_dim);
            q_lo[r] = has_lo ? q_ptr[lane_id] : half2(0.0h);
            q_hi[r] = has_hi ? q_ptr[lane_id + 32u] : half2(0.0h);
        } else {
            q_lo[r] = half2(0.0h);
            q_hi[r] = half2(0.0h);
        }
    }

    uint tokens_per_split = (active_len + num_splits - 1u) / num_splits;
    uint t_start = min(split_id * tokens_per_split, active_len);
    uint t_end = min(t_start + tokens_per_split, active_len);

    uint num_groups = max(1u, half_hd / 32u);
    uint kv_scale_stride = max_seq_len * num_groups;
    device const half *k_scale_base = k_scales + kv_h * kv_scale_stride;
    device const half *v_scale_base = v_scales + kv_h * kv_scale_stride;
    device const uchar2 *k_base = k_cache + kv_h * max_seq_len * half_hd;
    device const uchar2 *v_base = v_cache + kv_h * max_seq_len * half_hd;

    for (uint t = t_start; t < t_end; ++t) {
        device const uchar2 *k_ptr = k_base + t * half_hd;
        device const uchar2 *v_ptr = v_base + t * half_hd;
        device const half *k_sc = k_scale_base + t * num_groups;
        device const half *v_sc = v_scale_base + t * num_groups;

        half ks_lo = k_sc[0];
        half ks_hi = (num_groups > 1u) ? k_sc[1] : ks_lo;
        half vs_lo = v_sc[0];
        half vs_hi = (num_groups > 1u) ? v_sc[1] : vs_lo;

        uchar2 raw_k_lo = has_lo ? k_ptr[lane_id] : uchar2(0);
        uchar2 raw_k_hi = has_hi ? k_ptr[lane_id + 32u] : uchar2(0);
        uchar2 raw_v_lo = has_lo ? v_ptr[lane_id] : uchar2(0);
        uchar2 raw_v_hi = has_hi ? v_ptr[lane_id + 32u] : uchar2(0);

        float2 k_lo = has_lo ? float2(
            decode_fp8_e4m3fn(raw_k_lo.x) * ks_lo,
            decode_fp8_e4m3fn(raw_k_lo.y) * ks_lo
        ) : float2(0.0f);
        float2 k_hi = has_hi ? float2(
            decode_fp8_e4m3fn(raw_k_hi.x) * ks_hi,
            decode_fp8_e4m3fn(raw_k_hi.y) * ks_hi
        ) : float2(0.0f);

        float2 v_lo = has_lo ? float2(
            decode_fp8_e4m3fn(raw_v_lo.x) * vs_lo,
            decode_fp8_e4m3fn(raw_v_lo.y) * vs_lo
        ) : float2(0.0f);
        float2 v_hi = has_hi ? float2(
            decode_fp8_e4m3fn(raw_v_hi.x) * vs_hi,
            decode_fp8_e4m3fn(raw_v_hi.y) * vs_hi
        ) : float2(0.0f);

        for (uint r = 0; r < q_cnt; ++r) {
            float p = fma(float(q_lo[r].x), k_lo.x, float(q_lo[r].y) * k_lo.y);
            if (has_hi) {
                p += fma(float(q_hi[r].x), k_hi.x, float(q_hi[r].y) * k_hi.y);
            }
            float score = simd_sum(p) * scale;
            float m_new = max(m_arr[r], score);
            float a = fast::exp(m_arr[r] - m_new);
            float b = fast::exp(score - m_new);
            l_arr[r] = fma(l_arr[r], a, b);
            m_arr[r] = m_new;
            o_lo[r] = o_lo[r] * a + v_lo * b;
            if (has_hi) {
                o_hi[r] = o_hi[r] * a + v_hi * b;
            }
        }
    }

    if (num_splits == 1u) {
        for (uint r = 0; r < q_cnt; ++r) {
            uint qh = kv_h * n_rep + q_sub_base + r;
            device half2 *out_ptr = (device half2 *)(out + qh * head_dim);
            float inv_l = (l_arr[r] > 0.0f) ? (1.0f / l_arr[r]) : 0.0f;
            if (has_lo) out_ptr[lane_id] = half2(o_lo[r] * inv_l);
            if (has_hi) out_ptr[lane_id + 32u] = half2(o_hi[r] * inv_l);
        }
    } else {
        for (uint r = 0; r < q_cnt; ++r) {
            uint qh = kv_h * n_rep + q_sub_base + r;
            uint p_base = qh * (num_splits * half_hd) + split_id * half_hd;
            if (has_lo) partial_O[p_base + lane_id] = half2(o_lo[r]);
            if (has_hi) partial_O[p_base + lane_id + 32u] = half2(o_hi[r]);
            if (lane_id == 0u) {
                uint ml_idx = qh * num_splits + split_id;
                partial_m[ml_idx] = m_arr[r];
                partial_l[ml_idx] = l_arr[r];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 7A-INT4. Generalized GQA Vector Flash-Decode Split-KV Kernel (T_q = 1, INT4 KV Cache)
// Fuses in-flight symmetric INT4 dequantization directly inside private SIMD registers
// during the QK^T dot product and V accumulation, reading 0.5 Bytes/dim from DRAM.
// ----------------------------------------------------------------------------
kernel void gqa_flash_decode_split_kv_int4_kernel(
    device const half *q [[buffer(0)]],
    device const uchar *k_cache [[buffer(1)]],
    device const uchar *v_cache [[buffer(2)]],
    device half *out [[buffer(3)]],
    device half2 *partial_O [[buffer(4)]],
    device float *partial_m [[buffer(5)]],
    device float *partial_l [[buffer(6)]],
    device const half *k_scales [[buffer(7)]],
    device const half *v_scales [[buffer(8)]],
    constant uint &max_seq_len [[buffer(9)]],
    constant uint &active_len [[buffer(10)]],
    constant uint &num_splits [[buffer(11)]],
    constant float &scale [[buffer(12)]],
    constant uint &n_head [[buffer(13)]],
    constant uint &n_kv_head [[buffer(14)]],
    constant uint &head_dim [[buffer(15)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    uint kv_h = tgpig.x;
    uint split_id = tgpig.y;
    uint rep_tile = tgpig.z;
    uint lane_id = simd_lane_id;

    uint n_rep = n_head / n_kv_head;
    uint q_sub_base = rep_tile * 8u;
    if (kv_h >= n_kv_head || q_sub_base >= n_rep) return;
    uint q_cnt = min(n_rep - q_sub_base, 8u);
    uint half_hd = head_dim >> 1;

    bool has_lo = (lane_id < half_hd);
    bool has_hi = ((lane_id + 32u) < half_hd);

    half2 q_lo[8];
    half2 q_hi[8];
    float m_arr[8];
    float l_arr[8];
    float2 o_lo[8];
    float2 o_hi[8];

    #pragma unroll
    for (uint r = 0; r < 8; ++r) {
        m_arr[r] = -1e38f;
        l_arr[r] = 0.0f;
        o_lo[r] = float2(0.0f);
        o_hi[r] = float2(0.0f);
        if (r < q_cnt) {
            uint qh = kv_h * n_rep + q_sub_base + r;
            device const half2 *q_ptr = (device const half2 *)(q + qh * head_dim);
            q_lo[r] = has_lo ? q_ptr[lane_id] : half2(0.0h);
            q_hi[r] = has_hi ? q_ptr[lane_id + 32u] : half2(0.0h);
        } else {
            q_lo[r] = half2(0.0h);
            q_hi[r] = half2(0.0h);
        }
    }

    uint tokens_per_split = (active_len + num_splits - 1u) / num_splits;
    uint t_start = min(split_id * tokens_per_split, active_len);
    uint t_end = min(t_start + tokens_per_split, active_len);

    uint num_groups = max(1u, half_hd / 32u);
    uint kv_scale_stride = max_seq_len * num_groups;
    device const half *k_scale_base = k_scales + kv_h * kv_scale_stride;
    device const half *v_scale_base = v_scales + kv_h * kv_scale_stride;
    device const uchar *k_base = k_cache + kv_h * max_seq_len * half_hd;
    device const uchar *v_base = v_cache + kv_h * max_seq_len * half_hd;

    for (uint t = t_start; t < t_end; ++t) {
        device const uchar *k_ptr = k_base + t * half_hd;
        device const uchar *v_ptr = v_base + t * half_hd;
        device const half *k_sc = k_scale_base + t * num_groups;
        device const half *v_sc = v_scale_base + t * num_groups;

        half ks_lo = k_sc[0];
        half ks_hi = (num_groups > 1u) ? k_sc[1] : ks_lo;
        half vs_lo = v_sc[0];
        half vs_hi = (num_groups > 1u) ? v_sc[1] : vs_lo;

        uchar raw_k_lo = has_lo ? k_ptr[lane_id] : uchar(0);
        uchar raw_k_hi = has_hi ? k_ptr[lane_id + 32u] : uchar(0);
        uchar raw_v_lo = has_lo ? v_ptr[lane_id] : uchar(0);
        uchar raw_v_hi = has_hi ? v_ptr[lane_id + 32u] : uchar(0);

        half2 dec_k_lo = has_lo ? dequant_int4_packed(raw_k_lo, ks_lo) : half2(0.0h);
        half2 dec_k_hi = has_hi ? dequant_int4_packed(raw_k_hi, ks_hi) : half2(0.0h);
        half2 dec_v_lo = has_lo ? dequant_int4_packed(raw_v_lo, vs_lo) : half2(0.0h);
        half2 dec_v_hi = has_hi ? dequant_int4_packed(raw_v_hi, vs_hi) : half2(0.0h);

        float2 k_lo = float2(dec_k_lo);
        float2 k_hi = float2(dec_k_hi);
        float2 v_lo = float2(dec_v_lo);
        float2 v_hi = float2(dec_v_hi);

        for (uint r = 0; r < q_cnt; ++r) {
            float p = fma(float(q_lo[r].x), k_lo.x, float(q_lo[r].y) * k_lo.y);
            if (has_hi) {
                p += fma(float(q_hi[r].x), k_hi.x, float(q_hi[r].y) * k_hi.y);
            }
            float score = simd_sum(p) * scale;
            float m_new = max(m_arr[r], score);
            float a = fast::exp(m_arr[r] - m_new);
            float b = fast::exp(score - m_new);
            l_arr[r] = fma(l_arr[r], a, b);
            m_arr[r] = m_new;
            o_lo[r] = o_lo[r] * a + v_lo * b;
            if (has_hi) {
                o_hi[r] = o_hi[r] * a + v_hi * b;
            }
        }
    }

    if (num_splits == 1u) {
        for (uint r = 0; r < q_cnt; ++r) {
            uint qh = kv_h * n_rep + q_sub_base + r;
            device half2 *out_ptr = (device half2 *)(out + qh * head_dim);
            float inv_l = (l_arr[r] > 0.0f) ? (1.0f / l_arr[r]) : 0.0f;
            if (has_lo) out_ptr[lane_id] = half2(o_lo[r] * inv_l);
            if (has_hi) out_ptr[lane_id + 32u] = half2(o_hi[r] * inv_l);
        }
    } else {
        for (uint r = 0; r < q_cnt; ++r) {
            uint qh = kv_h * n_rep + q_sub_base + r;
            uint p_base = qh * (num_splits * half_hd) + split_id * half_hd;
            if (has_lo) partial_O[p_base + lane_id] = half2(o_lo[r]);
            if (has_hi) partial_O[p_base + lane_id + 32u] = half2(o_hi[r]);
            if (lane_id == 0u) {
                uint ml_idx = qh * num_splits + split_id;
                partial_m[ml_idx] = m_arr[r];
                partial_l[ml_idx] = l_arr[r];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 7D-FP8. SpecInfer Topology-Aware Tree Attention Split-KV Kernel (T <= 32, FP8 KV Cache)
// ----------------------------------------------------------------------------
kernel void gqa_tree_verify_split_kv_fp8_kernel(
    device const half *q [[buffer(0)]],
    device const uchar2 *k_cache [[buffer(1)]],
    device const uchar2 *v_cache [[buffer(2)]],
    device half *out [[buffer(3)]],
    device half2 *partial_O [[buffer(4)]],
    device float *partial_m [[buffer(5)]],
    device float *partial_l [[buffer(6)]],
    device const half *k_scales [[buffer(7)]],
    device const half *v_scales [[buffer(8)]],
    constant const uint *tree_mask [[buffer(9)]],
    constant uint &max_seq_len [[buffer(10)]],
    constant uint &prefix_len [[buffer(11)]],
    constant uint &write_pos [[buffer(12)]],
    constant uint &T [[buffer(13)]],
    constant uint &num_splits [[buffer(14)]],
    constant float &scale [[buffer(15)]],
    constant uint &n_head [[buffer(16)]],
    constant uint &n_kv_head [[buffer(17)]],
    constant uint &head_dim [[buffer(18)]],
    constant uint &q_stride [[buffer(19)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    uint qh = tgpig.x;
    uint node_base = tgpig.y * 4u;
    uint split_id = tgpig.z;
    uint lane_id = simd_lane_id;

    if (qh >= n_head || node_base >= T) return;
    uint u_cnt = min(T - node_base, 4u);
    uint n_rep = n_head / n_kv_head;
    uint kv_h = qh / n_rep;
    uint half_hd = head_dim >> 1;

    bool has_lo = (lane_id < half_hd);
    bool has_hi = ((lane_id + 32u) < half_hd);

    half2 q_lo[4];
    half2 q_hi[4];
    float m_arr[4];
    float l_arr[4];
    float2 o_lo[4];
    float2 o_hi[4];

    #pragma unroll
    for (uint u = 0; u < 4; ++u) {
        m_arr[u] = -1e38f;
        l_arr[u] = 0.0f;
        o_lo[u] = float2(0.0f);
        o_hi[u] = float2(0.0f);
        if (u < u_cnt) {
            device const half2 *q_ptr = (device const half2 *)(q + (node_base + u) * q_stride + qh * head_dim);
            q_lo[u] = has_lo ? q_ptr[lane_id] : half2(0.0h);
            q_hi[u] = has_hi ? q_ptr[lane_id + 32u] : half2(0.0h);
        } else {
            q_lo[u] = half2(0.0h);
            q_hi[u] = half2(0.0h);
        }
    }

    uint num_groups = max(1u, half_hd / 32u);
    uint kv_scale_stride = max_seq_len * num_groups;
    device const half *k_scale_base = k_scales + kv_h * kv_scale_stride;
    device const half *v_scale_base = v_scales + kv_h * kv_scale_stride;
    device const uchar2 *k_base = k_cache + kv_h * max_seq_len * half_hd;
    device const uchar2 *v_base = v_cache + kv_h * max_seq_len * half_hd;

    // 1. Committed prefix tokens [t_start, t_end)
    uint tokens_per_split = (prefix_len > 0u) ? ((prefix_len + num_splits - 1u) / num_splits) : 0u;
    uint t_start = min(split_id * tokens_per_split, prefix_len);
    uint t_end = min(t_start + tokens_per_split, prefix_len);

    for (uint t = t_start; t < t_end; ++t) {
        device const uchar2 *k_ptr = k_base + t * half_hd;
        device const uchar2 *v_ptr = v_base + t * half_hd;
        device const half *k_sc = k_scale_base + t * num_groups;
        device const half *v_sc = v_scale_base + t * num_groups;

        half ks_lo = k_sc[0];
        half ks_hi = (num_groups > 1u) ? k_sc[1] : ks_lo;
        half vs_lo = v_sc[0];
        half vs_hi = (num_groups > 1u) ? v_sc[1] : vs_lo;

        uchar2 raw_k_lo = has_lo ? k_ptr[lane_id] : uchar2(0);
        uchar2 raw_k_hi = has_hi ? k_ptr[lane_id + 32u] : uchar2(0);
        uchar2 raw_v_lo = has_lo ? v_ptr[lane_id] : uchar2(0);
        uchar2 raw_v_hi = has_hi ? v_ptr[lane_id + 32u] : uchar2(0);

        float2 k_lo = has_lo ? float2(decode_fp8_e4m3fn(raw_k_lo.x) * ks_lo, decode_fp8_e4m3fn(raw_k_lo.y) * ks_lo) : float2(0.0f);
        float2 k_hi = has_hi ? float2(decode_fp8_e4m3fn(raw_k_hi.x) * ks_hi, decode_fp8_e4m3fn(raw_k_hi.y) * ks_hi) : float2(0.0f);
        float2 v_lo = has_lo ? float2(decode_fp8_e4m3fn(raw_v_lo.x) * vs_lo, decode_fp8_e4m3fn(raw_v_lo.y) * vs_lo) : float2(0.0f);
        float2 v_hi = has_hi ? float2(decode_fp8_e4m3fn(raw_v_hi.x) * vs_hi, decode_fp8_e4m3fn(raw_v_hi.y) * vs_hi) : float2(0.0f);

        for (uint u = 0; u < u_cnt; ++u) {
            float p = fma(float(q_lo[u].x), k_lo.x, float(q_lo[u].y) * k_lo.y);
            if (has_hi) {
                p += fma(float(q_hi[u].x), k_hi.x, float(q_hi[u].y) * k_hi.y);
            }
            float score = simd_sum(p) * scale;
            float m_new = max(m_arr[u], score);
            float a = fast::exp(m_arr[u] - m_new);
            float b = fast::exp(score - m_new);
            l_arr[u] = fma(l_arr[u], a, b);
            m_arr[u] = m_new;
            o_lo[u] = o_lo[u] * a + v_lo * b;
            if (has_hi) {
                o_hi[u] = o_hi[u] * a + v_hi * b;
            }
        }
    }

    // 2. Speculative tree tokens [0, T)
    if (split_id == 0u) {
        uint masks[4];
        #pragma unroll
        for (uint u = 0; u < 4; ++u) {
            masks[u] = (u < u_cnt) ? tree_mask[node_base + u] : 0u;
        }
        uint union_mask = masks[0] | masks[1] | masks[2] | masks[3];

        for (uint j = 0; j < T; ++j) {
            uint bit = 1u << j;
            if ((union_mask & bit) == 0u) continue;

            uint slot = (write_pos + j) % max_seq_len;
            device const uchar2 *k_ptr = k_base + slot * half_hd;
            device const uchar2 *v_ptr = v_base + slot * half_hd;
            device const half *k_sc = k_scale_base + slot * num_groups;
            device const half *v_sc = v_scale_base + slot * num_groups;

            half ks_lo = k_sc[0];
            half ks_hi = (num_groups > 1u) ? k_sc[1] : ks_lo;
            half vs_lo = v_sc[0];
            half vs_hi = (num_groups > 1u) ? v_sc[1] : vs_lo;

            uchar2 raw_k_lo = has_lo ? k_ptr[lane_id] : uchar2(0);
            uchar2 raw_k_hi = has_hi ? k_ptr[lane_id + 32u] : uchar2(0);
            uchar2 raw_v_lo = has_lo ? v_ptr[lane_id] : uchar2(0);
            uchar2 raw_v_hi = has_hi ? v_ptr[lane_id + 32u] : uchar2(0);

            float2 k_lo = has_lo ? float2(decode_fp8_e4m3fn(raw_k_lo.x) * ks_lo, decode_fp8_e4m3fn(raw_k_lo.y) * ks_lo) : float2(0.0f);
            float2 k_hi = has_hi ? float2(decode_fp8_e4m3fn(raw_k_hi.x) * ks_hi, decode_fp8_e4m3fn(raw_k_hi.y) * ks_hi) : float2(0.0f);
            float2 v_lo = has_lo ? float2(decode_fp8_e4m3fn(raw_v_lo.x) * vs_lo, decode_fp8_e4m3fn(raw_v_lo.y) * vs_lo) : float2(0.0f);
            float2 v_hi = has_hi ? float2(decode_fp8_e4m3fn(raw_v_hi.x) * vs_hi, decode_fp8_e4m3fn(raw_v_hi.y) * vs_hi) : float2(0.0f);

            for (uint u = 0; u < u_cnt; ++u) {
                if ((masks[u] & bit) != 0u) {
                    float p = fma(float(q_lo[u].x), k_lo.x, float(q_lo[u].y) * k_lo.y);
                    if (has_hi) {
                        p += fma(float(q_hi[u].x), k_hi.x, float(q_hi[u].y) * k_hi.y);
                    }
                    float score = simd_sum(p) * scale;
                    float m_new = max(m_arr[u], score);
                    float a = fast::exp(m_arr[u] - m_new);
                    float b = fast::exp(score - m_new);
                    l_arr[u] = fma(l_arr[u], a, b);
                    m_arr[u] = m_new;
                    o_lo[u] = o_lo[u] * a + v_lo * b;
                    if (has_hi) {
                        o_hi[u] = o_hi[u] * a + v_hi * b;
                    }
                }
            }
        }
    }

    if (num_splits == 1u) {
        for (uint u = 0; u < u_cnt; ++u) {
            uint node_idx = node_base + u;
            device half2 *out_ptr = (device half2 *)(out + node_idx * q_stride + qh * head_dim);
            float inv_l = (l_arr[u] > 0.0f) ? (1.0f / l_arr[u]) : 0.0f;
            if (has_lo) out_ptr[lane_id] = half2(o_lo[u] * inv_l);
            if (has_hi) out_ptr[lane_id + 32u] = half2(o_hi[u] * inv_l);
        }
    } else {
        for (uint u = 0; u < u_cnt; ++u) {
            uint node_idx = node_base + u;
            uint p_base = (node_idx * n_head + qh) * (num_splits * half_hd) + split_id * half_hd;
            if (has_lo) partial_O[p_base + lane_id] = half2(o_lo[u]);
            if (has_hi) partial_O[p_base + lane_id + 32u] = half2(o_hi[u]);
            if (lane_id == 0u) {
                uint ml_idx = (node_idx * n_head + qh) * num_splits + split_id;
                partial_m[ml_idx] = m_arr[u];
                partial_l[ml_idx] = l_arr[u];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 7D-INT4. SpecInfer Topology-Aware Tree Attention Split-KV Kernel (T <= 32, INT4 KV Cache)
// ----------------------------------------------------------------------------
kernel void gqa_tree_verify_split_kv_int4_kernel(
    device const half *q [[buffer(0)]],
    device const uchar *k_cache [[buffer(1)]],
    device const uchar *v_cache [[buffer(2)]],
    device half *out [[buffer(3)]],
    device half2 *partial_O [[buffer(4)]],
    device float *partial_m [[buffer(5)]],
    device float *partial_l [[buffer(6)]],
    device const half *k_scales [[buffer(7)]],
    device const half *v_scales [[buffer(8)]],
    constant const uint *tree_mask [[buffer(9)]],
    constant uint &max_seq_len [[buffer(10)]],
    constant uint &prefix_len [[buffer(11)]],
    constant uint &write_pos [[buffer(12)]],
    constant uint &T [[buffer(13)]],
    constant uint &num_splits [[buffer(14)]],
    constant float &scale [[buffer(15)]],
    constant uint &n_head [[buffer(16)]],
    constant uint &n_kv_head [[buffer(17)]],
    constant uint &head_dim [[buffer(18)]],
    constant uint &q_stride [[buffer(19)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    uint qh = tgpig.x;
    uint node_base = tgpig.y * 4u;
    uint split_id = tgpig.z;
    uint lane_id = simd_lane_id;

    if (qh >= n_head || node_base >= T) return;
    uint u_cnt = min(T - node_base, 4u);
    uint n_rep = n_head / n_kv_head;
    uint kv_h = qh / n_rep;
    uint half_hd = head_dim >> 1;

    bool has_lo = (lane_id < half_hd);
    bool has_hi = ((lane_id + 32u) < half_hd);

    half2 q_lo[4];
    half2 q_hi[4];
    float m_arr[4];
    float l_arr[4];
    float2 o_lo[4];
    float2 o_hi[4];

    #pragma unroll
    for (uint u = 0; u < 4; ++u) {
        m_arr[u] = -1e38f;
        l_arr[u] = 0.0f;
        o_lo[u] = float2(0.0f);
        o_hi[u] = float2(0.0f);
        if (u < u_cnt) {
            device const half2 *q_ptr = (device const half2 *)(q + (node_base + u) * q_stride + qh * head_dim);
            q_lo[u] = has_lo ? q_ptr[lane_id] : half2(0.0h);
            q_hi[u] = has_hi ? q_ptr[lane_id + 32u] : half2(0.0h);
        } else {
            q_lo[u] = half2(0.0h);
            q_hi[u] = half2(0.0h);
        }
    }

    uint num_groups = max(1u, half_hd / 32u);
    uint kv_scale_stride = max_seq_len * num_groups;
    device const half *k_scale_base = k_scales + kv_h * kv_scale_stride;
    device const half *v_scale_base = v_scales + kv_h * kv_scale_stride;
    device const uchar *k_base = k_cache + kv_h * max_seq_len * half_hd;
    device const uchar *v_base = v_cache + kv_h * max_seq_len * half_hd;

    // 1. Committed prefix tokens [t_start, t_end)
    uint tokens_per_split = (prefix_len > 0u) ? ((prefix_len + num_splits - 1u) / num_splits) : 0u;
    uint t_start = min(split_id * tokens_per_split, prefix_len);
    uint t_end = min(t_start + tokens_per_split, prefix_len);

    for (uint t = t_start; t < t_end; ++t) {
        device const uchar *k_ptr = k_base + t * half_hd;
        device const uchar *v_ptr = v_base + t * half_hd;
        device const half *k_sc = k_scale_base + t * num_groups;
        device const half *v_sc = v_scale_base + t * num_groups;

        half ks_lo = k_sc[0];
        half ks_hi = (num_groups > 1u) ? k_sc[1] : ks_lo;
        half vs_lo = v_sc[0];
        half vs_hi = (num_groups > 1u) ? v_sc[1] : vs_lo;

        uchar raw_k_lo = has_lo ? k_ptr[lane_id] : uchar(0);
        uchar raw_k_hi = has_hi ? k_ptr[lane_id + 32u] : uchar(0);
        uchar raw_v_lo = has_lo ? v_ptr[lane_id] : uchar(0);
        uchar raw_v_hi = has_hi ? v_ptr[lane_id + 32u] : uchar(0);

        half2 dec_k_lo = has_lo ? dequant_int4_packed(raw_k_lo, ks_lo) : half2(0.0h);
        half2 dec_k_hi = has_hi ? dequant_int4_packed(raw_k_hi, ks_hi) : half2(0.0h);
        half2 dec_v_lo = has_lo ? dequant_int4_packed(raw_v_lo, vs_lo) : half2(0.0h);
        half2 dec_v_hi = has_hi ? dequant_int4_packed(raw_v_hi, vs_hi) : half2(0.0h);

        float2 k_lo = float2(dec_k_lo);
        float2 k_hi = float2(dec_k_hi);
        float2 v_lo = float2(dec_v_lo);
        float2 v_hi = float2(dec_v_hi);

        for (uint u = 0; u < u_cnt; ++u) {
            float p = fma(float(q_lo[u].x), k_lo.x, float(q_lo[u].y) * k_lo.y);
            if (has_hi) {
                p += fma(float(q_hi[u].x), k_hi.x, float(q_hi[u].y) * k_hi.y);
            }
            float score = simd_sum(p) * scale;
            float m_new = max(m_arr[u], score);
            float a = fast::exp(m_arr[u] - m_new);
            float b = fast::exp(score - m_new);
            l_arr[u] = fma(l_arr[u], a, b);
            m_arr[u] = m_new;
            o_lo[u] = o_lo[u] * a + v_lo * b;
            if (has_hi) {
                o_hi[u] = o_hi[u] * a + v_hi * b;
            }
        }
    }

    // 2. Speculative tree tokens [0, T)
    if (split_id == 0u) {
        uint masks[4];
        #pragma unroll
        for (uint u = 0; u < 4; ++u) {
            masks[u] = (u < u_cnt) ? tree_mask[node_base + u] : 0u;
        }
        uint union_mask = masks[0] | masks[1] | masks[2] | masks[3];

        for (uint j = 0; j < T; ++j) {
            uint bit = 1u << j;
            if ((union_mask & bit) == 0u) continue;

            uint slot = (write_pos + j) % max_seq_len;
            device const uchar *k_ptr = k_base + slot * half_hd;
            device const uchar *v_ptr = v_base + slot * half_hd;
            device const half *k_sc = k_scale_base + slot * num_groups;
            device const half *v_sc = v_scale_base + slot * num_groups;

            half ks_lo = k_sc[0];
            half ks_hi = (num_groups > 1u) ? k_sc[1] : ks_lo;
            half vs_lo = v_sc[0];
            half vs_hi = (num_groups > 1u) ? v_sc[1] : vs_lo;

            uchar raw_k_lo = has_lo ? k_ptr[lane_id] : uchar(0);
            uchar raw_k_hi = has_hi ? k_ptr[lane_id + 32u] : uchar(0);
            uchar raw_v_lo = has_lo ? v_ptr[lane_id] : uchar(0);
            uchar raw_v_hi = has_hi ? v_ptr[lane_id + 32u] : uchar(0);

            half2 dec_k_lo = has_lo ? dequant_int4_packed(raw_k_lo, ks_lo) : half2(0.0h);
            half2 dec_k_hi = has_hi ? dequant_int4_packed(raw_k_hi, ks_hi) : half2(0.0h);
            half2 dec_v_lo = has_lo ? dequant_int4_packed(raw_v_lo, vs_lo) : half2(0.0h);
            half2 dec_v_hi = has_hi ? dequant_int4_packed(raw_v_hi, vs_hi) : half2(0.0h);

            float2 k_lo = float2(dec_k_lo);
            float2 k_hi = float2(dec_k_hi);
            float2 v_lo = float2(dec_v_lo);
            float2 v_hi = float2(dec_v_hi);

            for (uint u = 0; u < u_cnt; ++u) {
                if ((masks[u] & bit) != 0u) {
                    float p = fma(float(q_lo[u].x), k_lo.x, float(q_lo[u].y) * k_lo.y);
                    if (has_hi) {
                        p += fma(float(q_hi[u].x), k_hi.x, float(q_hi[u].y) * k_hi.y);
                    }
                    float score = simd_sum(p) * scale;
                    float m_new = max(m_arr[u], score);
                    float a = fast::exp(m_arr[u] - m_new);
                    float b = fast::exp(score - m_new);
                    l_arr[u] = fma(l_arr[u], a, b);
                    m_arr[u] = m_new;
                    o_lo[u] = o_lo[u] * a + v_lo * b;
                    if (has_hi) {
                        o_hi[u] = o_hi[u] * a + v_hi * b;
                    }
                }
            }
        }
    }

    if (num_splits == 1u) {
        for (uint u = 0; u < u_cnt; ++u) {
            uint node_idx = node_base + u;
            device half2 *out_ptr = (device half2 *)(out + node_idx * q_stride + qh * head_dim);
            float inv_l = (l_arr[u] > 0.0f) ? (1.0f / l_arr[u]) : 0.0f;
            if (has_lo) out_ptr[lane_id] = half2(o_lo[u] * inv_l);
            if (has_hi) out_ptr[lane_id + 32u] = half2(o_hi[u] * inv_l);
        }
    } else {
        for (uint u = 0; u < u_cnt; ++u) {
            uint node_idx = node_base + u;
            uint p_base = (node_idx * n_head + qh) * (num_splits * half_hd) + split_id * half_hd;
            if (has_lo) partial_O[p_base + lane_id] = half2(o_lo[u]);
            if (has_hi) partial_O[p_base + lane_id + 32u] = half2(o_hi[u]);
            if (lane_id == 0u) {
                uint ml_idx = (node_idx * n_head + qh) * num_splits + split_id;
                partial_m[ml_idx] = m_arr[u];
                partial_l[ml_idx] = l_arr[u];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 7F-FP8. In-Place Topological KV-Cache Compaction Kernel (FP8)
// ----------------------------------------------------------------------------
kernel void compact_tree_kv_fp8_kernel(
    device uchar2 *cache_ptr [[buffer(0)]],
    device half *scales_ptr [[buffer(1)]],
    constant const int *accepted_indices [[buffer(2)]],
    constant uint &n_layer [[buffer(3)]],
    constant uint &max_batch_size [[buffer(4)]],
    constant uint &n_kv_head [[buffer(5)]],
    constant uint &max_seq_len [[buffer(6)]],
    constant uint &half_head_dim [[buffer(7)]],
    constant uint &write_pos [[buffer(8)]],
    constant uint &num_accepted [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    uint total_streams = n_layer * 2u * n_kv_head * half_head_dim;
    if (tid >= total_streams || num_accepted == 0u) return;

    uint d_pair = tid % half_head_dim;
    uint rem = tid / half_head_dim;
    uint kv_h = rem % n_kv_head;
    rem /= n_kv_head;
    uint kv_type = rem & 1u;
    uint layer = rem >> 1;

    uint num_groups = max(1u, half_head_dim / 32u);
    uint base_idx = (((layer * 2u + kv_type) * max_batch_size) * n_kv_head + kv_h) * max_seq_len * half_head_dim;
    uint scale_base_idx = (((layer * 2u + kv_type) * max_batch_size) * n_kv_head + kv_h) * max_seq_len * num_groups;

    uchar2 vals[32];
    uint cnt = min(num_accepted, 32u);
    for (uint m = 0; m < cnt; ++m) {
        uint src_slot = (write_pos + uint(accepted_indices[m])) % max_seq_len;
        vals[m] = cache_ptr[base_idx + src_slot * half_head_dim + d_pair];
    }
    for (uint m = 0; m < cnt; ++m) {
        uint dst_slot = (write_pos + m) % max_seq_len;
        cache_ptr[base_idx + dst_slot * half_head_dim + d_pair] = vals[m];
    }

    if (d_pair < num_groups && scales_ptr != nullptr) {
        half s_vals[32];
        for (uint m = 0; m < cnt; ++m) {
            uint src_slot = (write_pos + uint(accepted_indices[m])) % max_seq_len;
            s_vals[m] = scales_ptr[scale_base_idx + src_slot * num_groups + d_pair];
        }
        for (uint m = 0; m < cnt; ++m) {
            uint dst_slot = (write_pos + m) % max_seq_len;
            scales_ptr[scale_base_idx + dst_slot * num_groups + d_pair] = s_vals[m];
        }
    }
}

// ----------------------------------------------------------------------------
// 7F-INT4. In-Place Topological KV-Cache Compaction Kernel (INT4)
// ----------------------------------------------------------------------------
kernel void compact_tree_kv_int4_kernel(
    device uchar *cache_ptr [[buffer(0)]],
    device half *scales_ptr [[buffer(1)]],
    constant const int *accepted_indices [[buffer(2)]],
    constant uint &n_layer [[buffer(3)]],
    constant uint &max_batch_size [[buffer(4)]],
    constant uint &n_kv_head [[buffer(5)]],
    constant uint &max_seq_len [[buffer(6)]],
    constant uint &half_head_dim [[buffer(7)]],
    constant uint &write_pos [[buffer(8)]],
    constant uint &num_accepted [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    uint total_streams = n_layer * 2u * n_kv_head * half_head_dim;
    if (tid >= total_streams || num_accepted == 0u) return;

    uint d_pair = tid % half_head_dim;
    uint rem = tid / half_head_dim;
    uint kv_h = rem % n_kv_head;
    rem /= n_kv_head;
    uint kv_type = rem & 1u;
    uint layer = rem >> 1;

    uint num_groups = max(1u, half_head_dim / 32u);
    uint base_idx = (((layer * 2u + kv_type) * max_batch_size) * n_kv_head + kv_h) * max_seq_len * half_head_dim;
    uint scale_base_idx = (((layer * 2u + kv_type) * max_batch_size) * n_kv_head + kv_h) * max_seq_len * num_groups;

    uchar vals[32];
    uint cnt = min(num_accepted, 32u);
    for (uint m = 0; m < cnt; ++m) {
        uint src_slot = (write_pos + uint(accepted_indices[m])) % max_seq_len;
        vals[m] = cache_ptr[base_idx + src_slot * half_head_dim + d_pair];
    }
    for (uint m = 0; m < cnt; ++m) {
        uint dst_slot = (write_pos + m) % max_seq_len;
        cache_ptr[base_idx + dst_slot * half_head_dim + d_pair] = vals[m];
    }

    if (d_pair < num_groups && scales_ptr != nullptr) {
        half s_vals[32];
        for (uint m = 0; m < cnt; ++m) {
            uint src_slot = (write_pos + uint(accepted_indices[m])) % max_seq_len;
            s_vals[m] = scales_ptr[scale_base_idx + src_slot * num_groups + d_pair];
        }
        for (uint m = 0; m < cnt; ++m) {
            uint dst_slot = (write_pos + m) % max_seq_len;
            scales_ptr[scale_base_idx + dst_slot * num_groups + d_pair] = s_vals[m];
        }
    }
}

// ============================================================================
// Phase 3 Task 3.3: BitNet 1.58-bit Ternary & Q2_K Sub-4-Bit GEMV/GEMM Kernels
// ============================================================================

// ----------------------------------------------------------------------------
// 8A. BitNet 1.58-bit Matrix-Vector Multiply (M=1)
// Weights W in {-1, 0, +1} packed 4 ternary elements per byte (0.25 B/weight).
// 16 lanes per sub-warp process 1 group of 64 weights. 4 SIMDgroups = 16 rows/TG.
// Multiplications are replaced with branchless sign selection & accumulation.
// ----------------------------------------------------------------------------
kernel void qmv_ternary_half_kernel(
    device const half *x [[buffer(0)]],
    device const uchar *w_packed [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    device const half *bias [[buffer(6)]],
    constant uint &has_bias [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint sg_row_base = tgpig.x * 16u + (simd_group_id << 2);
    if (sg_row_base >= N) return;

    uint sub_id = (simd_group_id << 1) | (simd_lane_id >> 4);
    uint lane16 = simd_lane_id & 15u;
    uint row_base = tgpig.x * 16u + (sub_id << 1);

    uint K_packed = K >> 2;       // K / 4 bytes per row
    uint K_groups = K >> 6;       // K / 64 groups per row

    bool valid0 = (row_base < N);
    bool valid1 = (row_base + 1u < N);
    float acc0 = 0.0f, acc1 = 0.0f;

    device const uchar *w0 = w_packed + row_base * K_packed;
    device const uchar *w1 = w0 + K_packed;
    device const half *s0 = scales + row_base * K_groups;
    device const half *s1 = s0 + K_groups;

    for (uint g = 0; g < K_groups; ++g) {
        uint idx = (g << 4) + lane16;
        half4 x_vec = *(device const half4 *)(x + (idx << 2));
        float4 fx = float4(x_vec);

        if (valid0) {
            uchar b0 = w0[idx];
            float w_0 = float(int(b0 & 1u) - int((b0 >> 1) & 1u));
            float w_1 = float(int((b0 >> 2) & 1u) - int((b0 >> 3) & 1u));
            float w_2 = float(int((b0 >> 4) & 1u) - int((b0 >> 5) & 1u));
            float w_3 = float(int((b0 >> 6) & 1u) - int((b0 >> 7) & 1u));
            float p0 = fma(fx.x, w_0, fma(fx.y, w_1, fma(fx.z, w_2, fx.w * w_3)));
            acc0 = fma(p0, float(s0[g]), acc0);
        }
        if (valid1) {
            uchar b1 = w1[idx];
            float w_0 = float(int(b1 & 1u) - int((b1 >> 1) & 1u));
            float w_1 = float(int((b1 >> 2) & 1u) - int((b1 >> 3) & 1u));
            float w_2 = float(int((b1 >> 4) & 1u) - int((b1 >> 5) & 1u));
            float w_3 = float(int((b1 >> 6) & 1u) - int((b1 >> 7) & 1u));
            float p1 = fma(fx.x, w_0, fma(fx.y, w_1, fma(fx.z, w_2, fx.w * w_3)));
            acc1 = fma(p1, float(s1[g]), acc1);
        }
    }

    acc0 += simd_shuffle_xor(acc0, 8u);
    acc0 += simd_shuffle_xor(acc0, 4u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc0 += simd_shuffle_xor(acc0, 1u);

    acc1 += simd_shuffle_xor(acc1, 8u);
    acc1 += simd_shuffle_xor(acc1, 4u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);

    if (lane16 == 0u) {
        if (valid0) {
            float b_val = (has_bias != 0u && bias != nullptr) ? float(bias[row_base]) : 0.0f;
            y[row_base] = half(acc0 + b_val);
        }
        if (valid1) {
            float b_val = (has_bias != 0u && bias != nullptr) ? float(bias[row_base + 1u]) : 0.0f;
            y[row_base + 1u] = half(acc1 + b_val);
        }
    }
}

// ----------------------------------------------------------------------------
// 8B. BitNet 1.58-bit Batched Matrix-Vector Multiply (Tall-and-Skinny GEMM)
// Evaluates M tokens (tiled by 8) against BitNet 1.58-bit ternary weights.
// Cooperative SRAM x-tiling + multiplication-free ternary accumulation.
// ----------------------------------------------------------------------------
kernel void qmm_ternary_half_kernel(
    device const half *x [[buffer(0)]],
    device const uchar *w_packed [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &N [[buffer(6)]],
    device const half *bias [[buffer(7)]],
    constant uint &has_bias [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_base = tgpig.x * 32u;
    uint m_base = tgpig.y * 8u;
    if (tg_row_base >= N || m_base >= M) return;

    uint tid = (simd_group_id << 5) | simd_lane_id; // 0..127
    uint m_ld = tid >> 4;                           // 0..7
    uint c_ld = tid & 15u;                          // 0..15

    uint sub_id = (simd_group_id << 1) | (simd_lane_id >> 4); // 0..7
    uint lane16 = simd_lane_id & 15u;                         // 0..15
    uint row_base = tg_row_base + (sub_id << 2);

    uint K_packed = K >> 2;
    uint K_groups = K >> 6;
    uint actual_M = min(M - m_base, 8u);

    threadgroup float4 s_x[8][16];

    float acc[8][4] = {{0.0f}};

    for (uint g = 0; g < K_groups; ++g) {
        if (m_ld < actual_M) {
            uint in_k = (g << 6) + (c_ld << 2);
            device const half4 *x_ptr = (device const half4 *)(x + (m_base + m_ld) * K + in_k);
            s_x[m_ld][c_ld] = float4(*x_ptr);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint idx = (g << 4) + lane16;
        for (uint r = 0; r < 4; ++r) {
            uint row = row_base + r;
            if (row < N) {
                uchar b = w_packed[row * K_packed + idx];
                float w_0 = float(int(b & 1u) - int((b >> 1) & 1u));
                float w_1 = float(int((b >> 2) & 1u) - int((b >> 3) & 1u));
                float w_2 = float(int((b >> 4) & 1u) - int((b >> 5) & 1u));
                float w_3 = float(int((b >> 6) & 1u) - int((b >> 7) & 1u));

                float sc = float(scales[row * K_groups + g]);
                for (uint m = 0; m < actual_M; ++m) {
                    float4 fx = s_x[m][lane16];
                    float p = fma(fx.x, w_0, fma(fx.y, w_1, fma(fx.z, w_2, fx.w * w_3)));
                    acc[m][r] = fma(p, sc, acc[m][r]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint m = 0; m < 8; ++m) {
        if (m < actual_M) {
            for (uint r = 0; r < 4; ++r) {
                acc[m][r] += simd_shuffle_xor(acc[m][r], 8u);
                acc[m][r] += simd_shuffle_xor(acc[m][r], 4u);
                acc[m][r] += simd_shuffle_xor(acc[m][r], 2u);
                acc[m][r] += simd_shuffle_xor(acc[m][r], 1u);
            }
        }
    }

    if (lane16 == 0u) {
        for (uint m = 0; m < actual_M; ++m) {
            for (uint r = 0; r < 4; ++r) {
                uint row = row_base + r;
                if (row < N) {
                    float b_val = (has_bias != 0u && bias != nullptr) ? float(bias[row]) : 0.0f;
                    y[(m_base + m) * N + row] = half(acc[m][r] + b_val);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 8C. Q2_K 2-Bit Quantized Matrix-Vector Multiply (M=1)
// 2 bits per weight: values in {0, 1, 2, 3} mapped to scale * q + bias.
// 4 weights packed per byte (0.25 B/weight). Group size G=64.
// ----------------------------------------------------------------------------
kernel void qmv_q2_k_half_kernel(
    device const half *x [[buffer(0)]],
    device const uchar *w_packed [[buffer(1)]],
    device const half2 *scales_biases [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    device const half *bias [[buffer(6)]],
    constant uint &has_bias [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint sg_row_base = tgpig.x * 16u + (simd_group_id << 2);
    if (sg_row_base >= N) return;

    uint sub_id = (simd_group_id << 1) | (simd_lane_id >> 4);
    uint lane16 = simd_lane_id & 15u;
    uint row_base = tgpig.x * 16u + (sub_id << 1);

    uint K_packed = K >> 2;       // K / 4 bytes per row
    uint K_groups = K >> 6;       // K / 64 groups per row

    bool valid0 = (row_base < N);
    bool valid1 = (row_base + 1u < N);
    float acc0 = 0.0f, acc1 = 0.0f;

    device const uchar *w0 = w_packed + row_base * K_packed;
    device const uchar *w1 = w0 + K_packed;
    device const half2 *sb0 = scales_biases + row_base * K_groups;
    device const half2 *sb1 = sb0 + K_groups;

    for (uint g = 0; g < K_groups; ++g) {
        uint idx = (g << 4) + lane16;
        half4 x_vec = *(device const half4 *)(x + (idx << 2));
        float4 fx = float4(x_vec);
        float x_sum = (fx.x + fx.y) + (fx.z + fx.w);

        if (valid0) {
            uchar b0 = w0[idx];
            float4 q0 = float4(float(b0 & 3u), float((b0 >> 2) & 3u), float((b0 >> 4) & 3u), float((b0 >> 6) & 3u));
            half2 s0 = sb0[g];
            acc0 = fma(dot(fx, q0), float(s0.x), fma(float(s0.y), x_sum, acc0));
        }
        if (valid1) {
            uchar b1 = w1[idx];
            float4 q1 = float4(float(b1 & 3u), float((b1 >> 2) & 3u), float((b1 >> 4) & 3u), float((b1 >> 6) & 3u));
            half2 s1 = sb1[g];
            acc1 = fma(dot(fx, q1), float(s1.x), fma(float(s1.y), x_sum, acc1));
        }
    }

    acc0 += simd_shuffle_xor(acc0, 8u);
    acc0 += simd_shuffle_xor(acc0, 4u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc0 += simd_shuffle_xor(acc0, 1u);

    acc1 += simd_shuffle_xor(acc1, 8u);
    acc1 += simd_shuffle_xor(acc1, 4u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);

    if (lane16 == 0u) {
        if (valid0) {
            float b_val = (has_bias != 0u && bias != nullptr) ? float(bias[row_base]) : 0.0f;
            y[row_base] = half(acc0 + b_val);
        }
        if (valid1) {
            float b_val = (has_bias != 0u && bias != nullptr) ? float(bias[row_base + 1u]) : 0.0f;
            y[row_base + 1u] = half(acc1 + b_val);
        }
    }
}

// ----------------------------------------------------------------------------
// 8D. Q2_K 2-Bit Batched Matrix-Vector Multiply (Tall-and-Skinny GEMM)
// Evaluates M tokens (tiled by 8) against Q2_K affine 2-bit weights.
// Cooperative SRAM x-tiling + fused scale/bias evaluation.
// ----------------------------------------------------------------------------
kernel void qmm_q2_k_half_kernel(
    device const half *x [[buffer(0)]],
    device const uchar *w_packed [[buffer(1)]],
    device const half2 *scales_biases [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &N [[buffer(6)]],
    device const half *bias [[buffer(7)]],
    constant uint &has_bias [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_base = tgpig.x * 32u;
    uint m_base = tgpig.y * 8u;
    if (tg_row_base >= N || m_base >= M) return;

    uint tid = (simd_group_id << 5) | simd_lane_id; // 0..127
    uint m_ld = tid >> 4;                           // 0..7
    uint c_ld = tid & 15u;                          // 0..15

    uint sub_id = (simd_group_id << 1) | (simd_lane_id >> 4); // 0..7
    uint lane16 = simd_lane_id & 15u;                         // 0..15
    uint row_base = tg_row_base + (sub_id << 2);

    uint K_packed = K >> 2;
    uint K_groups = K >> 6;
    uint actual_M = min(M - m_base, 8u);

    threadgroup float4 s_x[8][16];
    threadgroup float s_xsum[8][16];

    float acc[8][4] = {{0.0f}};

    for (uint g = 0; g < K_groups; ++g) {
        if (m_ld < actual_M) {
            uint in_k = (g << 6) + (c_ld << 2);
            device const half4 *x_ptr = (device const half4 *)(x + (m_base + m_ld) * K + in_k);
            float4 fx = float4(*x_ptr);
            s_x[m_ld][c_ld] = fx;
            s_xsum[m_ld][c_ld] = (fx.x + fx.y) + (fx.z + fx.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint idx = (g << 4) + lane16;
        for (uint r = 0; r < 4; ++r) {
            uint row = row_base + r;
            if (row < N) {
                uchar b = w_packed[row * K_packed + idx];
                float4 q = float4(float(b & 3u), float((b >> 2) & 3u), float((b >> 4) & 3u), float((b >> 6) & 3u));
                half2 sb = scales_biases[row * K_groups + g];
                float sc = float(sb.x);
                float bi = float(sb.y);

                for (uint m = 0; m < actual_M; ++m) {
                    float4 fx = s_x[m][lane16];
                    float x_sum = s_xsum[m][lane16];
                    acc[m][r] = fma(dot(fx, q), sc, fma(bi, x_sum, acc[m][r]));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint m = 0; m < 8; ++m) {
        if (m < actual_M) {
            for (uint r = 0; r < 4; ++r) {
                acc[m][r] += simd_shuffle_xor(acc[m][r], 8u);
                acc[m][r] += simd_shuffle_xor(acc[m][r], 4u);
                acc[m][r] += simd_shuffle_xor(acc[m][r], 2u);
                acc[m][r] += simd_shuffle_xor(acc[m][r], 1u);
            }
        }
    }

    if (lane16 == 0u) {
        for (uint m = 0; m < actual_M; ++m) {
            for (uint r = 0; r < 4; ++r) {
                uint row = row_base + r;
                if (row < N) {
                    float b_val = (has_bias != 0u && bias != nullptr) ? float(bias[row]) : 0.0f;
                    y[(m_base + m) * N + row] = half(acc[m][r] + b_val);
                }
            }
        }
    }
}
