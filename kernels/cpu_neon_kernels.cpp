#include <arm_neon.h>
#include <cmath>
#include <torch/extension.h>
#include <tuple>
#include <vector>

// ============================================================================
// Axiom-LM Apple Silicon M3 Pro Vectorized ARM NEON SIMD Kernel Engine
// ============================================================================

namespace axiom {

inline float exact_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ----------------------------------------------------------------------------
// 1. Fused RMSNorm Forward Pass (ARM NEON SIMD)
// ----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor>
rmsnorm_forward_neon(torch::Tensor x, torch::Tensor weight, float eps) {
  TORCH_CHECK(x.is_contiguous(), "Input tensor x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be Float32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32,
              "weight must be Float32");

  int64_t dim = x.size(-1);
  int64_t num_rows = x.numel() / dim;

  auto out = torch::empty_like(x);
  auto rsqrt_cache = torch::empty({num_rows}, x.options());

  const float *x_ptr = x.data_ptr<float>();
  const float *w_ptr = weight.data_ptr<float>();
  float *out_ptr = out.data_ptr<float>();
  float *rsqrt_ptr = rsqrt_cache.data_ptr<float>();

  at::parallel_for(0, num_rows, 0, [&](int64_t start, int64_t end) {
    for (int64_t r = start; r < end; ++r) {
      const float *row_x = x_ptr + r * dim;
      float *row_out = out_ptr + r * dim;

      // Vectorized sum of squares: sum(x_i^2)
      float32x4_t sum_sq_vec = vdupq_n_f32(0.0f);
      int64_t i = 0;
      for (; i <= dim - 4; i += 4) {
        float32x4_t val = vld1q_f32(row_x + i);
        sum_sq_vec = vmlaq_f32(sum_sq_vec, val, val);
      }
      float sum_sq = vaddvq_f32(sum_sq_vec);
      for (; i < dim; ++i) {
        sum_sq += row_x[i] * row_x[i];
      }

      float mean_sq = sum_sq / static_cast<float>(dim);
      float rsqrt_val = 1.0f / std::sqrt(mean_sq + eps);
      rsqrt_ptr[r] = rsqrt_val;

      float32x4_t rsqrt_vec = vdupq_n_f32(rsqrt_val);

      // Vectorized scale and weight affine transformation
      i = 0;
      for (; i <= dim - 4; i += 4) {
        float32x4_t val = vld1q_f32(row_x + i);
        float32x4_t w = vld1q_f32(w_ptr + i);
        float32x4_t norm_val = vmulq_f32(val, rsqrt_vec);
        float32x4_t result = vmulq_f32(norm_val, w);
        vst1q_f32(row_out + i, result);
      }
      for (; i < dim; ++i) {
        row_out[i] = row_x[i] * rsqrt_val * w_ptr[i];
      }
    }
  });

  return std::make_tuple(out, rsqrt_cache);
}

// ----------------------------------------------------------------------------
// 2. Fused RMSNorm Backward Pass (ARM NEON SIMD)
// ----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor>
rmsnorm_backward_neon(torch::Tensor grad_y, torch::Tensor x,
                      torch::Tensor weight, torch::Tensor rsqrt_cache) {
  TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(rsqrt_cache.is_contiguous(), "rsqrt_cache must be contiguous");

  int64_t dim = x.size(-1);
  int64_t num_rows = x.numel() / dim;

  auto grad_x = torch::empty_like(x);
  auto grad_w = torch::zeros_like(weight);

  const float *gy_ptr = grad_y.data_ptr<float>();
  const float *x_ptr = x.data_ptr<float>();
  const float *w_ptr = weight.data_ptr<float>();
  const float *rsqrt_ptr = rsqrt_cache.data_ptr<float>();
  float *gx_ptr = grad_x.data_ptr<float>();
  float *gw_ptr = grad_w.data_ptr<float>();

  // Step A: Compute grad_x for each row in parallel
  at::parallel_for(0, num_rows, 0, [&](int64_t start, int64_t end) {
    for (int64_t r = start; r < end; ++r) {
      const float *row_gy = gy_ptr + r * dim;
      const float *row_x = x_ptr + r * dim;
      float *row_gx = gx_ptr + r * dim;
      float rsqrt_val = rsqrt_ptr[r];

      // Compute inner product: sum(gy_i * w_i * x_i)
      float32x4_t inner_vec = vdupq_n_f32(0.0f);
      int64_t i = 0;
      for (; i <= dim - 4; i += 4) {
        float32x4_t gy = vld1q_f32(row_gy + i);
        float32x4_t w = vld1q_f32(w_ptr + i);
        float32x4_t x_val = vld1q_f32(row_x + i);
        float32x4_t gy_w = vmulq_f32(gy, w);
        inner_vec = vmlaq_f32(inner_vec, gy_w, x_val);
      }
      float inner_sum = vaddvq_f32(inner_vec);
      for (; i < dim; ++i) {
        inner_sum += row_gy[i] * w_ptr[i] * row_x[i];
      }

      float scale_term = (inner_sum / static_cast<float>(dim)) *
                         (rsqrt_val * rsqrt_val * rsqrt_val);
      float32x4_t rsqrt_vec = vdupq_n_f32(rsqrt_val);
      float32x4_t scale_vec = vdupq_n_f32(scale_term);

      // Compute grad_x = rsqrt_val * (gy * w) - scale_term * x
      i = 0;
      for (; i <= dim - 4; i += 4) {
        float32x4_t gy = vld1q_f32(row_gy + i);
        float32x4_t w = vld1q_f32(w_ptr + i);
        float32x4_t x_val = vld1q_f32(row_x + i);
        float32x4_t term1 = vmulq_f32(vmulq_f32(gy, w), rsqrt_vec);
        float32x4_t term2 = vmulq_f32(x_val, scale_vec);
        float32x4_t gx = vsubq_f32(term1, term2);
        vst1q_f32(row_gx + i, gx);
      }
      for (; i < dim; ++i) {
        row_gx[i] =
            (row_gy[i] * w_ptr[i] * rsqrt_val) - (row_x[i] * scale_term);
      }
    }
  });

  // Step B: Accumulate grad_w across rows
  for (int64_t r = 0; r < num_rows; ++r) {
    const float *row_gy = gy_ptr + r * dim;
    const float *row_x = x_ptr + r * dim;
    float rsqrt_val = rsqrt_ptr[r];
    float32x4_t rsqrt_vec = vdupq_n_f32(rsqrt_val);

    int64_t i = 0;
    for (; i <= dim - 4; i += 4) {
      float32x4_t gy = vld1q_f32(row_gy + i);
      float32x4_t x_val = vld1q_f32(row_x + i);
      float32x4_t gw = vld1q_f32(gw_ptr + i);
      float32x4_t d_w_partial = vmulq_f32(vmulq_f32(gy, x_val), rsqrt_vec);
      gw = vaddq_f32(gw, d_w_partial);
      vst1q_f32(gw_ptr + i, gw);
    }
    for (; i < dim; ++i) {
      gw_ptr[i] += row_gy[i] * row_x[i] * rsqrt_val;
    }
  }

  return std::make_tuple(grad_x, grad_w);
}

// ----------------------------------------------------------------------------
// 3. Fused SwiGLU Forward Pass (High-Precision Parallel Pass)
// ----------------------------------------------------------------------------
torch::Tensor swiglu_forward_neon(torch::Tensor gate, torch::Tensor up) {
  TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
  TORCH_CHECK(up.is_contiguous(), "up must be contiguous");
  TORCH_CHECK(gate.sizes() == up.sizes(),
              "gate and up must have identical shapes");

  auto out = torch::empty_like(gate);
  int64_t total_elements = gate.numel();

  const float *g_ptr = gate.data_ptr<float>();
  const float *u_ptr = up.data_ptr<float>();
  float *out_ptr = out.data_ptr<float>();

  at::parallel_for(0, total_elements, 2048, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      float g = g_ptr[i];
      float u = u_ptr[i];
      float sig = exact_sigmoid(g);
      float silu_g = g * sig;
      out_ptr[i] = silu_g * u;
    }
  });

  return out;
}

// ----------------------------------------------------------------------------
// 4. Fused SwiGLU Backward Pass (High-Precision Parallel Pass)
// ----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor>
swiglu_backward_neon(torch::Tensor grad_y, torch::Tensor gate,
                     torch::Tensor up) {
  TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
  TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
  TORCH_CHECK(up.is_contiguous(), "up must be contiguous");

  auto grad_gate = torch::empty_like(gate);
  auto grad_up = torch::empty_like(up);
  int64_t total_elements = gate.numel();

  const float *gy_ptr = grad_y.data_ptr<float>();
  const float *g_ptr = gate.data_ptr<float>();
  const float *u_ptr = up.data_ptr<float>();
  float *gg_ptr = grad_gate.data_ptr<float>();
  float *gu_ptr = grad_up.data_ptr<float>();

  at::parallel_for(0, total_elements, 2048, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      float gy = gy_ptr[i];
      float g = g_ptr[i];
      float u = u_ptr[i];

      float sig = exact_sigmoid(g);
      float silu_g = g * sig;
      gu_ptr[i] = gy * silu_g;

      float d_silu = sig * (1.0f + g * (1.0f - sig));
      gg_ptr[i] = gy * u * d_silu;
    }
  });

  return std::make_tuple(grad_gate, grad_up);
}

} // namespace axiom

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward_neon", &axiom::rmsnorm_forward_neon,
        "Axiom RMSNorm Forward (ARM NEON)");
  m.def("rmsnorm_backward_neon", &axiom::rmsnorm_backward_neon,
        "Axiom RMSNorm Backward (ARM NEON)");
  m.def("swiglu_forward_neon", &axiom::swiglu_forward_neon,
        "Axiom SwiGLU Forward (ARM NEON)");
  m.def("swiglu_backward_neon", &axiom::swiglu_backward_neon,
        "Axiom SwiGLU Backward (ARM NEON)");
}
