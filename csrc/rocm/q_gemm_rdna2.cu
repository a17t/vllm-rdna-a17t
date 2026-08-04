// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 GPTQ kernel for RDNA2 (gfx1030), templated on the activation dtype
// (half or __hip_bfloat16). Adapted from q_gemm_rdna3.cu with these changes:
//
//   1. No WMMA — RDNA2 lacks v_wmma_*; both fp16 and bf16 widen to fp32.
//   2. No v_dot2_f32_f16 — RDNA2 lacks this; uses explicit fp32 widening
//      + v_fma_f32 for fp16 path.
//   3. No v_dot2_f32_bf16 — RDNA2 lacks this; uses explicit fp32 widening
//      + v_fma_f32 for bf16 path.
//   4. Atomic CAS-loop for fp16/bf16 output (same as RDNA3).
//   5. Wave32 geometry: THREADS_X=256, BLOCK_KN_SIZE=256.

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "qdq_4_rdna2.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace gptq_rdna2 {

using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;

#define BLOCK_KN_SIZE 256
#define THREADS_X 256

// ---------------------------------------------------------------------------
// Per-dtype helpers
// ---------------------------------------------------------------------------

template <typename T>
__forceinline__ __device__ T tzero();

template <>
__forceinline__ __device__ half tzero<half>() {
  return __float2half_rn(0.0f);
}

template <>
__forceinline__ __device__ bf16_t tzero<bf16_t>() {
  return __float2bfloat16(0.0f);
}

// RDNA2 has no v_dot2_f32_f16 or v_pk_fma_f16 — widen both halfs to fp32
// and accumulate with v_fma_f32.
__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  float result = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t aw, dw;
    __builtin_memcpy(&aw, a_ptr + 2 * i, sizeof(uint32_t));
    __builtin_memcpy(&dw, &dq[i], sizeof(uint32_t));
    float a_x = __uint_as_float((aw & 0xFFFFu) << 16);
    float a_y = __uint_as_float(aw & 0xFFFF0000u);
    float d_x = __uint_as_float((dw & 0xFFFFu) << 16);
    float d_y = __uint_as_float(dw & 0xFFFF0000u);
    result = __fmaf_rn(d_x, a_x, result);
    result = __fmaf_rn(d_y, a_y, result);
  }
  return result;
}

// Same for bf16 — widen to fp32 and use v_fma_f32.
__forceinline__ __device__ float dot22_8_f(bf162_t (&dq)[4], const bf16_t* a_ptr) {
  float result = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t aw, dw;
    __builtin_memcpy(&aw, a_ptr + 2 * i, sizeof(uint32_t));
    __builtin_memcpy(&dw, &dq[i], sizeof(uint32_t));
    float a_x = __uint_as_float((aw & 0xFFFFu) << 16);
    float a_y = __uint_as_float(aw & 0xFFFF0000u);
    float d_x = __uint_as_float((dw & 0xFFFFu) << 16);
    float d_y = __uint_as_float(dw & 0xFFFF0000u);
    result = __fmaf_rn(d_x, a_x, result);
    result = __fmaf_rn(d_y, a_y, result);
  }
  return result;
}

// Packed atomic-add via CAS-loop (same as RDNA3 — gfx10 has no native
// v_global_atomic_pk_add_f16).
__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                    half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

__forceinline__ __device__ void atomic_add_pk4_bf16(bf16_t* addr, bf162_t v01,
                                                     bf162_t v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      bf162_t b2[2];
    } cur, sum;
    cur.u = old;
    sum.b2[0] = __hadd2(cur.b2[0], v01);
    sum.b2[1] = __hadd2(cur.b2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// Load 4 zero-point nibbles from a [groups, N/8] uint32 tensor.
__forceinline__ __device__ void load4_zeros(const uint32_t* qzeros_row, int n,
                                             int (&zeros)[4]) {
  int qcol = n / 8;
  int shift = (n & 0x07) * 4;
  uint32_t d = qzeros_row[qcol] >> shift;
  zeros[0] = (int)(d & 0xF);
  zeros[1] = (int)((d >> 4) & 0xF);
  zeros[2] = (int)((d >> 8) & 0xF);
  zeros[3] = (int)((d >> 12) & 0xF);
}

template <typename T>
__forceinline__ __device__ void load4_scales(const T* scales_row, int n,
                                              T (&scales)[4]) {
  scales[0] = scales_row[n + 0];
  scales[1] = scales_row[n + 1];
  scales[2] = scales_row[n + 2];
  scales[3] = scales_row[n + 3];
}

// BF16→fp32 scalar dequant: produces fp32 z and y for bias correction.
__forceinline__ __device__ void prep_zero_scale_bf16_f32(uint32_t zero,
                                                           bf16_t scale,
                                                           float& z_f, float& y_f) {
  float sf = __bfloat162float(scale);
  z_f = -(128.0f + (float)zero) * sf;
  y_f = sf;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <typename T, int M_COUNT>
__global__ void gemm_q4_kernel_rdna2(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int offset_m = blockIdx.y * M_COUNT;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  constexpr int LDS_PAD = 8;
  __shared__ T block_a[M_COUNT][BLOCK_KN_SIZE + LDS_PAD];

  static_assert(BLOCK_KN_SIZE == THREADS_X,
                "BLOCK_KN_SIZE must equal THREADS_X");
  // RDNA2: no v_dot2 for bf16, always stage A through LDS.
  if (offset_k + t < end_k) {
#pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
      T av;
      if (offset_m + m < size_m) {
        const T* a_row = a + (offset_m + m) * size_k;
        if (b_q_perm)
          av = a_row[b_q_perm[offset_k + t]];
        else
          av = a_row[offset_k + t];
      } else {
        av = tzero<T>();
      }
      block_a[m][t] = av;
    }
  }
  __syncthreads();
  if (n >= size_n) return;

  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  int qk = offset_k / 8;
  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  half2 z1z16_h[4][2], y1y16_h[4][2];
  float z_b_f[4], y_b_f[4];

  auto refresh_group = [&](int g) {
    const uint32_t* qz_row = b_qzeros + g * (size_n / 8);
    const T* sc_row = b_scales + g * size_n;
    int zeros[4];
    T scales[4];
    load4_zeros(qz_row, n, zeros);
    load4_scales<T>(sc_row, n, scales);
    if constexpr (std::is_same<T, half>::value) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset), scales[i],
                              z1z16_h[i], y1y16_h[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        prep_zero_scale_bf16_f32((uint32_t)(zeros[i] + zero_offset), scales[i],
                                  z_b_f[i], y_b_f[i]);
      }
    }
  };

  refresh_group(group);

  float block_c[M_COUNT][4];
#pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
#pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group(group);
    }

    int4 b_w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      b_w[j] = *(const int4*)(b_ptr + j * size_n);
    }
    b_ptr += 4 * size_n;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;

      if constexpr (std::is_same<T, half>::value) {
        half2 dq[4][4];
        dequant_4bit_8_fp16((uint32_t)b_w[j].x, dq[0], z1z16_h[0], y1y16_h[0]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].y, dq[1], z1z16_h[1], y1y16_h[1]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].z, dq[2], z1z16_h[2], y1y16_h[2]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].w, dq[3], z1z16_h[3], y1y16_h[3]);

#pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          const half* a_ptr = reinterpret_cast<const half*>(&block_a[m][a_off]);
          block_c[m][0] += dot22_8_f(dq[0], a_ptr);
          block_c[m][1] += dot22_8_f(dq[1], a_ptr);
          block_c[m][2] += dot22_8_f(dq[2], a_ptr);
          block_c[m][3] += dot22_8_f(dq[3], a_ptr);
        }
      } else {
        // bf16 path: widen to fp32, use v_fma_f32 for dot, bias correct
        // with z_b_f / y_b_f (same as RDNA3 bf16 M>1 path minus v_dot2).
        typedef short __attribute__((ext_vector_type(2))) bf16x2_t;
        union pack4 {
          float f[4];
          uint32_t u[4];
        };

        uint32_t w[4];
        __builtin_memcpy(w, &b_w[j], sizeof(int4));

        pack4 a_pack[M_COUNT];
#pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          const uint32_t* a_words =
              reinterpret_cast<const uint32_t*>(&block_a[m][a_off]);
          a_pack[m].u[0] = a_words[0];
          a_pack[m].u[1] = a_words[1];
          a_pack[m].u[2] = a_words[2];
          a_pack[m].u[3] = a_words[3];
        }

        // sum_a[m] = Σ a[m][i] via fp32 widening (no v_dot2 on RDNA2).
        float sum_a[M_COUNT];
#pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          float s = 0.0f;
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            uint32_t aw = a_pack[m].u[b];
            float a_x = __uint_as_float((aw & 0xFFFFu) << 16);
            float a_y = __uint_as_float(aw & 0xFFFF0000u);
            s = __fmaf_rn(1.0f, a_x, s);
            s = __fmaf_rn(1.0f, a_y, s);
          }
          sum_a[m] = s;
        }

#pragma unroll 1
        for (int col = 0; col < 4; ++col) {
          pack4 q_pack;
          const uint32_t qa = w[col];
          // Build bf16(128 + nibble) pairs via bit manipulation
          q_pack.u[0] = ((qa >> 0) & 0x000F000Fu) | 0x43004300u;
          q_pack.u[1] = ((qa >> 4) & 0x000F000Fu) | 0x43004300u;
          q_pack.u[2] = ((qa >> 8) & 0x000F000Fu) | 0x43004300u;
          q_pack.u[3] = ((qa >> 12) & 0x000F000Fu) | 0x43004300u;

#pragma unroll
          for (int m = 0; m < M_COUNT; ++m) {
            float partial = 0.0f;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
              uint32_t aw = a_pack[m].u[b];
              uint32_t qw = q_pack.u[b];
              float a_x = __uint_as_float((aw & 0xFFFFu) << 16);
              float a_y = __uint_as_float(aw & 0xFFFF0000u);
              float q_x = __uint_as_float((qw & 0xFFFFu) << 16);
              float q_y = __uint_as_float(qw & 0xFFFF0000u);
              partial = __fmaf_rn(q_x, a_x, partial);
              partial = __fmaf_rn(q_y, a_y, partial);
            }
            block_c[m][col] =
                __fmaf_rn(y_b_f[col], partial,
                          __fmaf_rn(z_b_f[col], sum_a[m], block_c[m][col]));
          }
        }
      }
    }
    k += 32;
  }

#pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    if (offset_m + m >= size_m) continue;
    T* out = c + (offset_m + m) * size_n + n;
    if constexpr (std::is_same<T, half>::value) {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                  __float2half_rn(block_c[m][1]));
      half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                  __float2half_rn(block_c[m][3]));
      atomic_add_pk4_f16(out, r01, r23);
    } else {
      bf162_t r01;
      r01.x = __float2bfloat16(block_c[m][0]);
      r01.y = __float2bfloat16(block_c[m][1]);
      bf162_t r23;
      r23.x = __float2bfloat16(block_c[m][2]);
      r23.y = __float2bfloat16(block_c[m][3]);
      atomic_add_pk4_bf16(out, r01, r23);
    }
  }
}

#else

template <typename T, int M_COUNT>
__global__ void gemm_q4_kernel_rdna2(const T*, const uint32_t*, const uint32_t*,
                                      const T*, T*, const int, const int,
                                      const int, const int, const int,
                                      const int*) {}

#endif

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename T, int M_COUNT>
void launch_gemm_q4_for_mcount(const T* a, const uint32_t* b_q_weight,
                                const uint32_t* b_qzeros, const T* b_scales,
                                const int* b_q_perm, T* c, int size_m,
                                int size_n, int size_k, int groups,
                                int zero_offset, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  gemm_q4_kernel_rdna2<T, M_COUNT><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
}

template <typename T>
void launch_gemm_q4(const T* a, const uint32_t* b_q_weight,
                     const uint32_t* b_qzeros, const T* b_scales,
                     const int* b_q_perm, T* c, int size_m, int size_n,
                     int size_k, int groups, bool use_v2_format,
                     cudaStream_t stream) {
  const int zero_offset = use_v2_format ? 0 : 1;

  if (size_m == 1) {
    launch_gemm_q4_for_mcount<T, 1>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  } else if (size_m <= 3) {
    launch_gemm_q4_for_mcount<T, 2>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  } else if (size_m <= 7) {
    launch_gemm_q4_for_mcount<T, 4>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  } else {
    launch_gemm_q4_for_mcount<T, 8>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  }
}

} // namespace gptq_rdna2
} // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

torch::Tensor gptq_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                               torch::Tensor b_qzeros, torch::Tensor b_scales,
                               torch::Tensor b_g_idx, bool use_v2_format) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_qzeros.is_cuda(), "b_qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N]");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
      "a must be half or bfloat16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(1);
  int groups = (int)b_qzeros.size(0);

  TORCH_CHECK(b_q_weight.size(0) * 8 == size_k,
              "b_q_weight first dim must be K/8");
  TORCH_CHECK(b_scales.size(0) == groups,
              "b_scales must have same group count as qzeros");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales last dim must be N");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8 (64-bit atomic CAS)");

  auto opts = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  at::Tensor c = torch::zeros({size_m, size_n}, opts);

  const int* g_idx_ptr = nullptr;
  if (!b_g_idx.device().is_meta() && b_g_idx.numel() > 0) {
    TORCH_CHECK(b_g_idx.scalar_type() == torch::kInt32,
                "b_g_idx must be int32");
    g_idx_ptr = (const int*)b_g_idx.data_ptr();
  }

  if (a.scalar_type() == torch::kHalf) {
    vllm::gptq_rdna2::launch_gemm_q4<half>(
        (const half*)a.data_ptr(), (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(), (const half*)b_scales.data_ptr(),
        g_idx_ptr, (half*)c.data_ptr(), size_m, size_n, size_k, groups,
        use_v2_format, stream);
  } else {
    vllm::gptq_rdna2::launch_gemm_q4<vllm::gptq_rdna2::bf16_t>(
        (const vllm::gptq_rdna2::bf16_t*)a.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(),
        (const vllm::gptq_rdna2::bf16_t*)b_scales.data_ptr(), g_idx_ptr,
        (vllm::gptq_rdna2::bf16_t*)c.data_ptr(), size_m, size_n, size_k, groups,
        use_v2_format, stream);
  }

  return c;
}
