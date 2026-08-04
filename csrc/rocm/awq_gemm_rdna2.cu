// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 AWQ kernel for RDNA2 (gfx1030), fp16 only.
//
// Reads weights in sequential GPTQ-style packing: [K, N//8] int32,
// each int32 holds 8 consecutive N-values at bit offsets [0,4,...,28].
//
// Dequant: w_fp = (w_int4 - zero) * scale  (no +1 zero offset for AWQ)
//
// Geometry: wave32, THREADS_X=256, BLOCK_KN_SIZE=256.
// Each thread handles 4 consecutive N columns, processes K one-at-a-time.
// All arithmetic in fp32 (RDNA2 has no v_pk_fma_f16).

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "qdq_awq_rdna2.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__AWQ_RDNA2__
#endif

namespace vllm {
namespace awq_rdna2 {

#define BLOCK_KN_SIZE 256
#define THREADS_X 256

// Packed atomic-add via CAS-loop (same as GPTQ kernel).
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

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

#if defined(__HIP__AWQ_RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <int M_COUNT>
__global__ void gemm_awq_q4_kernel_rdna2(
    const half* __restrict__ a,
    const uint32_t* __restrict__ b_q_weight,  // [K, N//8]
    const uint32_t* __restrict__ b_qzeros,    // [G, N//8]
    const half* __restrict__ b_scales,        // [G, N]
    half* __restrict__ c,                     // [M, N]
    const int size_m, const int size_n, const int size_k,
    const int groups) {

  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int offset_m = blockIdx.y * M_COUNT;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  constexpr int LDS_PAD = 8;
  __shared__ half block_a[M_COUNT][BLOCK_KN_SIZE + LDS_PAD];

  // Stage A through LDS.
  if (offset_k + t < end_k) {
  #pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
      half av;
      if (offset_m + m < size_m) {
        av = a[(offset_m + m) * size_k + (offset_k + t)];
      } else {
        av = __float2half_rn(0.0f);
      }
      block_a[m][t] = av;
    }
  }
  __syncthreads();
  if (n >= size_n) return;

  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // Precompute per-column nibble extraction parameters.
  int n_packed[4], shift[4];
  #pragma unroll
  for (int col = 0; col < 4; ++col) {
    int nc = n + col;
    n_packed[col] = nc / 8;
    shift[col] = (nc & 0x07) * 4;
  }

  // Per-column fp32 dequant constants: w_fp32 = nibble * y_f + z_f
  float z_f[4], y_f[4];

  auto refresh_group = [&](int g) {
  #pragma unroll
    for (int col = 0; col < 4; ++col) {
      int nc = n + col;
      // Extract zero from packed qzeros row.
      int qcol = nc / 8;
      int s = (nc & 0x07) * 4;
      uint32_t d = b_qzeros[g * (size_n / 8) + qcol] >> s;
      int zero = (int)(d & 0xF);
      // Get scale for this column.
      half scale = b_scales[g * size_n + nc];
      prep_zero_scale_fp32(zero, scale, z_f[col], y_f[col]);
    }
  };

  refresh_group(group);

  float block_c[M_COUNT][4];
  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
  #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  // Process K in batches of 8 for reduced loop overhead.
  int k = offset_k;
  while (k + 7 < end_k) {
    // Check if this batch of 8 starts at or crosses a group boundary.
    if (k >= nextgroup || k + 7 >= nextgroup) {
      // Process K values up to the boundary one-at-a-time with current group.
      while (k < nextgroup && k + 7 < end_k) {
        #pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          float a_f = __half2float(block_a[m][k - offset_k]);
          #pragma unroll
          for (int col = 0; col < 4; ++col) {
            uint32_t qa = b_q_weight[k * (size_n / 8) + n_packed[col]];
            int nibble = (qa >> shift[col]) & 0xF;
            float w = __fmaf_rn((float)nibble, y_f[col], z_f[col]);
            block_c[m][col] = __fmaf_rn(w, a_f, block_c[m][col]);
          }
        }
        k++;
      }
      if (k >= nextgroup) {
        group++;
        nextgroup += groupsize;
        refresh_group(group);
      }
      // Continue to next iteration for remaining K in batches or tail.
      continue;
    }

    // Process 8 K positions at once.
    // Widen using __half22float2: 2 fp16 -> 2 fp32 per call.
    float a_f[8][M_COUNT];
    #pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
      #pragma unroll
      for (int i = 0; i < 8; i += 2) {
        half2 av = __halves2half2(block_a[m][k + i - offset_k],
                                  block_a[m][k + i + 1 - offset_k]);
        float2 f2 = __half22float2(av);
        a_f[i][m] = f2.x;
        a_f[i + 1][m] = f2.y;
      }
    }

    #pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
      #pragma unroll
      for (int col = 0; col < 4; ++col) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          uint32_t qa = b_q_weight[(k + i) * (size_n / 8) + n_packed[col]];
          int nibble = (qa >> shift[col]) & 0xF;
          float w = __fmaf_rn((float)nibble, y_f[col], z_f[col]);
          block_c[m][col] = __fmaf_rn(w, a_f[i][m], block_c[m][col]);
        }
      }
    }

    k += 8;
  }

  // Tail: process remaining K positions one-at-a-time.
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group(group);
    }

    #pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
      float a_f = __half2float(block_a[m][k - offset_k]);
      #pragma unroll
      for (int col = 0; col < 4; ++col) {
        uint32_t qa = b_q_weight[k * (size_n / 8) + n_packed[col]];
        int nibble = (qa >> shift[col]) & 0xF;
        float w = __fmaf_rn((float)nibble, y_f[col], z_f[col]);
        block_c[m][col] = __fmaf_rn(w, a_f, block_c[m][col]);
      }
    }

    k++;
  }

  // Atomic write-back.
  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    if (offset_m + m >= size_m) continue;
    half* out = c + (offset_m + m) * size_n + n;
    half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                __float2half_rn(block_c[m][1]));
    half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                __float2half_rn(block_c[m][3]));
    atomic_add_pk4_f16(out, r01, r23);
  }
}

#else

template <int M_COUNT>
__global__ void gemm_awq_q4_kernel_rdna2(const half*, const uint32_t*,
                                          const uint32_t*, const half*,
                                          half*, const int, const int,
                                          const int, const int) {}

#endif

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <int M_COUNT>
void launch_awq_gemm_q4_for_mcount(const half* a, const uint32_t* b_q_weight,
                                    const uint32_t* b_qzeros,
                                    const half* b_scales, half* c,
                                    int size_m, int size_n, int size_k,
                                    int groups, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  gemm_awq_q4_kernel_rdna2<M_COUNT><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups);
}

void launch_awq_gemm_q4(const half* a, const uint32_t* b_q_weight,
                        const uint32_t* b_qzeros, const half* b_scales,
                        half* c, int size_m, int size_n, int size_k,
                        int groups, cudaStream_t stream) {
  if (size_m == 1) {
    launch_awq_gemm_q4_for_mcount<1>(a, b_q_weight, b_qzeros, b_scales, c,
                                      size_m, size_n, size_k, groups, stream);
  } else if (size_m <= 3) {
    launch_awq_gemm_q4_for_mcount<2>(a, b_q_weight, b_qzeros, b_scales, c,
                                      size_m, size_n, size_k, groups, stream);
  } else if (size_m <= 7) {
    launch_awq_gemm_q4_for_mcount<4>(a, b_q_weight, b_qzeros, b_scales, c,
                                      size_m, size_n, size_k, groups, stream);
  } else {
    launch_awq_gemm_q4_for_mcount<8>(a, b_q_weight, b_qzeros, b_scales, c,
                                      size_m, size_n, size_k, groups, stream);
  }
}

} // namespace awq_rdna2
} // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

torch::Tensor awq_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_qzeros.is_cuda(), "b_qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K, N/8]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "awq_gemm_rdna2 only supports fp16 activations");
  TORCH_CHECK(b_scales.scalar_type() == torch::kHalf,
              "b_scales must be fp16");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(1) * 8;
  int groups = (int)b_qzeros.size(0);

  TORCH_CHECK(b_q_weight.size(0) == size_k,
              "b_q_weight first dim must be K");
  TORCH_CHECK(b_q_weight.size(1) == size_n / 8,
              "b_q_weight second dim must be N/8");
  TORCH_CHECK(b_scales.size(0) == groups,
              "b_scales must have same group count as qzeros");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales last dim must be N");
  TORCH_CHECK(b_qzeros.size(1) == size_n / 8,
              "b_qzeros second dim must be N/8");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8");

  auto opts = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  at::Tensor c = torch::zeros({size_m, size_n}, opts);

  vllm::awq_rdna2::launch_awq_gemm_q4(
      (const half*)a.data_ptr(), (const uint32_t*)b_q_weight.data_ptr(),
      (const uint32_t*)b_qzeros.data_ptr(), (const half*)b_scales.data_ptr(),
      (half*)c.data_ptr(), size_m, size_n, size_k, groups, stream);

  return c;
}
