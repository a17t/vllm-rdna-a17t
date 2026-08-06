// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Native RDNA2 (gfx1030) GDN decode kernels:
//   1. gdn_conv1d_update_rdna2   (causal_conv1d_update, KERNEL_WIDTH=4, silu)
//   2. gdn_sigmoid_gating_delta_rule_update_rdna2
//
// Design: fp32 accumulation, __half I/O, LDS for matrix tiles,
// no warp shuffles (RDNA2 wave32), no early returns in __syncthreads paths.

#include <cmath>
#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__GDN_RDNA2__
#endif

namespace vllm {
namespace gdn_rdna2 {

// ---------------------------------------------------------------------------
// 1. causal_conv1d_update (fp16, KERNEL_WIDTH=4, silu)
//
// Algorithm (matching Triton _causal_conv1d_update_kernel exactly):
//
//   For each batch item (idx_seq):
//     1. Read col0, col1, col2 from OLD conv_state (at conv_state_token_offset=0)
//     2. Read source for new state:
//        - new_cs[t] = old_cs[t + seqlen]  if t + seqlen < state_len
//        - new_cs[t] = x[t - (state_len - seqlen)] otherwise
//        All reads from OLD state into registers first.
//     3. Write new state back to conv_state (same cache_idx for non-APC)
//     4. For each token:
//        - acc = bias + col0*w0 + col1*w1 + col2*w2 + x_new*w3
//        - silu(acc) -> output
//        - col0=col1, col1=col2, col2=x_new
//
// x layout:     [batch, dim, seqlen]
// w layout:     [dim, 4]
// bias:         [dim] (optional)
// conv_state:   [num_cache_lines, dim, state_len]
// conv_state_indices: [batch] int32
// ---------------------------------------------------------------------------

#if defined(__HIP__GDN_RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <bool HAS_BIAS>
__global__ void conv1d_update_kernel_fp16(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ bias,
    __half* __restrict__ conv_state,
    const int32_t* __restrict__ conv_state_indices,
    __half* __restrict__ o,
    int batch, int dim, int seqlen, int state_len, int num_cache_lines,
    int stride_x_seq, int stride_x_dim, int stride_x_token,
    int stride_o_seq, int stride_o_dim, int stride_o_token,
    int stride_cs_seq, int stride_cs_dim, int stride_cs_tok,
    int stride_w_dim, int stride_w_width) {

    int idx_seq = blockIdx.x;
    if (idx_seq >= batch) return;

    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    int cache_idx = 0;
    if (conv_state_indices) {
        cache_idx = conv_state_indices[idx_seq];
    }
    // Skip null block (block 0 is the null block by default in vLLM)
    if (cache_idx == 0) return;

    for (int d = tid; d < dim; d += total_threads) {
        __half* cs_d = conv_state + cache_idx * stride_cs_seq + d * stride_cs_dim;
        const __half* x_d = x + idx_seq * stride_x_seq + d * stride_x_dim;
        __half* o_d = o + idx_seq * stride_o_seq + d * stride_o_dim;
        const __half* w_d = w + d * stride_w_dim;

        // STEP 1: Read old state columns (col0=oldest=cs[0], col2=newest=cs[2])
        float col0 = __half2float(cs_d[0 * stride_cs_tok]);
        float col1 = __half2float(cs_d[1 * stride_cs_tok]);
        float col2 = __half2float(cs_d[2 * stride_cs_tok]);

        // Load weights
        float w0 = __half2float(w_d[0 * stride_w_width]);
        float w1 = __half2float(w_d[1 * stride_w_width]);
        float w2 = __half2float(w_d[2 * stride_w_width]);
        float w3 = __half2float(w_d[3 * stride_w_width]);

        float acc_base = HAS_BIAS ? __half2float(bias[d]) : 0.0f;

        // STEP 2: Read new state values into registers FIRST (avoid in-place corruption)
        // Triton: new_cs[t] = cs[t+seqlen] if t+seqlen < state_len, else x[t-(state_len-seqlen)]
        float new_state[16]; // max state_len is small (typically 3-255)
        for (int t = 0; t < state_len; t++) {
            int src = t + seqlen;
            if (src < state_len) {
                new_state[t] = __half2float(cs_d[src * stride_cs_tok]);
            } else {
                int x_tok = t - (state_len - seqlen);
                new_state[t] = (x_tok >= 0 && x_tok < seqlen)
                    ? __half2float(x_d[x_tok * stride_x_token]) : 0.0f;
            }
        }
        // Now write new state back (safe: all reads from old state are done)
        for (int t = 0; t < state_len; t++) {
            cs_d[t * stride_cs_tok] = __float2half_rn(new_state[t]);
        }

        // STEP 3-5: Compute conv for each token (KERNEL_WIDTH=4)
        for (int token = 0; token < seqlen; token++) {
            float x_new = __half2float(x_d[token * stride_x_token]);
            float acc = acc_base + col0 * w0 + col1 * w1 + col2 * w2 + x_new * w3;
            acc = acc / (1.0f + expf(-acc));  // silu
            o_d[token * stride_o_token] = __float2half_rn(acc);

            // Shift: col0<-col1, col1<-col2, col2<-x_new
            float new_col0 = col1;
            float new_col1 = col2;
            col2 = x_new;
            col0 = new_col0;
            col1 = new_col1;
        }
    }
}

#else

template <bool HAS_BIAS>
__global__ void conv1d_update_kernel_fp16(
    const void*, const void*, const void*, const void*,
    const int32_t*, void*, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int) {}

#endif

// ---------------------------------------------------------------------------
// 2. fused_sigmoid_gating_delta_rule_update (fp16)
//
// Grid: (NK=1, NV, N*HV)
// Each block = one (i_k, i_v, i_nh) -> one (i_n, i_hv)
// BK=128, BV=32, 256 threads
//
// LDS: sh_h[32][128] = 16KB, sh_q[256] = 1KB, sh_k[256] = 1KB,
//       sh_v[32] = 128B, sh_v_corr[256] = 1KB -> total ~19.25KB
//
// Each thread handles exactly 1 vv with 16 kk values (tid/8 = vv)
// No warp shuffles — all reductions use LDS shared memory.
// ---------------------------------------------------------------------------

#define GDN_BK 256
#define GDN_BV 16
#define GDN_THREADS 256

#if defined(__HIP__GDN_RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// 256 threads, each handles one (vv, kk) pair:
//   vv = tid / 16, kk = tid % 16
// Grid-stride for h update (each thread does 16 kk values for its vv)
__global__ void sigmoid_gating_delta_rule_kernel_fp16(
    const __half* __restrict__ A_log,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    const __half* __restrict__ dt_bias,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ o,
    const __half* __restrict__ h0,
    __half* __restrict__ ht,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ ssm_state_indices,
    float beta, float threshold, float scale,
    int N, int T, int B, int H, int HV, int K, int V,
    int NK, int NV,
    int stride_init_state_token,
    int stride_final_state_token,
    int stride_indices_seq,
    int stride_indices_tok,
    int use_initial_state,
    int inplace_final_state,
    int use_qk_l2norm,
    int is_varlen,
    int is_continuous_batching) {

    __shared__ float sh_h[GDN_BV][GDN_BK];
    __shared__ float sh_q[GDN_BK];
    __shared__ float sh_k[GDN_BK];
    __shared__ float sh_v[GDN_BV];

    int i_k = blockIdx.x;
    int i_v = blockIdx.y;
    int i_nh = blockIdx.z;
    int i_n = i_nh / HV;
    int i_hv = i_nh % HV;
    int i_h = i_hv / (HV / H);

    int o_k_base = i_k * GDN_BK;
    int o_v_base = i_v * GDN_BV;

    int tid = threadIdx.x;
    int threads_per_vv = GDN_THREADS / GDN_BV;  // 16
    int my_vv = tid / threads_per_vv;          // 0..15
    int my_kk = tid % threads_per_vv;          // 0..15

    int bos, eos, seq_T;
    if (is_varlen) {
        bos = cu_seqlens[i_n];
        eos = cu_seqlens[i_n + 1];
        seq_T = eos - bos;
    } else {
        bos = i_n * T;
        eos = bos + T;
        seq_T = T;
    }
    if (seq_T == 0) {
        for (int idx = tid; idx < GDN_BV * GDN_BK; idx += GDN_THREADS) {
            sh_h[idx / GDN_BK][idx % GDN_BK] = 0.0f;
        }
        __syncthreads();
        return;
    }

    int all = B * T;

    // Initialize h[BV, BK] to zero
    for (int idx = tid; idx < GDN_BV * GDN_BK; idx += GDN_THREADS) {
        sh_h[idx / GDN_BK][idx % GDN_BK] = 0.0f;
    }
    __syncthreads();

    // Load initial state
    if (use_initial_state) {
        if (is_continuous_batching && ssm_state_indices) {
            int state_idx = ssm_state_indices[i_n * stride_indices_seq];
            if (state_idx > 0) {
                int p_h0_base = state_idx * stride_init_state_token;
                for (int idx = tid; idx < GDN_BV * GDN_BK; idx += GDN_THREADS) {
                    int vv = idx / GDN_BK;
                    int kk = idx % GDN_BK;
                    int v_abs = o_v_base + vv;
                    int k_abs = o_k_base + kk;
                    if (v_abs < V && k_abs < K) {
                        int h0_off = p_h0_base + i_hv * V * K + v_abs * K + k_abs;
                        sh_h[vv][kk] = __half2float(h0[h0_off]);
                    }
                }
            }
        } else {
            int p_h0_base = bos * HV * V * K;
            for (int idx = tid; idx < GDN_BV * GDN_BK; idx += GDN_THREADS) {
                int vv = idx / GDN_BK;
                int kk = idx % GDN_BK;
                int v_abs = o_v_base + vv;
                int k_abs = o_k_base + kk;
                if (v_abs < V && k_abs < K) {
                    int h0_off = p_h0_base + i_hv * V * K + v_abs * K + k_abs;
                    sh_h[vv][kk] = __half2float(h0[h0_off]);
                }
            }
        }
        __syncthreads();
    }

    int p_q_off = (bos * H + i_h) * K + o_k_base;
    int p_k_off = (bos * H + i_h) * K + o_k_base;
    int p_v_off = (bos * HV + i_hv) * V + o_v_base;
    int p_a_off = bos * HV + i_hv;
    int p_b_off = bos * HV + i_hv;
    int p_o_off = ((i_k * all + bos) * HV + i_hv) * V + o_v_base;

    float A_log_f = __half2float(A_log[i_hv]);
    float dt_bias_f = __half2float(dt_bias[i_hv]);

    for (int t = 0; t < seq_T; t++) {
        // Load q[BK], k[BK] into LDS
        for (int idx = tid; idx < GDN_BK; idx += GDN_THREADS) {
            int k_abs = o_k_base + idx;
            sh_q[idx] = (k_abs < K) ? __half2float(q[p_q_off + idx]) : 0.0f;
            sh_k[idx] = (k_abs < K) ? __half2float(k[p_k_off + idx]) : 0.0f;
        }
        // Load v[BV] into LDS
        for (int idx = tid; idx < GDN_BV; idx += GDN_THREADS) {
            int v_abs = o_v_base + idx;
            sh_v[idx] = (v_abs < V) ? __half2float(v[p_v_off + idx]) : 0.0f;
        }
        __syncthreads();

        float b_f = __half2float(b[p_b_off]);
        float x_f = __half2float(a[p_a_off]) + dt_bias_f;
        float beta_x = beta * x_f;
        float softplus_x = (beta_x <= threshold) ?
            (1.0f / beta) * logf(1.0f + expf(beta_x)) : x_f;
        float b_g = -expf(A_log_f) * softplus_x;
        float exp_b_g = expf(b_g);
        float b_beta = 1.0f / (1.0f + expf(-b_f));

        // L2 normalize q and k
        if (use_qk_l2norm) {
            for (int idx = tid; idx < GDN_BK; idx += GDN_THREADS) {
                sh_q[idx] *= sh_q[idx];
                sh_k[idx] *= sh_k[idx];
            }
            __syncthreads();
            for (int stride = GDN_BK / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    sh_q[tid] += sh_q[tid + stride];
                    sh_k[tid] += sh_k[tid + stride];
                }
                __syncthreads();
            }
            float q_norm = sqrtf(sh_q[0] + 1e-6f);
            float k_norm = sqrtf(sh_k[0] + 1e-6f);
            float q_s = scale / q_norm;
            float k_s = 1.0f / k_norm;
            q_s = __shfl_sync(0xFFFFFFFFull, q_s, 0);
            k_s = __shfl_sync(0xFFFFFFFFull, k_s, 0);
            for (int idx = tid; idx < GDN_BK; idx += GDN_THREADS) {
                int k_abs = o_k_base + idx;
                sh_q[idx] = (k_abs < K) ? __half2float(q[p_q_off + idx]) : 0.0f;
                sh_k[idx] = (k_abs < K) ? __half2float(k[p_k_off + idx]) : 0.0f;
            }
            __syncthreads();
            for (int idx = tid; idx < GDN_BK; idx += GDN_THREADS) {
                sh_q[idx] *= q_s;
                sh_k[idx] *= k_s;
            }
        } else {
            for (int idx = tid; idx < GDN_BK; idx += GDN_THREADS) {
                sh_q[idx] *= scale;
            }
        }
        __syncthreads();

        int v_abs = o_v_base + my_vv;
        bool valid = (my_vv < GDN_BV && v_abs < V);

        // Each thread grid-strides over all BK values for its vv
        // h *= exp(b_g)
        if (valid) {
            for (int kk = my_kk; kk < GDN_BK; kk += threads_per_vv) {
                int k_abs = o_k_base + kk;
                if (k_abs < K) {
                    sh_h[my_vv][kk] *= exp_b_g;
                }
            }
        }
        __syncthreads();

        // v_corr = sum(h * k, K) per vv — grid-stride partial, then reduce in vv group
        {
            float vcorr_local = 0.0f;
            if (valid) {
                for (int kk = my_kk; kk < GDN_BK; kk += threads_per_vv) {
                    int k_abs = o_k_base + kk;
                    if (k_abs < K) {
                        vcorr_local += sh_h[my_vv][kk] * sh_k[kk];
                    }
                }
            }
            sh_h[my_vv][16 + my_kk] = vcorr_local;
            __syncthreads();
            if (my_kk == 0 && valid) {
                float vcorr_sum = 0.0f;
                for (int e = 0; e < threads_per_vv; e++) {
                    vcorr_sum += sh_h[my_vv][16 + e];
                }
                sh_v[my_vv] -= vcorr_sum;
                sh_v[my_vv] *= b_beta;
            }
            __syncthreads();
        }

        // h += outer(v, k)
        if (valid) {
            for (int kk = my_kk; kk < GDN_BK; kk += threads_per_vv) {
                int k_abs = o_k_base + kk;
                if (k_abs < K) {
                    sh_h[my_vv][kk] += sh_v[my_vv] * sh_k[kk];
                }
            }
        }
        __syncthreads();

        // o = sum(h * q, K) per vv
        {
            float o_local = 0.0f;
            if (valid) {
                for (int kk = my_kk; kk < GDN_BK; kk += threads_per_vv) {
                    int k_abs = o_k_base + kk;
                    if (k_abs < K) {
                        o_local += sh_h[my_vv][kk] * sh_q[kk];
                    }
                }
            }
            sh_h[my_vv][16 + my_kk] = o_local;
            __syncthreads();
            if (my_kk == 0 && valid) {
                float o_sum = 0.0f;
                for (int e = 0; e < threads_per_vv; e++) {
                    o_sum += sh_h[my_vv][16 + e];
                }
                o[p_o_off + my_vv] = __float2half_rn(o_sum);
            }
        }

        // Store final state
        if (inplace_final_state && valid) {
            int ht_idx;
            if (is_continuous_batching && ssm_state_indices) {
                int fs_idx = ssm_state_indices[i_n * stride_indices_seq + t];
                if (fs_idx > 0) {
                    ht_idx = fs_idx * stride_final_state_token;
                    for (int kk = my_kk; kk < GDN_BK; kk += threads_per_vv) {
                        int k_abs = o_k_base + kk;
                        if (k_abs < K) {
                            int ht_off = ht_idx + i_hv * V * K + v_abs * K + k_abs;
                            ht[ht_off] = __float2half_rn(sh_h[my_vv][kk]);
                        }
                    }
                }
            } else {
                ht_idx = (bos + t) * stride_final_state_token;
                for (int kk = my_kk; kk < GDN_BK; kk += threads_per_vv) {
                    int k_abs = o_k_base + kk;
                    if (k_abs < K) {
                        int ht_off = ht_idx + i_hv * V * K + v_abs * K + k_abs;
                        ht[ht_off] = __float2half_rn(sh_h[my_vv][kk]);
                    }
                }
            }
        }

        p_q_off += H * K;
        p_k_off += H * K;
        p_o_off += HV * V;
        p_v_off += HV * V;
        p_a_off += HV;
        p_b_off += HV;
    }
}

#else

__global__ void sigmoid_gating_delta_rule_kernel_fp16(
    const void*, const void*, const void*, const void*,
    const void*, const void*, const void*, void*,
    const void*, void*, const int32_t*, const int32_t*,
    float, float, float, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int, int) {}

#endif

} // namespace gdn_rdna2
} // namespace vllm

// ---------------------------------------------------------------------------
// Public C++ entry points
// ---------------------------------------------------------------------------

torch::Tensor gdn_conv1d_update_rdna2(
    torch::Tensor x,
    torch::Tensor weight,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor conv_state,
    const std::optional<torch::Tensor>& conv_state_indices) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(conv_state.is_cuda(), "conv_state must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kHalf, "weight must be fp16");
    TORCH_CHECK(conv_state.scalar_type() == torch::kHalf, "conv_state must be fp16");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto stream = at::cuda::getCurrentCUDAStream();

    int batch = (int)x.size(0);
    int dim = (int)x.size(1);
    int seqlen = (int)x.size(2);
    int width = (int)weight.size(1);
    int state_len = (int)conv_state.size(2);

    TORCH_CHECK(dim == weight.size(0), "dim mismatch");
    TORCH_CHECK(dim == conv_state.size(1), "conv_state dim mismatch");
    TORCH_CHECK(width == 4, "Only KERNEL_WIDTH=4 supported");

    bool has_bias = bias.has_value() && bias->numel() > 0;

    at::Tensor out = x.clone();

    int stride_x_seq = (int)x.stride(0);
    int stride_x_dim = (int)x.stride(1);
    int stride_x_token = (int)x.stride(2);
    int stride_o_seq = (int)out.stride(0);
    int stride_o_dim = (int)out.stride(1);
    int stride_o_token = (int)out.stride(2);
    int stride_cs_seq = (int)conv_state.stride(0);
    int stride_cs_dim = (int)conv_state.stride(1);
    int stride_cs_tok = (int)conv_state.stride(2);
    int stride_w_dim = (int)weight.stride(0);
    int stride_w_width = (int)weight.stride(1);

    int threads = 256;
    dim3 grid(batch);

    if (has_bias) {
        vllm::gdn_rdna2::conv1d_update_kernel_fp16<true><<<grid, threads, 0, stream>>>(
            (const __half*)x.data_ptr(),
            (const __half*)weight.data_ptr(),
            (const __half*)bias->data_ptr(),
            (__half*)conv_state.data_ptr(),
            conv_state_indices.has_value() ? (const int32_t*)conv_state_indices->data_ptr() : nullptr,
            (__half*)out.data_ptr(),
            batch, dim, seqlen, state_len, (int)conv_state.size(0),
            stride_x_seq, stride_x_dim, stride_x_token,
            stride_o_seq, stride_o_dim, stride_o_token,
            stride_cs_seq, stride_cs_dim, stride_cs_tok,
            stride_w_dim, stride_w_width);
    } else {
        vllm::gdn_rdna2::conv1d_update_kernel_fp16<false><<<grid, threads, 0, stream>>>(
            (const __half*)x.data_ptr(),
            (const __half*)weight.data_ptr(),
            nullptr,
            (__half*)conv_state.data_ptr(),
            conv_state_indices.has_value() ? (const int32_t*)conv_state_indices->data_ptr() : nullptr,
            (__half*)out.data_ptr(),
            batch, dim, seqlen, state_len, (int)conv_state.size(0),
            stride_x_seq, stride_x_dim, stride_x_token,
            stride_o_seq, stride_o_dim, stride_o_token,
            stride_cs_seq, stride_cs_dim, stride_cs_tok,
            stride_w_dim, stride_w_width);
    }

    return out;
}

// ---------------------------------------------------------------------------
// Ragged conv1d update for spec-decode: variable-length sequences in one launch.
// Each block processes one sequence; its length comes from query_start_loc.
// No host sync, no Python loop -> CUDA-graph compatible.
// ---------------------------------------------------------------------------
template <bool HAS_BIAS>
__global__ void conv1d_update_ragged_kernel_fp16(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ bias,
    __half* __restrict__ conv_state,
    const int32_t* __restrict__ conv_state_indices,
    const int32_t* __restrict__ query_start_loc,
    __half* __restrict__ o,
    int batch, int dim, int state_len, int num_cache_lines,
    int stride_x_token, int stride_x_dim,
    int stride_o_token, int stride_o_dim,
    int stride_cs_seq, int stride_cs_dim, int stride_cs_tok,
    int stride_w_dim, int stride_w_width) {

    int idx_seq = blockIdx.x;
    if (idx_seq >= batch) return;

    int cache_idx = 0;
    if (conv_state_indices) cache_idx = conv_state_indices[idx_seq];
    if (cache_idx == 0) return;  // null block

    int seq_start = query_start_loc[idx_seq];
    int seqlen = query_start_loc[idx_seq + 1] - seq_start;

    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    for (int d = tid; d < dim; d += total_threads) {
        __half* cs_d =
            conv_state + cache_idx * stride_cs_seq + d * stride_cs_dim;
        const __half* x_d =
            x + seq_start * stride_x_token + d * stride_x_dim;
        __half* o_d =
            o + seq_start * stride_o_token + d * stride_o_dim;
        const __half* w_d = w + d * stride_w_dim;

        float col0 = __half2float(cs_d[0 * stride_cs_tok]);
        float col1 = __half2float(cs_d[1 * stride_cs_tok]);
        float col2 = __half2float(cs_d[2 * stride_cs_tok]);

        float w0 = __half2float(w_d[0 * stride_w_width]);
        float w1 = __half2float(w_d[1 * stride_w_width]);
        float w2 = __half2float(w_d[2 * stride_w_width]);
        float w3 = __half2float(w_d[3 * stride_w_width]);

        float acc_base = HAS_BIAS ? __half2float(bias[d]) : 0.0f;

        float new_state[16];
        for (int t = 0; t < state_len; t++) {
            int src = t + seqlen;
            if (src < state_len) {
                new_state[t] = __half2float(cs_d[src * stride_cs_tok]);
            } else {
                int x_tok = t - (state_len - seqlen);
                new_state[t] =
                    (x_tok >= 0 && x_tok < seqlen)
                        ? __half2float(x_d[x_tok * stride_x_token])
                        : 0.0f;
            }
        }
        for (int t = 0; t < state_len; t++)
            cs_d[t * stride_cs_tok] = __float2half_rn(new_state[t]);

        for (int token = 0; token < seqlen; token++) {
            float x_new = __half2float(x_d[token * stride_x_token]);
            float acc =
                acc_base + col0 * w0 + col1 * w1 + col2 * w2 + x_new * w3;
            acc = acc / (1.0f + expf(-acc));
            o_d[token * stride_o_token] = __float2half_rn(acc);
            float new_col0 = col1;
            float new_col1 = col2;
            col2 = x_new;
            col0 = new_col0;
            col1 = new_col1;
        }
    }
}

torch::Tensor gdn_conv1d_update_spec_rdna2(
    torch::Tensor x,
    torch::Tensor weight,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor conv_state,
    torch::Tensor conv_state_indices,
    torch::Tensor query_start_loc) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [total_tokens, dim]");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(conv_state.is_cuda(), "conv_state must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto stream = at::cuda::getCurrentCUDAStream();

    int batch = (int)query_start_loc.size(0) - 1;
    int dim = (int)x.size(1);
    int width = (int)weight.size(1);
    int state_len = (int)conv_state.size(2);

    TORCH_CHECK(dim == weight.size(0), "dim mismatch");
    TORCH_CHECK(width == 4, "Only KERNEL_WIDTH=4 supported");

    bool has_bias = bias.has_value() && bias->numel() > 0;
    at::Tensor out = x.clone();

    int threads = 256;
    dim3 grid(batch);

    if (has_bias) {
        vllm::gdn_rdna2::conv1d_update_ragged_kernel_fp16<true>
            <<<grid, threads, 0, stream>>>(
                (const __half*)x.data_ptr(),
                (const __half*)weight.data_ptr(),
                (const __half*)bias->data_ptr(),
                (__half*)conv_state.data_ptr(),
                (const int32_t*)conv_state_indices.data_ptr(),
                (const int32_t*)query_start_loc.data_ptr(),
                (__half*)out.data_ptr(),
                batch, dim, state_len, (int)conv_state.size(0),
                (int)x.stride(0), (int)x.stride(1),
                (int)out.stride(0), (int)out.stride(1),
                (int)conv_state.stride(0), (int)conv_state.stride(1),
                (int)conv_state.stride(2),
                (int)weight.stride(0), (int)weight.stride(1));
    } else {
        vllm::gdn_rdna2::conv1d_update_ragged_kernel_fp16<false>
            <<<grid, threads, 0, stream>>>(
                (const __half*)x.data_ptr(),
                (const __half*)weight.data_ptr(),
                nullptr,
                (__half*)conv_state.data_ptr(),
                (const int32_t*)conv_state_indices.data_ptr(),
                (const int32_t*)query_start_loc.data_ptr(),
                (__half*)out.data_ptr(),
                batch, dim, state_len, (int)conv_state.size(0),
                (int)x.stride(0), (int)x.stride(1),
                (int)out.stride(0), (int)out.stride(1),
                (int)conv_state.stride(0), (int)conv_state.stride(1),
                (int)conv_state.stride(2),
                (int)weight.stride(0), (int)weight.stride(1));
    }

    return out;
}

void gdn_sigmoid_gating_delta_rule_update_rdna2(
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor dt_bias,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor initial_state,
    const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& ssm_state_indices,
    double beta_d,
    double threshold_d,
    double scale_d,
    bool inplace_final_state,
    bool use_qk_l2norm,
    torch::Tensor o,
    torch::Tensor final_state) {

    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(k.scalar_type() == torch::kHalf, "k must be fp16");
    TORCH_CHECK(v.scalar_type() == torch::kHalf, "v must be fp16");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    int B = (int)q.size(0);
    int T = (int)q.size(1);
    int H = (int)q.size(2);
    int K = (int)q.size(3);
    int HV = (int)v.size(2);
    int V = (int)v.size(3);

    int N = cu_seqlens.has_value() ? (int)cu_seqlens->size(0) - 1 : B;
    int NK = 1;
    int NV = 1;
    TORCH_CHECK(K <= GDN_BK, "K > GDN_BK not supported");
    TORCH_CHECK(V <= GDN_BV, "V > GDN_BV not supported");

    int stride_init_state_token = initial_state.stride(0);
    int stride_final_state_token = final_state.stride(0);

    int stride_indices_seq = 1, stride_indices_tok = 1;
    if (ssm_state_indices.has_value()) {
        stride_indices_seq = ssm_state_indices->stride(0);
        if (ssm_state_indices->dim() > 1) {
            stride_indices_tok = ssm_state_indices->stride(1);
        }
    }

    int use_initial_state = initial_state.defined() && initial_state.numel() > 0;
    int is_varlen = cu_seqlens.has_value() && cu_seqlens->numel() > 0;
    int is_continuous_batching = ssm_state_indices.has_value() && ssm_state_indices->numel() > 0;

    dim3 grid(NK, NV, N * HV);
    dim3 block(GDN_THREADS);

    vllm::gdn_rdna2::sigmoid_gating_delta_rule_kernel_fp16<<<grid, block, 0, stream>>>(
        (const __half*)A_log.data_ptr(),
        (const __half*)a.data_ptr(),
        (const __half*)b.data_ptr(),
        (const __half*)dt_bias.data_ptr(),
        (const __half*)q.data_ptr(),
        (const __half*)k.data_ptr(),
        (const __half*)v.data_ptr(),
        (__half*)o.data_ptr(),
        use_initial_state ? (const __half*)initial_state.data_ptr() : nullptr,
        (__half*)final_state.data_ptr(),
        is_varlen ? (const int32_t*)(cu_seqlens->data_ptr()) : nullptr,
        is_continuous_batching ? (const int32_t*)(ssm_state_indices->data_ptr()) : nullptr,
        (float)beta_d, (float)threshold_d, (float)scale_d,
        N, T, B, H, HV, K, V,
        NK, NV,
        stride_init_state_token,
        stride_final_state_token,
        stride_indices_seq,
        stride_indices_tok,
        use_initial_state,
        inplace_final_state ? 1 : 0,
        use_qk_l2norm ? 1 : 0,
        is_varlen,
        is_continuous_batching);
}
