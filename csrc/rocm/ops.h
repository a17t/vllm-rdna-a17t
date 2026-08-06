#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor gptq_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format);

torch::Tensor awq_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                             torch::Tensor b_qzeros, torch::Tensor b_scales);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales,
                                   torch::Tensor b_g_idx, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);

torch::Tensor gdn_conv1d_update_rdna2(
    torch::Tensor x, torch::Tensor weight,
    const std::optional<torch::Tensor>& bias, torch::Tensor conv_state,
    const std::optional<torch::Tensor>& conv_state_indices);

torch::Tensor gdn_conv1d_update_spec_rdna2(
    torch::Tensor x, torch::Tensor weight,
    const std::optional<torch::Tensor>& bias, torch::Tensor conv_state,
    torch::Tensor conv_state_indices, torch::Tensor query_start_loc);

void gdn_sigmoid_gating_delta_rule_update_rdna2(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b,
    torch::Tensor dt_bias, torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor initial_state,
    const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& ssm_state_indices, double beta_d,
    double threshold_d, double scale_d, bool inplace_final_state,
    bool use_qk_l2norm, torch::Tensor o, torch::Tensor final_state);

void paged_attention_custom_launcher_rdna2(
    torch::Tensor out, torch::Tensor exp_sums, torch::Tensor max_logits,
    torch::Tensor tmp_out, torch::Tensor query, torch::Tensor key_cache,
    torch::Tensor value_cache, int num_kv_heads, float scale,
    torch::Tensor block_tables, torch::Tensor seq_lens,
    torch::Tensor query_start_loc, int block_size, int max_seq_len,
    torch::Tensor alibi_slopes, torch::Tensor k_scale, torch::Tensor v_scale,
    torch::Tensor fp8_out_scale);

void moe_awq_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                        torch::Tensor b_q_weight, torch::Tensor b_scales,
                        torch::Tensor b_qzeros, torch::Tensor topk_weights,
                        torch::Tensor sorted_token_ids,
                        torch::Tensor expert_ids,
                        torch::Tensor num_tokens_post_padded, int64_t top_k,
                        int64_t block_size_m, bool mul_topk_weight,
                        int64_t output_topk);
