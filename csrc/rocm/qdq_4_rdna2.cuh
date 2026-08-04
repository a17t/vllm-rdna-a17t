// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 dequant primitives for RDNA2 (gfx1030). The fp16 path uses the
// classic exllamav2 bit-trick (same as RDNA3) but the dot product must
// widen to fp32 because RDNA2 lacks v_pk_fma_f16. The fp32 widening is
// identical to the RDNA3 bf16 path.

#ifndef _qdq_4_rdna2_cuh
#define _qdq_4_rdna2_cuh

#include <cstdint>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace gptq_rdna2 {

using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;

// Bit-shuffle for an int32 holding 8 sequential 4-bit weights q[0..7]:
//   in:  q[7] q[6] q[5] q[4] q[3] q[2] q[1] q[0]   (LSB first)
//   out: q[7] q[5] q[3] q[1] q[6] q[4] q[2] q[0]   (even/odd interleaved)
__forceinline__ __device__ void shuffle_4bit_8(uint32_t* q) {
  uint32_t qa = q[0];
  uint32_t qb = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t qa0 = qa & 0x0F;
    uint32_t qa1 = (qa & 0xF0) >> 4;
    qa >>= 8;
    qb |= (qa1 << (i * 4 + 16));
    qb |= (qa0 << (i * 4));
  }
  q[0] = qb;
}

// ---------------------------------------------------------------------------
// fp16 path — RDNA2 has no v_pk_fma_f16, must widen to fp32
// ---------------------------------------------------------------------------

__forceinline__ __device__ void prep_zero_scale_fp16(uint32_t zero, half scale,
                                                     half2 (&z1z16)[2],
                                                     half2 (&y1y16)[2]) {
  union {
    uint16_t u;
    half h;
  } z1u;
  z1u.u = (uint16_t)(0xE400 | zero);
  half z1 = z1u.h;
  half z16 = __hsub(__int2half_rn(-64), __int2half_rn((int)zero));

  half2 scale2 = __half2half2(scale);
  z1z16[0] = __hmul2(scale2, __half2half2(z1));
  z1z16[1] = __hmul2(scale2, __half2half2(z16));

  half y1 = __float2half_rn(1.0f);
  half y16 = __float2half_rn(1.0f / 16.0f);
  y1y16[0] = __hmul2(scale2, __half2half2(y1));
  y1y16[1] = __hmul2(scale2, __half2half2(y16));
}

__forceinline__ __device__ void dequant_4bit_8_fp16(uint32_t qa, half2 (&dq)[4],
                                                     half2 (&z1z16)[2],
                                                     half2 (&y1y16)[2]) {
  const uint32_t c0 = 0x64006400;

  union {
    uint32_t u;
    half2 h2;
  } q0, q1, q2, q3;
  q0.u = (qa & 0x000F000F) | c0;
  q1.u = (qa & 0x00F000F0) | c0;
  uint32_t qa_hi = qa >> 8;
  q2.u = (qa_hi & 0x000F000F) | c0;
  q3.u = (qa_hi & 0x00F000F0) | c0;

  dq[0] = __hfma2(y1y16[0], q0.h2, z1z16[0]);
  dq[1] = __hfma2(y1y16[1], q1.h2, z1z16[1]);
  dq[2] = __hfma2(y1y16[0], q2.h2, z1z16[0]);
  dq[3] = __hfma2(y1y16[1], q3.h2, z1z16[1]);
}

// ---------------------------------------------------------------------------
// bf16 path — widen to fp32 (RDNA2 has no v_pk_fma_bf16)
// ---------------------------------------------------------------------------

__forceinline__ __device__ void prep_zero_scale_bf16(uint32_t zero, bf16_t scale,
                                                      bf162_t& z_prep,
                                                      bf162_t& y_prep) {
  float scale_f = __bfloat162float(scale);
  float zf = -(128.0f + (float)zero) * scale_f;
  bf16_t zb = __float2bfloat16(zf);
  z_prep = __bfloat162bfloat162(zb);
  y_prep = __bfloat162bfloat162(scale);
}

__forceinline__ __device__ void dequant_4bit_8_bf16(uint32_t qa, bf162_t (&dq)[4],
                                                     bf162_t z_prep,
                                                     bf162_t y_prep) {
  const uint32_t c0 = 0x43004300;

  union {
    uint32_t u;
    bf162_t b2;
  } q0, q1, q2, q3;
  q0.u = ((qa >> 0) & 0x000F000F) | c0;
  q1.u = ((qa >> 4) & 0x000F000F) | c0;
  q2.u = ((qa >> 8) & 0x000F000F) | c0;
  q3.u = ((qa >> 12) & 0x000F000F) | c0;

  dq[0] = __hfma2(q0.b2, y_prep, z_prep);
  dq[1] = __hfma2(q1.b2, y_prep, z_prep);
  dq[2] = __hfma2(q2.b2, y_prep, z_prep);
  dq[3] = __hfma2(q3.b2, y_prep, z_prep);
}

} // namespace gptq_rdna2
} // namespace vllm

#endif // _qdq_4_rdna2_cuh
