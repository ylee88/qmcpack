//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: Youngjun Lee, leey@anl.gov, Argonne National Laboratory
//
// File created by: Youngjun Lee, leey@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#include "AccelBLAS_CUDA.hpp"

#if defined(QMC_BLAS_FP64_EMULATION) && !defined(QMC_CUDA2HIP)

#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace qmcplusplus
{
namespace compute
{
namespace BLAS
{
namespace detail
{
inline int ceildiv(const int value, const int divisor) { return (value + divisor - 1) / divisor; }

std::size_t getFixedPointWorkspaceSizeInBytes(const int m,
                                              const int n,
                                              const int k,
                                              const int batch_count,
                                              const bool is_complex,
                                              const cudaEmulationMantissaControl_t mantissa_control,
                                              const int max_mantissa_bit_count)
{
  constexpr std::size_t FIXED_POINT_CONSTANT_WORKSPACE_BYTES = 128ULL * 1024ULL * 1024ULL;
  constexpr double FIXED_POINT_WORKSPACE_MULTIPLIER          = 1.25;

  const std::size_t mult       = is_complex ? 2 : 1;
  const int num_slices         = ceildiv(max_mantissa_bit_count + 1, 8);
  const int max_splitk         = ceildiv(k, 8192);
  const int padded_m           = ceildiv(m, 1024) * 1024;
  const int padded_n           = ceildiv(n, 1024) * 1024;
  const int padded_k           = ceildiv(k, 128) * 128;
  const int num_blocks_k       = ceildiv(k, 64);
  const std::size_t sm32_limit = static_cast<std::size_t>(1ULL << 32);

  std::size_t gemm_workspace = sizeof(int8_t) *
      (static_cast<std::size_t>(padded_m) * padded_k + static_cast<std::size_t>(padded_n) * padded_k) * mult *
      num_slices;
  gemm_workspace += sizeof(int32_t) * (static_cast<std::size_t>(padded_m) + padded_n) * mult;

  std::size_t acc_workspace_ver1 = 0;
  if (is_complex)
    acc_workspace_ver1 += sizeof(double) * static_cast<std::size_t>(m) * n * mult * mult;

  const std::size_t acc_workspace_ver2_base =
      sizeof(int32_t) * static_cast<std::size_t>(padded_m) * padded_n * mult * mult * num_slices;
  const std::size_t acc_workspace_ver2 = std::min(acc_workspace_ver2_base, sm32_limit) * max_splitk;
  gemm_workspace += std::max(acc_workspace_ver1, acc_workspace_ver2);

  std::size_t adp_workspace = 0;
  if (mantissa_control == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC)
    adp_workspace = sizeof(int32_t) *
        (static_cast<std::size_t>(m) * num_blocks_k + static_cast<std::size_t>(n) * num_blocks_k +
         static_cast<std::size_t>(m) * n) *
        mult;

  const std::size_t emulation_workspace = std::max(gemm_workspace, adp_workspace) * batch_count;
  return static_cast<std::size_t>(std::ceil(emulation_workspace * FIXED_POINT_WORKSPACE_MULTIPLIER)) +
      FIXED_POINT_CONSTANT_WORKSPACE_BYTES;
}

void gemmFp64EmulatedFixedPoint(BLASHandle<PlatformKind::CUDA>& handle,
                                const char transa,
                                const char transb,
                                int m,
                                int n,
                                int k,
                                const double& alpha,
                                const double* A,
                                int lda,
                                const double* B,
                                int ldb,
                                const double& beta,
                                double* C,
                                int ldc,
                                const BLASPolicy<PlatformKind::CUDA>& policy)
{
  const std::size_t required_workspace_bytes =
      getFixedPointWorkspaceSizeInBytes(m, n, k, 1, false, CUDA_EMULATION_MANTISSA_CONTROL_FIXED,
                                        policy.max_mantissa_bits);
  const std::size_t requested_workspace_bytes = std::max(policy.min_workspace_bytes, required_workspace_bytes);
  auto& lt_emulation_context                  = handle.ensureLtEmulationContext(requested_workspace_bytes);

  cublasLtMatmulDesc_t operation_desc    = nullptr;
  cublasLtMatrixLayout_t a_desc          = nullptr;
  cublasLtMatrixLayout_t b_desc          = nullptr;
  cublasLtMatrixLayout_t c_desc          = nullptr;
  cublasLtEmulationDesc_t emulation_desc = nullptr;

  const cublasOperation_t transa_op = cuBLAS::convertOperation(transa);
  const cublasOperation_t transb_op = cuBLAS::convertOperation(transb);

  const int rows_a = (transa_op == CUBLAS_OP_N) ? m : k;
  const int cols_a = (transa_op == CUBLAS_OP_N) ? k : m;
  const int rows_b = (transb_op == CUBLAS_OP_N) ? k : n;
  const int cols_b = (transb_op == CUBLAS_OP_N) ? n : k;

  cublasErrorCheck(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F),
                   "cublasLtMatmulDescCreate failed!");
  cublasErrorCheck(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa_op,
                                                  sizeof(transa_op)),
                   "cublasLtMatmulDescSetAttribute TRANSA failed!");
  cublasErrorCheck(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb_op,
                                                  sizeof(transb_op)),
                   "cublasLtMatmulDescSetAttribute TRANSB failed!");

  cublasErrorCheck(cublasLtEmulationDescCreate(&emulation_desc), "cublasLtEmulationDescCreate failed!");

  const cublasEmulationStrategy_t strategy              = CUBLAS_EMULATION_STRATEGY_EAGER;
  const cudaEmulationMantissaControl_t mantissa_control = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;

  cublasErrorCheck(cublasLtEmulationDescSetAttribute(emulation_desc, CUBLASLT_EMULATION_DESC_STRATEGY, &strategy,
                                                     sizeof(strategy)),
                   "cublasLtEmulationDescSetAttribute STRATEGY failed!");
  cublasErrorCheck(cublasLtEmulationDescSetAttribute(emulation_desc,
                                                     CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL,
                                                     &mantissa_control, sizeof(mantissa_control)),
                   "cublasLtEmulationDescSetAttribute MANTISSA_CONTROL failed!");
  cublasErrorCheck(cublasLtEmulationDescSetAttribute(emulation_desc,
                                                     CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
                                                     &policy.max_mantissa_bits, sizeof(policy.max_mantissa_bits)),
                   "cublasLtEmulationDescSetAttribute MAX_MANTISSA_BIT_COUNT failed!");
  cublasErrorCheck(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
                                                  &emulation_desc, sizeof(emulation_desc)),
                   "cublasLtMatmulDescSetAttribute EMULATION_DESCRIPTOR failed!");

  cublasErrorCheck(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_64F, rows_a, cols_a, lda),
                   "cublasLtMatrixLayoutCreate A failed!");
  cublasErrorCheck(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_64F, rows_b, cols_b, ldb),
                   "cublasLtMatrixLayoutCreate B failed!");
  cublasErrorCheck(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_64F, m, n, ldc), "cublasLtMatrixLayoutCreate C failed!");

  cublasErrorCheck(cublasLtMatmul(lt_emulation_context.getLtHandle(), operation_desc, &alpha, A, a_desc, B, b_desc,
                                  &beta, C, c_desc, C, c_desc, nullptr, lt_emulation_context.getWorkspacePtr(),
                                  lt_emulation_context.getWorkspaceSize(), handle.h_stream),
                   "cublasLtMatmul failed!");

  cublasErrorCheck(cublasLtMatrixLayoutDestroy(c_desc), "cublasLtMatrixLayoutDestroy C failed!");
  cublasErrorCheck(cublasLtMatrixLayoutDestroy(b_desc), "cublasLtMatrixLayoutDestroy B failed!");
  cublasErrorCheck(cublasLtMatrixLayoutDestroy(a_desc), "cublasLtMatrixLayoutDestroy A failed!");
  cublasErrorCheck(cublasLtEmulationDescDestroy(emulation_desc), "cublasLtEmulationDescDestroy failed!");
  cublasErrorCheck(cublasLtMatmulDescDestroy(operation_desc), "cublasLtMatmulDescDestroy failed!");
}

void gemmBatchedFp64EmulatedFixedPoint(BLASHandle<PlatformKind::CUDA>& handle,
                                       const char transa,
                                       const char transb,
                                       int m,
                                       int n,
                                       int k,
                                       const double& alpha,
                                       const double* const A[],
                                       int lda,
                                       const double* const B[],
                                       int ldb,
                                       const double& beta,
                                       double* const C[],
                                       int ldc,
                                       int batchCount,
                                       const BLASPolicy<PlatformKind::CUDA>& policy)
{
  const std::size_t required_workspace_bytes =
      getFixedPointWorkspaceSizeInBytes(m, n, k, batchCount, false, CUDA_EMULATION_MANTISSA_CONTROL_FIXED,
                                        policy.max_mantissa_bits);
  const std::size_t requested_workspace_bytes = std::max(policy.min_workspace_bytes, required_workspace_bytes);
  auto& lt_emulation_context                  = handle.ensureLtEmulationContext(requested_workspace_bytes);

  cublasLtMatmulDesc_t operation_desc    = nullptr;
  cublasLtMatrixLayout_t a_desc          = nullptr;
  cublasLtMatrixLayout_t b_desc          = nullptr;
  cublasLtMatrixLayout_t c_desc          = nullptr;
  cublasLtEmulationDesc_t emulation_desc = nullptr;

  const cublasOperation_t transa_op = cuBLAS::convertOperation(transa);
  const cublasOperation_t transb_op = cuBLAS::convertOperation(transb);

  const int rows_a = (transa_op == CUBLAS_OP_N) ? m : k;
  const int cols_a = (transa_op == CUBLAS_OP_N) ? k : m;
  const int rows_b = (transb_op == CUBLAS_OP_N) ? k : n;
  const int cols_b = (transb_op == CUBLAS_OP_N) ? n : k;

  cublasErrorCheck(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F),
                   "cublasLtMatmulDescCreate failed!");
  cublasErrorCheck(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa_op,
                                                  sizeof(transa_op)),
                   "cublasLtMatmulDescSetAttribute TRANSA failed!");
  cublasErrorCheck(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb_op,
                                                  sizeof(transb_op)),
                   "cublasLtMatmulDescSetAttribute TRANSB failed!");

  cublasErrorCheck(cublasLtEmulationDescCreate(&emulation_desc), "cublasLtEmulationDescCreate failed!");

  const cublasEmulationStrategy_t strategy              = CUBLAS_EMULATION_STRATEGY_EAGER;
  const cudaEmulationMantissaControl_t mantissa_control = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;

  cublasErrorCheck(cublasLtEmulationDescSetAttribute(emulation_desc, CUBLASLT_EMULATION_DESC_STRATEGY, &strategy,
                                                     sizeof(strategy)),
                   "cublasLtEmulationDescSetAttribute STRATEGY failed!");
  cublasErrorCheck(cublasLtEmulationDescSetAttribute(emulation_desc,
                                                     CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL,
                                                     &mantissa_control, sizeof(mantissa_control)),
                   "cublasLtEmulationDescSetAttribute MANTISSA_CONTROL failed!");
  cublasErrorCheck(cublasLtEmulationDescSetAttribute(emulation_desc,
                                                     CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
                                                     &policy.max_mantissa_bits, sizeof(policy.max_mantissa_bits)),
                   "cublasLtEmulationDescSetAttribute MAX_MANTISSA_BIT_COUNT failed!");
  cublasErrorCheck(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
                                                  &emulation_desc, sizeof(emulation_desc)),
                   "cublasLtMatmulDescSetAttribute EMULATION_DESCRIPTOR failed!");

  cublasErrorCheck(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_64F, rows_a, cols_a, lda),
                   "cublasLtMatrixLayoutCreate A failed!");
  cublasErrorCheck(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_64F, rows_b, cols_b, ldb),
                   "cublasLtMatrixLayoutCreate B failed!");
  cublasErrorCheck(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_64F, m, n, ldc), "cublasLtMatrixLayoutCreate C failed!");

  const cublasLtBatchMode_t batch_mode = CUBLASLT_BATCH_MODE_POINTER_ARRAY;
  cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount,
                                                    sizeof(batchCount)),
                   "cublasLtMatrixLayoutSetAttribute A BATCH_COUNT failed!");
  cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount,
                                                    sizeof(batchCount)),
                   "cublasLtMatrixLayoutSetAttribute B BATCH_COUNT failed!");
  cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount,
                                                    sizeof(batchCount)),
                   "cublasLtMatrixLayoutSetAttribute C BATCH_COUNT failed!");
  cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode,
                                                    sizeof(batch_mode)),
                   "cublasLtMatrixLayoutSetAttribute A BATCH_MODE failed!");
  cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode,
                                                    sizeof(batch_mode)),
                   "cublasLtMatrixLayoutSetAttribute B BATCH_MODE failed!");
  cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode,
                                                    sizeof(batch_mode)),
                   "cublasLtMatrixLayoutSetAttribute C BATCH_MODE failed!");

  auto non_const_C = const_cast<BottomConstRemoved<decltype(C)>::type>(C);

  cublasErrorCheck(cublasLtMatmul(lt_emulation_context.getLtHandle(), operation_desc, &alpha, A, a_desc, B, b_desc,
                                  &beta, C, c_desc, non_const_C, c_desc, nullptr,
                                  lt_emulation_context.getWorkspacePtr(), lt_emulation_context.getWorkspaceSize(),
                                  handle.h_stream),
                   "cublasLtMatmul batched failed!");

  cublasErrorCheck(cublasLtMatrixLayoutDestroy(c_desc), "cublasLtMatrixLayoutDestroy C failed!");
  cublasErrorCheck(cublasLtMatrixLayoutDestroy(b_desc), "cublasLtMatrixLayoutDestroy B failed!");
  cublasErrorCheck(cublasLtMatrixLayoutDestroy(a_desc), "cublasLtMatrixLayoutDestroy A failed!");
  cublasErrorCheck(cublasLtEmulationDescDestroy(emulation_desc), "cublasLtEmulationDescDestroy failed!");
  cublasErrorCheck(cublasLtMatmulDescDestroy(operation_desc), "cublasLtMatmulDescDestroy failed!");
}

} // namespace detail
} // namespace BLAS
} // namespace compute
} // namespace qmcplusplus

#endif
