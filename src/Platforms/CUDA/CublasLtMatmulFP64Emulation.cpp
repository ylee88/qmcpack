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

#include <vector>

namespace qmcplusplus
{
namespace compute
{
namespace BLAS
{
namespace detail
{

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
  auto& lt_emulation_context = handle.ensureLtEmulationContext(policy.workspace_size_bytes);

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
  std::vector<const double*> h_A(batchCount);
  std::vector<const double*> h_B(batchCount);
  std::vector<double*> h_C(batchCount);

  cudaErrorCheck(cudaMemcpy(h_A.data(), A, sizeof(double*) * batchCount, cudaMemcpyDefault),
                 "cudaMemcpy A pointers for fp64 emulation failed!");
  cudaErrorCheck(cudaMemcpy(h_B.data(), B, sizeof(double*) * batchCount, cudaMemcpyDefault),
                 "cudaMemcpy B pointers for fp64 emulation failed!");
  cudaErrorCheck(cudaMemcpy(h_C.data(), C, sizeof(double*) * batchCount, cudaMemcpyDefault),
                 "cudaMemcpy C pointers for fp64 emulation failed!");

  // TODO: Implement a proper batched version of the fixed-point emulated GEMM
  //       using cublasLt's strided batched interface
  for (int ib = 0; ib < batchCount; ib++)
    gemmFp64EmulatedFixedPoint(handle, transa, transb, m, n, k, alpha, h_A[ib], lda, h_B[ib], ldb, beta, h_C[ib], ldc,
                               policy);
}

} // namespace detail
} // namespace BLAS
} // namespace compute
} // namespace qmcplusplus

#endif
