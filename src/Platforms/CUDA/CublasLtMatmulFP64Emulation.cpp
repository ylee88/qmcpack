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
template<typename T>
struct EmuTypeTraits;

template<>
struct EmuTypeTraits<double>
{
  static constexpr cudaDataType_t data_type = CUDA_R_64F;
  static constexpr bool is_complex          = false;
};

template<>
struct EmuTypeTraits<cuDoubleComplex>
{
  static constexpr cudaDataType_t data_type = CUDA_C_64F;
  static constexpr bool is_complex          = true;
};

struct MatmulDescs
{
  cublasLtMatmulDesc_t operation_desc    = nullptr;
  cublasLtEmulationDesc_t emulation_desc = nullptr;
  cublasLtMatrixLayout_t a_desc          = nullptr;
  cublasLtMatrixLayout_t b_desc          = nullptr;
  cublasLtMatrixLayout_t c_desc          = nullptr;

  ~MatmulDescs()
  {
    if (c_desc)
      cublasLtMatrixLayoutDestroy(c_desc);
    if (b_desc)
      cublasLtMatrixLayoutDestroy(b_desc);
    if (a_desc)
      cublasLtMatrixLayoutDestroy(a_desc);
    if (emulation_desc)
      cublasLtEmulationDescDestroy(emulation_desc);
    if (operation_desc)
      cublasLtMatmulDescDestroy(operation_desc);
  }
  MatmulDescs() = default;
  MatmulDescs(MatmulDescs&& o) noexcept
      : operation_desc(o.operation_desc),
        emulation_desc(o.emulation_desc),
        a_desc(o.a_desc),
        b_desc(o.b_desc),
        c_desc(o.c_desc)
  {
    o.operation_desc = nullptr;
    o.emulation_desc = nullptr;
    o.a_desc         = nullptr;
    o.b_desc         = nullptr;
    o.c_desc         = nullptr;
  }
  MatmulDescs(const MatmulDescs&)            = delete;
  MatmulDescs& operator=(const MatmulDescs&) = delete;
};

template<typename T>
MatmulDescs makeMatmulDescs(const cublasOperation_t transa_op,
                            const cublasOperation_t transb_op,
                            const int rows_a,
                            const int cols_a,
                            const int lda,
                            const int rows_b,
                            const int cols_b,
                            const int ldb,
                            const int m,
                            const int n,
                            const int ldc,
                            const BLASPolicy& policy,
                            const int batchCount,
                            const bool use_batch)
{
  MatmulDescs d;

  cublasErrorCheck(cublasLtMatmulDescCreate(&d.operation_desc, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
                                            EmuTypeTraits<T>::data_type),
                   "cublasLtMatmulDescCreate failed!");
  cublasErrorCheck(cublasLtMatmulDescSetAttribute(d.operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa_op,
                                                  sizeof(transa_op)),
                   "cublasLtMatmulDescSetAttribute TRANSA failed!");
  cublasErrorCheck(cublasLtMatmulDescSetAttribute(d.operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb_op,
                                                  sizeof(transb_op)),
                   "cublasLtMatmulDescSetAttribute TRANSB failed!");

  cublasErrorCheck(cublasLtEmulationDescCreate(&d.emulation_desc), "cublasLtEmulationDescCreate failed!");

  const cublasEmulationStrategy_t strategy              = CUBLAS_EMULATION_STRATEGY_EAGER;
  const cudaEmulationMantissaControl_t mantissa_control = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;

  cublasErrorCheck(cublasLtEmulationDescSetAttribute(d.emulation_desc, CUBLASLT_EMULATION_DESC_STRATEGY, &strategy,
                                                     sizeof(strategy)),
                   "cublasLtEmulationDescSetAttribute STRATEGY failed!");
  cublasErrorCheck(cublasLtEmulationDescSetAttribute(d.emulation_desc,
                                                     CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL,
                                                     &mantissa_control, sizeof(mantissa_control)),
                   "cublasLtEmulationDescSetAttribute MANTISSA_CONTROL failed!");
  cublasErrorCheck(cublasLtEmulationDescSetAttribute(d.emulation_desc,
                                                     CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
                                                     &policy.max_mantissa_bits, sizeof(policy.max_mantissa_bits)),
                   "cublasLtEmulationDescSetAttribute MAX_MANTISSA_BIT_COUNT failed!");
  cublasErrorCheck(cublasLtMatmulDescSetAttribute(d.operation_desc, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
                                                  &d.emulation_desc, sizeof(d.emulation_desc)),
                   "cublasLtMatmulDescSetAttribute EMULATION_DESCRIPTOR failed!");

  cublasErrorCheck(cublasLtMatrixLayoutCreate(&d.a_desc, EmuTypeTraits<T>::data_type, rows_a, cols_a, lda),
                   "cublasLtMatrixLayoutCreate A failed!");
  cublasErrorCheck(cublasLtMatrixLayoutCreate(&d.b_desc, EmuTypeTraits<T>::data_type, rows_b, cols_b, ldb),
                   "cublasLtMatrixLayoutCreate B failed!");
  cublasErrorCheck(cublasLtMatrixLayoutCreate(&d.c_desc, EmuTypeTraits<T>::data_type, m, n, ldc),
                   "cublasLtMatrixLayoutCreate C failed!");

  if (use_batch)
  {
    const cublasLtBatchMode_t batch_mode = CUBLASLT_BATCH_MODE_POINTER_ARRAY;
    cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(d.a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount,
                                                      sizeof(batchCount)),
                     "cublasLtMatrixLayoutSetAttribute A BATCH_COUNT failed!");
    cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(d.b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount,
                                                      sizeof(batchCount)),
                     "cublasLtMatrixLayoutSetAttribute B BATCH_COUNT failed!");
    cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(d.c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount,
                                                      sizeof(batchCount)),
                     "cublasLtMatrixLayoutSetAttribute C BATCH_COUNT failed!");
    cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(d.a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode,
                                                      sizeof(batch_mode)),
                     "cublasLtMatrixLayoutSetAttribute A BATCH_MODE failed!");
    cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(d.b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode,
                                                      sizeof(batch_mode)),
                     "cublasLtMatrixLayoutSetAttribute B BATCH_MODE failed!");
    cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(d.c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode,
                                                      sizeof(batch_mode)),
                     "cublasLtMatrixLayoutSetAttribute C BATCH_MODE failed!");
  }

  return d;
}

template<typename T>
void gemmFP64EmulatedFixedPoint(BLASHandle<PlatformKind::CUDA>& handle,
                                const cublasOperation_t transa_op,
                                const cublasOperation_t transb_op,
                                int m,
                                int n,
                                int k,
                                const T* alpha,
                                const T* A,
                                int lda,
                                const T* B,
                                int ldb,
                                const T* beta,
                                T* C,
                                int ldc,
                                const BLASPolicy& policy)
{
  auto& lt_emulation_context = handle.ensureLtEmulationContext();

  const int rows_a = (transa_op == CUBLAS_OP_N) ? m : k;
  const int cols_a = (transa_op == CUBLAS_OP_N) ? k : m;
  const int rows_b = (transb_op == CUBLAS_OP_N) ? k : n;
  const int cols_b = (transb_op == CUBLAS_OP_N) ? n : k;

  MatmulDescs d = makeMatmulDescs<T>(transa_op, transb_op, rows_a, cols_a, lda, rows_b, cols_b, ldb, m, n, ldc, policy,
                                     /*batchCount=*/1, /*use_batch=*/false);

  const AlgoCacheKey key{m, n, k, lda, ldb, ldc, 1, transa_op, transb_op,
                         EmuTypeTraits<T>::data_type, policy.max_mantissa_bits};
  const auto& algo_entry =
      lt_emulation_context.getOrSelectAlgo(key, d.operation_desc, d.a_desc, d.b_desc, d.c_desc,
                                           policy.min_workspace_bytes);
  lt_emulation_context.ensureWorkspace(algo_entry.workspace_size);

  cublasErrorCheck(cublasLtMatmul(lt_emulation_context.getLtHandle(), d.operation_desc, alpha, A, d.a_desc, B, d.b_desc,
                                  beta, C, d.c_desc, C, d.c_desc, &algo_entry.algo,
                                  lt_emulation_context.getWorkspacePtr(), lt_emulation_context.getWorkspaceSize(),
                                  handle.h_stream),
                   "cublasLtMatmul failed!");
}

template<typename T>
void gemmBatchedFP64EmulatedFixedPoint(BLASHandle<PlatformKind::CUDA>& handle,
                                       const cublasOperation_t transa_op,
                                       const cublasOperation_t transb_op,
                                       int m,
                                       int n,
                                       int k,
                                       const T* alpha,
                                       const T* const A[],
                                       int lda,
                                       const T* const B[],
                                       int ldb,
                                       const T* beta,
                                       const T* const C[],
                                       int ldc,
                                       int batchCount,
                                       const BLASPolicy& policy)
{
  auto& lt_emulation_context = handle.ensureLtEmulationContext();

  const int rows_a = (transa_op == CUBLAS_OP_N) ? m : k;
  const int cols_a = (transa_op == CUBLAS_OP_N) ? k : m;
  const int rows_b = (transb_op == CUBLAS_OP_N) ? k : n;
  const int cols_b = (transb_op == CUBLAS_OP_N) ? n : k;

  MatmulDescs d =
      makeMatmulDescs<T>(transa_op, transb_op, rows_a, cols_a, lda, rows_b, cols_b, ldb, m, n, ldc, policy, batchCount,
                         /*use_batch=*/true);

  const AlgoCacheKey key{m, n, k, lda, ldb, ldc, batchCount, transa_op, transb_op,
                         EmuTypeTraits<T>::data_type, policy.max_mantissa_bits};
  const auto& algo_entry =
      lt_emulation_context.getOrSelectAlgo(key, d.operation_desc, d.a_desc, d.b_desc, d.c_desc,
                                           policy.min_workspace_bytes);
  lt_emulation_context.ensureWorkspace(algo_entry.workspace_size);

  auto non_const_C = const_cast<typename BottomConstRemoved<decltype(C)>::type>(C);

  cublasErrorCheck(cublasLtMatmul(lt_emulation_context.getLtHandle(), d.operation_desc, alpha, A, d.a_desc, B, d.b_desc,
                                  beta, C, d.c_desc, non_const_C, d.c_desc, &algo_entry.algo,
                                  lt_emulation_context.getWorkspacePtr(), lt_emulation_context.getWorkspaceSize(),
                                  handle.h_stream),
                   "cublasLtMatmul batched failed!");
}


// Explicit template instantiations

template void gemmFP64EmulatedFixedPoint<double>(BLASHandle<PlatformKind::CUDA>& handle,
                                                 const cublasOperation_t transa_op,
                                                 const cublasOperation_t transb_op,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 const double* alpha,
                                                 const double* A,
                                                 int lda,
                                                 const double* B,
                                                 int ldb,
                                                 const double* beta,
                                                 double* C,
                                                 int ldc,
                                                 const BLASPolicy& policy);

template void gemmFP64EmulatedFixedPoint<cuDoubleComplex>(BLASHandle<PlatformKind::CUDA>& handle,
                                                          const cublasOperation_t transa_op,
                                                          const cublasOperation_t transb_op,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuDoubleComplex* alpha,
                                                          const cuDoubleComplex* A,
                                                          int lda,
                                                          const cuDoubleComplex* B,
                                                          int ldb,
                                                          const cuDoubleComplex* beta,
                                                          cuDoubleComplex* C,
                                                          int ldc,
                                                          const BLASPolicy& policy);

template void gemmBatchedFP64EmulatedFixedPoint<double>(BLASHandle<PlatformKind::CUDA>& handle,
                                                        const cublasOperation_t transa_op,
                                                        const cublasOperation_t transb_op,
                                                        int m,
                                                        int n,
                                                        int k,
                                                        const double* alpha,
                                                        const double* const A[],
                                                        int lda,
                                                        const double* const B[],
                                                        int ldb,
                                                        const double* beta,
                                                        const double* const C[],
                                                        int ldc,
                                                        int batchCount,
                                                        const BLASPolicy& policy);

template void gemmBatchedFP64EmulatedFixedPoint<cuDoubleComplex>(BLASHandle<PlatformKind::CUDA>& handle,
                                                                 const cublasOperation_t transa_op,
                                                                 const cublasOperation_t transb_op,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const cuDoubleComplex* alpha,
                                                                 const cuDoubleComplex* const A[],
                                                                 int lda,
                                                                 const cuDoubleComplex* const B[],
                                                                 int ldb,
                                                                 const cuDoubleComplex* beta,
                                                                 const cuDoubleComplex* const C[],
                                                                 int ldc,
                                                                 int batchCount,
                                                                 const BLASPolicy& policy);

} // namespace detail
} // namespace BLAS
} // namespace compute
} // namespace qmcplusplus

#endif
