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

#ifndef QMCPLUSPLUS_CUBLASLT_MATMUL_FP64EMULATION_HPP
#define QMCPLUSPLUS_CUBLASLT_MATMUL_FP64EMULATION_HPP

#if defined(QMC_BLAS_FP64_EMULATION) && !defined(QMC_CUDA2HIP)

#include "Common/AccelBLASHandle.hpp"
#include "CUDA/cuBLAS.hpp"
#include "CUDA/AccelBLASPolicy_CUDA.hpp"

namespace qmcplusplus
{
namespace compute
{
namespace BLAS
{
namespace detail
{
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
                                const BLASPolicy& policy);

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
                                       const BLASPolicy& policy);

// Explicit template instantiations for FP64 emulation functions
extern template void gemmFP64EmulatedFixedPoint<double>(BLASHandle<PlatformKind::CUDA>& handle,
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

extern template void gemmFP64EmulatedFixedPoint<cuDoubleComplex>(BLASHandle<PlatformKind::CUDA>& handle,
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

extern template void gemmBatchedFP64EmulatedFixedPoint<double>(BLASHandle<PlatformKind::CUDA>& handle,
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

extern template void gemmBatchedFP64EmulatedFixedPoint<cuDoubleComplex>(BLASHandle<PlatformKind::CUDA>& handle,
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

#endif  // defined(QMC_BLAS_FP64_EMULATION) && !defined(QMC_CUDA2HIP)

#endif
