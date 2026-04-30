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

#include "Common/AccelBLASHandle.hpp"
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
void gemmFp64EmulatedFixedPoint(BLASHandle<PlatformKind::CUDA>& handle,
                                const char transa,
                                const char transb,
                                int m,
                                int n,
                                int k,
                                const T& alpha,
                                const T* A,
                                int lda,
                                const T* B,
                                int ldb,
                                const T& beta,
                                T* C,
                                int ldc,
                                const BLASPolicy& policy);

template<typename T>
void gemmBatchedFp64EmulatedFixedPoint(BLASHandle<PlatformKind::CUDA>& handle,
                                       const char transa,
                                       const char transb,
                                       int m,
                                       int n,
                                       int k,
                                       const T& alpha,
                                       const T* const A[],
                                       int lda,
                                       const T* const B[],
                                       int ldb,
                                       const T& beta,
                                       const T* const C[],
                                       int ldc,
                                       int batchCount,
                                       const BLASPolicy& policy);
} // namespace detail
} // namespace BLAS
} // namespace compute
} // namespace qmcplusplus

#endif
