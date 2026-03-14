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

#ifndef QMCPLUSPLUS_CUBLASLTEMULATIONCONTEXT_HPP
#define QMCPLUSPLUS_CUBLASLTEMULATIONCONTEXT_HPP

#include "CUDA/CUDAruntime.hpp"
#include "CUDA/cuBLAS.hpp"

#include <cstddef>

#if defined(QMC_BLAS_FP64_EMULATION) && !defined(QMC_CUDA2HIP)

#include <cublasLt.h>

namespace qmcplusplus
{
namespace compute
{

class CublasLtEmulationContext
{
public:
  CublasLtEmulationContext() { cublasErrorCheck(cublasLtCreate(&lt_handle_), "cublasLtCreate failed!"); }

  ~CublasLtEmulationContext()
  {
    if (workspace_ptr_ != nullptr)
      cudaErrorCheck(cudaFree(workspace_ptr_), "cudaFree fp64 emulation workspace failed!");
    cublasErrorCheck(cublasLtDestroy(lt_handle_), "cublasLtDestroy failed!");
  }

  CublasLtEmulationContext(const CublasLtEmulationContext&)            = delete;
  CublasLtEmulationContext& operator=(const CublasLtEmulationContext&) = delete;
  CublasLtEmulationContext(CublasLtEmulationContext&&)                 = delete;
  CublasLtEmulationContext& operator=(CublasLtEmulationContext&&)      = delete;

  void ensureWorkspace(const std::size_t workspace_size_bytes)
  {
    if (workspace_size_bytes > workspace_size_bytes_)
    {
      if (workspace_ptr_ != nullptr)
      {
        cudaErrorCheck(cudaFree(workspace_ptr_), "cudaFree fp64 emulation workspace failed!");
        workspace_ptr_ = nullptr;
      }
      if (workspace_size_bytes > 0)
        cudaErrorCheck(cudaMalloc(&workspace_ptr_, workspace_size_bytes),
                       "cudaMalloc fp64 emulation workspace failed!");

      workspace_size_bytes_ = workspace_size_bytes;
    }
  }

  cublasLtHandle_t getLtHandle() const { return lt_handle_; }

  void* getWorkspacePtr() const { return workspace_ptr_; }

  std::size_t getWorkspaceSize() const { return workspace_size_bytes_; }

private:
  cublasLtHandle_t lt_handle_       = nullptr;
  void* workspace_ptr_              = nullptr;
  std::size_t workspace_size_bytes_ = 0;
};

} // namespace compute
} // namespace qmcplusplus

#endif  // defined(QMC_BLAS_FP64_EMULATION) && !defined(QMC_CUDA2HIP)

#endif  // QMCPLUSPLUS_CUBLASLTEMULATIONCONTEXT_HPP
