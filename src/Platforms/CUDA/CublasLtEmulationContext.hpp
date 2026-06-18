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

#include <functional>
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace qmcplusplus
{
namespace compute
{

struct AlgoCacheKey
{
  int m, n, k;
  int lda, ldb, ldc;
  int batch_count;
  cublasOperation_t transa, transb;
  cudaDataType_t data_type;
  int max_mantissa_bits;

  bool operator==(const AlgoCacheKey& o) const noexcept
  {
    return m == o.m && n == o.n && k == o.k && lda == o.lda && ldb == o.ldb && ldc == o.ldc &&
           batch_count == o.batch_count && transa == o.transa && transb == o.transb &&
           data_type == o.data_type && max_mantissa_bits == o.max_mantissa_bits;
  }
};

struct AlgoCacheKeyHash
{
  std::size_t operator()(const AlgoCacheKey& k) const noexcept
  {
    // XOR-combine all fields using a simple mixing step
    auto h = [](std::size_t seed, std::size_t v) { return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2)); };
    std::size_t s = 0;
    s = h(s, std::hash<int>{}(k.m));
    s = h(s, std::hash<int>{}(k.n));
    s = h(s, std::hash<int>{}(k.k));
    s = h(s, std::hash<int>{}(k.lda));
    s = h(s, std::hash<int>{}(k.ldb));
    s = h(s, std::hash<int>{}(k.ldc));
    s = h(s, std::hash<int>{}(k.batch_count));
    s = h(s, std::hash<int>{}(static_cast<int>(k.transa)));
    s = h(s, std::hash<int>{}(static_cast<int>(k.transb)));
    s = h(s, std::hash<int>{}(static_cast<int>(k.data_type)));
    s = h(s, std::hash<int>{}(k.max_mantissa_bits));
    return s;
  }
};

struct AlgoCacheEntry
{
  cublasLtMatmulAlgo_t algo;
  std::size_t workspace_size;
};

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
      void* new_workspace_ptr = nullptr;
      if (workspace_size_bytes > 0)
        cudaErrorCheck(cudaMalloc(&new_workspace_ptr, workspace_size_bytes),
                       "cudaMalloc fp64 emulation workspace failed!");

      if (workspace_ptr_ != nullptr)
        cudaErrorCheck(cudaFree(workspace_ptr_), "cudaFree fp64 emulation workspace failed!");

      workspace_ptr_        = new_workspace_ptr;
      workspace_size_bytes_ = workspace_size_bytes;
    }
  }

  // Look up the cached algo for this GEMM shape, or run heuristic selection and cache it.
  // The descriptors must already be fully configured (shapes, batch attributes, emulation desc).
  // max_workspace_budget is the workspace ceiling passed to cublasLt heuristic; actual allocation
  // is deferred to ensureWorkspace(algo_entry.workspace_size) after this call returns.
  const AlgoCacheEntry& getOrSelectAlgo(const AlgoCacheKey& key,
                                        cublasLtMatmulDesc_t op_desc,
                                        cublasLtMatrixLayout_t a_desc,
                                        cublasLtMatrixLayout_t b_desc,
                                        cublasLtMatrixLayout_t c_desc,
                                        std::size_t max_workspace_budget)
  {
    auto it = algo_cache_.find(key);
    if (it != algo_cache_.end())
    {
      // Cache hit: promote to front of LRU list
      lru_order_.splice(lru_order_.begin(), lru_order_, it->second.lru_pos);
      return it->second.entry;
    }

    // Cache miss: run heuristic selection
    cublasLtMatmulPreference_t pref = nullptr;
    cublasErrorCheck(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate failed!");
    cublasErrorCheck(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &max_workspace_budget, sizeof(max_workspace_budget)),
                     "cublasLtMatmulPreferenceSetAttribute failed!");

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_count = 0;
    cublasErrorCheck(cublasLtMatmulAlgoGetHeuristic(lt_handle_, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1,
                                                    &heuristic, &returned_count),
                     "cublasLtMatmulAlgoGetHeuristic failed!");
    cublasLtMatmulPreferenceDestroy(pref);

    if (returned_count == 0)
      throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic: no valid algorithm found for current workspace size.");

    // Evict LRU entry if at capacity
    if (algo_cache_.size() >= MAX_CACHE_ENTRIES)
    {
      algo_cache_.erase(lru_order_.back());
      lru_order_.pop_back();
    }

    // Insert new entry
    lru_order_.push_front(key);
    auto [ins_it, _] = algo_cache_.emplace(key, CacheValue{AlgoCacheEntry{heuristic.algo, heuristic.workspaceSize},
                                                            lru_order_.begin()});
    return ins_it->second.entry;
  }

  cublasLtHandle_t getLtHandle() const { return lt_handle_; }
  void*            getWorkspacePtr() const { return workspace_ptr_; }
  std::size_t      getWorkspaceSize() const { return workspace_size_bytes_; }

private:
  static constexpr std::size_t MAX_CACHE_ENTRIES = 256;

  struct CacheValue
  {
    AlgoCacheEntry entry;
    std::list<AlgoCacheKey>::iterator lru_pos;
  };

  cublasLtHandle_t lt_handle_       = nullptr;
  void*            workspace_ptr_   = nullptr;
  std::size_t      workspace_size_bytes_ = 0;

  std::unordered_map<AlgoCacheKey, CacheValue, AlgoCacheKeyHash> algo_cache_;
  std::list<AlgoCacheKey>                                         lru_order_;
};

} // namespace compute
} // namespace qmcplusplus

#endif  // defined(QMC_BLAS_FP64_EMULATION) && !defined(QMC_CUDA2HIP)

#endif  // QMCPLUSPLUS_CUBLASLTEMULATIONCONTEXT_HPP
