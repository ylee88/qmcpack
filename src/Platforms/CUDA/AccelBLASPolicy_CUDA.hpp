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

#ifndef QMCPLUSPLUS_ACCELBLAS_POLICY_CUDA_H
#define QMCPLUSPLUS_ACCELBLAS_POLICY_CUDA_H

#include "config.h"

#include <cstddef>

namespace qmcplusplus
{
namespace compute
{

#if defined(QMC_BLAS_FP64_EMULATION) && !defined(QMC_CUDA2HIP)
enum class FP64EmulationMode
{
  NATIVE,
  FIXED_POINT,
};

struct BLASPolicy
{
  FP64EmulationMode fp64_emulation_mode = FP64EmulationMode::NATIVE;
  std::size_t min_workspace_bytes = 128ULL * 1024ULL * 1024ULL; // 128 MiB
  int max_mantissa_bits           = 55;
};
#endif

} // namespace compute
} // namespace qmcplusplus

#endif
