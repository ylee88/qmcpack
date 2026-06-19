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

#include <optional>

#if defined(QMC_BLAS_FP64_EMULATION) && !defined(QMC_CUDA2HIP)
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#endif

namespace qmcplusplus
{
namespace compute
{

enum class FP64EmulationMode
{
  NATIVE,
  FIXED_POINT,
};

struct BLASPolicy
{
  FP64EmulationMode fp64_emulation_mode = FP64EmulationMode::NATIVE;
  int max_mantissa_bits                 = 55;
};

#if defined(QMC_BLAS_FP64_EMULATION) && !defined(QMC_CUDA2HIP)
/// Read QMCPACK_FP64_EMU_DELAY and QMCPACK_FP64_EMU_MANTISSA_BITS from the environment.
/// Returns std::nullopt (native cuBLAS) when QMCPACK_FP64_EMU_DELAY is unset or not "1".
inline std::optional<BLASPolicy> blasPolicyFromEnv()
{
  const char* enable_env = std::getenv("QMCPACK_FP64_EMU_DELAY");
  if (!enable_env || std::string_view(enable_env) != "1")
    return std::nullopt;

  BLASPolicy policy;
  policy.fp64_emulation_mode = FP64EmulationMode::FIXED_POINT;

  const char* bits_env = std::getenv("QMCPACK_FP64_EMU_MANTISSA_BITS");
  if (bits_env)
  {
    const int bits = std::atoi(bits_env);
    if (bits <= 0 || bits > 55)
      throw std::runtime_error("QMCPACK_FP64_EMU_MANTISSA_BITS must be in [1,55].");
    policy.max_mantissa_bits = bits;
  }

  return policy;
}
#endif // defined(QMC_BLAS_FP64_EMULATION) && !defined(QMC_CUDA2HIP)

} // namespace compute
} // namespace qmcplusplus

#endif
