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

#include "catch.hpp"

#include <AccelBLAS.hpp>
#include <DualAllocatorAliases.hpp>
#include <OhmmsPETE/OhmmsMatrix.h>

namespace qmcplusplus
{
#if !defined(QMC_BLAS_FP64_EMULATION)
TEST_CASE("AccelBLAS CUDA DGEMM policy benchmark disabled", "[CUDA][BLAS][.benchmark]")
{
  SUCCEED("QMC_BLAS_FP64_EMULATION is OFF");
}
#else

namespace
{
using mat_t = Matrix<double, PinnedDualAllocator<double>>;

void fill_mat(mat_t& mat)
{
  for (int j = 0; j < mat.rows(); j++)
    for (int i = 0; i < mat.cols(); i++)
      mat[j][i] = static_cast<double>((i + j) % 11 + 1);
}

} // namespace

TEST_CASE("AccelBLAS CUDA DGEMM policy benchmark", "[CUDA][BLAS][.benchmark]")
{
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;

  mat_t A(K, M);
  mat_t B(N, K);
  mat_t C(N, M);

  fill_mat(A);
  fill_mat(B);
  A.updateTo();
  B.updateTo();
  C.updateTo();

  compute::Queue<PlatformKind::CUDA> queue;
  compute::BLASHandle<PlatformKind::CUDA> h_blas(queue);

  const double alpha = 1.0;
  const double beta  = 0.0;

  compute::BLASPolicy<PlatformKind::CUDA> native_policy;
  native_policy.fp64_emulation_mode = compute::FP64EmulationMode::NATIVE;

  BENCHMARK_ADVANCED("[CUDA/f64] dgemm_native_1024x1024x1024")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                          C.device_data(), M, native_policy);
      queue.sync();
    });
  };

  compute::BLASPolicy<PlatformKind::CUDA> emu_policy;
  emu_policy.fp64_emulation_mode  = compute::FP64EmulationMode::FIXED_POINT;
  emu_policy.workspace_size_bytes = 128ULL * 1024ULL * 1024ULL;
  emu_policy.max_mantissa_bits    = 55;

  BENCHMARK_ADVANCED("[CUDA/f64] dgemm_emu_fixedpoint_1024x1024x1024")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                          C.device_data(), M, emu_policy);
      queue.sync();
    });
  };
}

#endif // !defined(QMC_BLAS_FP64_EMULATION)

} // namespace qmcplusplus
