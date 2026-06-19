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

#include <complex>
#include <cstdlib>
#include <vector>

namespace qmcplusplus
{
#if !defined(QMC_BLAS_FP64_EMULATION)
TEST_CASE("AccelBLAS_CUDA FP64 emulation benchmark disabled", "[CUDA][BLAS][.benchmark]")
{ SUCCEED("QMC_BLAS_FP64_EMULATION is OFF"); }
#else

namespace
{
using mat_t = Matrix<double, PinnedDualAllocator<double>>;
using cmat_t = Matrix<std::complex<double>, PinnedDualAllocator<std::complex<double>>>;

compute::BLASPolicy makeEmuPolicy()
{
  compute::BLASPolicy p;
  p.fp64_emulation_mode = compute::FP64EmulationMode::FIXED_POINT;
  if (const char* env = std::getenv("QMCPACK_FP64_EMU_MANTISSA_BITS"))
    p.max_mantissa_bits = std::atoi(env);
  // else: default 55 from BLASPolicy constructor
  return p;
}

void fill_mat(mat_t& mat)
{
  for (int j = 0; j < mat.rows(); j++)
    for (int i = 0; i < mat.cols(); i++)
      mat[j][i] = static_cast<double>((i + j) % 11 + 1);
}

void fill_mat(cmat_t& mat)
{
  for (int j = 0; j < mat.rows(); j++)
    for (int i = 0; i < mat.cols(); i++)
      mat[j][i] = std::complex<double>(static_cast<double>((i + j) % 11 + 1), static_cast<double>((2 * i + j) % 7 - 3));
}

} // namespace

TEST_CASE("AccelBLAS_CUDA DGEMM FP64 emulation benchmark", "[CUDA][BLAS][.benchmark]")
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

  compute::BLASPolicy native_policy;
  native_policy.fp64_emulation_mode = compute::FP64EmulationMode::NATIVE;

  BENCHMARK_ADVANCED("[CUDA/f64] dgemm_native_1024x1024x1024")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                          C.device_data(), M, native_policy);
      queue.sync();
    });
  };

  compute::BLASPolicy emu_policy = makeEmuPolicy();

  BENCHMARK_ADVANCED("[CUDA/f64] dgemm_emu_fixedpoint_1024x1024x1024")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                          C.device_data(), M, emu_policy);
      queue.sync();
    });
  };
}

TEST_CASE("AccelBLAS_CUDA DGEMM batched FP64 emulation benchmark", "[CUDA][BLAS][.benchmark]")
{
  constexpr int M           = 256;
  constexpr int N           = 256;
  constexpr int K           = 256;
  constexpr int batch_count = 32;

  compute::Queue<PlatformKind::CUDA> queue;
  compute::BLASHandle<PlatformKind::CUDA> h_blas(queue);

  const double alpha = 1.0;
  const double beta  = 0.0;

  compute::BLASPolicy native_policy;
  native_policy.fp64_emulation_mode = compute::FP64EmulationMode::NATIVE;

  compute::BLASPolicy emu_policy = makeEmuPolicy();

  std::vector<mat_t> A_b(batch_count), B_b(batch_count), C_batched(batch_count);
  for (int ib = 0; ib < batch_count; ++ib)
  {
    A_b[ib]       = mat_t(K, M);
    B_b[ib]       = mat_t(N, K);
    C_batched[ib] = mat_t(N, M);

    fill_mat(A_b[ib]);
    fill_mat(B_b[ib]);
    A_b[ib].updateTo();
    B_b[ib].updateTo();
    C_batched[ib].updateTo();
  }

  Vector<const double*, PinnedDualAllocator<const double*>> Aarr(batch_count), Barr(batch_count);
  Vector<double*, PinnedDualAllocator<double*>> Carr_batched(batch_count);
  for (int ib = 0; ib < batch_count; ++ib)
  {
    Aarr[ib]         = A_b[ib].device_data();
    Barr[ib]         = B_b[ib].device_data();
    Carr_batched[ib] = C_batched[ib].device_data();
  }
  Aarr.updateTo();
  Barr.updateTo();
  Carr_batched.updateTo();

  compute::BLAS::gemm_batched(h_blas, 'N', 'N', M, N, K, alpha, Aarr.device_data(), M, Barr.device_data(), K, beta,
                              Carr_batched.device_data(), M, batch_count, native_policy);
  compute::BLAS::gemm_batched(h_blas, 'N', 'N', M, N, K, alpha, Aarr.device_data(), M, Barr.device_data(), K, beta,
                              Carr_batched.device_data(), M, batch_count, emu_policy);
  queue.sync();

  BENCHMARK_ADVANCED("[CUDA/f64] dgemm_native_batched_256x256x256_bs32")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      compute::BLAS::gemm_batched(h_blas, 'N', 'N', M, N, K, alpha, Aarr.device_data(), M, Barr.device_data(), K, beta,
                                  Carr_batched.device_data(), M, batch_count, native_policy);
      queue.sync();
    });
  };

  BENCHMARK_ADVANCED("[CUDA/f64] dgemm_emu_fixedpoint_batched_256x256x256_bs32")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      compute::BLAS::gemm_batched(h_blas, 'N', 'N', M, N, K, alpha, Aarr.device_data(), M, Barr.device_data(), K, beta,
                                  Carr_batched.device_data(), M, batch_count, emu_policy);
      queue.sync();
    });
  };
}

TEST_CASE("AccelBLAS_CUDA ZGEMM FP64 emulation benchmark", "[CUDA][BLAS][.benchmark]")
{
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;

  cmat_t A(K, M);
  cmat_t B(N, K);
  cmat_t C(N, M);

  fill_mat(A);
  fill_mat(B);
  A.updateTo();
  B.updateTo();
  C.updateTo();

  compute::Queue<PlatformKind::CUDA> queue;
  compute::BLASHandle<PlatformKind::CUDA> h_blas(queue);

  const std::complex<double> alpha(1.0, 0.0);
  const std::complex<double> beta(0.0, 0.0);

  compute::BLASPolicy native_policy;
  native_policy.fp64_emulation_mode = compute::FP64EmulationMode::NATIVE;

  BENCHMARK_ADVANCED("[CUDA/zf64] zgemm_native_1024x1024x1024")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                          C.device_data(), M, native_policy);
      queue.sync();
    });
  };

  compute::BLASPolicy emu_policy = makeEmuPolicy();

  BENCHMARK_ADVANCED("[CUDA/zf64] zgemm_emu_fixedpoint_1024x1024x1024")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                          C.device_data(), M, emu_policy);
      queue.sync();
    });
  };
}

TEST_CASE("AccelBLAS_CUDA ZGEMM batched FP64 emulation benchmark", "[CUDA][BLAS][.benchmark]")
{
  constexpr int M           = 256;
  constexpr int N           = 256;
  constexpr int K           = 256;
  constexpr int batch_count = 32;

  compute::Queue<PlatformKind::CUDA> queue;
  compute::BLASHandle<PlatformKind::CUDA> h_blas(queue);

  const std::complex<double> alpha(1.0, 0.0);
  const std::complex<double> beta(0.0, 0.0);

  compute::BLASPolicy native_policy;
  native_policy.fp64_emulation_mode = compute::FP64EmulationMode::NATIVE;

  compute::BLASPolicy emu_policy = makeEmuPolicy();

  std::vector<cmat_t> A_b(batch_count), B_b(batch_count), C_batched(batch_count);
  for (int ib = 0; ib < batch_count; ++ib)
  {
    A_b[ib]       = cmat_t(K, M);
    B_b[ib]       = cmat_t(N, K);
    C_batched[ib] = cmat_t(N, M);

    fill_mat(A_b[ib]);
    fill_mat(B_b[ib]);
    A_b[ib].updateTo();
    B_b[ib].updateTo();
    C_batched[ib].updateTo();
  }

  Vector<const std::complex<double>*, PinnedDualAllocator<const std::complex<double>*>> Aarr(batch_count),
      Barr(batch_count);
  Vector<std::complex<double>*, PinnedDualAllocator<std::complex<double>*>> Carr_batched(batch_count);
  for (int ib = 0; ib < batch_count; ++ib)
  {
    Aarr[ib]         = A_b[ib].device_data();
    Barr[ib]         = B_b[ib].device_data();
    Carr_batched[ib] = C_batched[ib].device_data();
  }
  Aarr.updateTo();
  Barr.updateTo();
  Carr_batched.updateTo();

  compute::BLAS::gemm_batched(h_blas, 'N', 'N', M, N, K, alpha, Aarr.device_data(), M, Barr.device_data(), K, beta,
                              Carr_batched.device_data(), M, batch_count, native_policy);
  compute::BLAS::gemm_batched(h_blas, 'N', 'N', M, N, K, alpha, Aarr.device_data(), M, Barr.device_data(), K, beta,
                              Carr_batched.device_data(), M, batch_count, emu_policy);
  queue.sync();

  BENCHMARK_ADVANCED("[CUDA/zf64] zgemm_native_batched_256x256x256_bs32")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      compute::BLAS::gemm_batched(h_blas, 'N', 'N', M, N, K, alpha, Aarr.device_data(), M, Barr.device_data(), K,
                                  beta, Carr_batched.device_data(), M, batch_count, native_policy);
      queue.sync();
    });
  };

  BENCHMARK_ADVANCED("[CUDA/zf64] zgemm_emu_fixedpoint_batched_256x256x256_bs32")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      compute::BLAS::gemm_batched(h_blas, 'N', 'N', M, N, K, alpha, Aarr.device_data(), M, Barr.device_data(), K,
                                  beta, Carr_batched.device_data(), M, batch_count, emu_policy);
      queue.sync();
    });
  };
}

#endif // !defined(QMC_BLAS_FP64_EMULATION)

} // namespace qmcplusplus
