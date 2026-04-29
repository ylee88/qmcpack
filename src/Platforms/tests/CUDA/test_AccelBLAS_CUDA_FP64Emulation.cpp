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
#include <CPU/BLAS.hpp>
#include <DualAllocatorAliases.hpp>
#include <OhmmsPETE/OhmmsMatrix.h>
#include <OhmmsPETE/OhmmsVector.h>

namespace qmcplusplus
{
#if defined(QMC_BLAS_FP64_EMULATION)
namespace
{
using mat_t = Matrix<double, PinnedDualAllocator<double>>;
using cmat_t = Matrix<std::complex<double>, PinnedDualAllocator<std::complex<double>>>;

void fill_mat(mat_t& mat, const double scale)
{
  for (int j = 0; j < mat.rows(); j++)
    for (int i = 0; i < mat.cols(); i++)
      mat[j][i] = (i + 1) * scale + (j + 2);
}

void fill_mat(cmat_t& mat, const double scale)
{
  for (int j = 0; j < mat.rows(); j++)
    for (int i = 0; i < mat.cols(); i++)
    {
      const double real_val = (i + 1) * scale + (j + 2);
      const double imag_val = (j + 1) * 0.5 * scale - (i + 3);
      mat[j][i]             = std::complex<double>(real_val, imag_val);
    }
}

} // namespace

TEST_CASE("AccelBLAS_CUDA_DGEMM_policy", "[CUDA][BLAS]")
{
  constexpr int M = 23;
  constexpr int N = 19;
  constexpr int K = 17;

  mat_t A(K, M);
  mat_t B(N, K);
  mat_t C_native(N, M);
  mat_t C_policy_native(N, M);
  mat_t C_ref(N, M);

  fill_mat(A, 2.0);
  fill_mat(B, 3.0);

  A.updateTo();
  B.updateTo();
  C_native.updateTo();
  C_policy_native.updateTo();

  compute::Queue<PlatformKind::CUDA> queue;
  compute::BLASHandle<PlatformKind::CUDA> h_blas(queue);

  const double alpha = 1.0;
  const double beta  = 0.0;

  compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                      C_native.device_data(), M);

  compute::BLASPolicy native_policy;
  native_policy.fp64_emulation_mode = compute::FP64EmulationMode::NATIVE;
  compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                      C_policy_native.device_data(), M, native_policy);

  queue.sync();
  C_native.updateFrom();
  C_policy_native.updateFrom();

  BLAS::gemm('N', 'N', M, N, K, alpha, A.data(), M, B.data(), K, beta, C_ref.data(), M);

  for (int j = 0; j < N; j++)
    for (int i = 0; i < M; i++)
    {
      CHECK(C_native[j][i] == Approx(C_ref[j][i]));
      CHECK(C_policy_native[j][i] == Approx(C_ref[j][i]));
    }
}

TEST_CASE("AccelBLAS_CUDA_DGEMM_policy_batched", "[CUDA][BLAS]")
{
  constexpr int M           = 17;
  constexpr int N           = 13;
  constexpr int K           = 11;
  constexpr int batch_count = 2;

  mat_t A0(K, M), A1(K, M);
  mat_t B0(N, K), B1(N, K);
  mat_t C0(N, M), C1(N, M);

  fill_mat(A0, 2.0);
  fill_mat(A1, 4.0);
  fill_mat(B0, 1.5);
  fill_mat(B1, 3.5);

  A0.updateTo();
  A1.updateTo();
  B0.updateTo();
  B1.updateTo();
  C0.updateTo();
  C1.updateTo();

  Vector<const double*, PinnedDualAllocator<const double*>> Aarr(batch_count), Barr(batch_count);
  Vector<double*, PinnedDualAllocator<double*>> Carr(batch_count);
  Aarr[0] = A0.device_data();
  Aarr[1] = A1.device_data();
  Barr[0] = B0.device_data();
  Barr[1] = B1.device_data();
  Carr[0] = C0.device_data();
  Carr[1] = C1.device_data();

  Aarr.updateTo();
  Barr.updateTo();
  Carr.updateTo();

  compute::Queue<PlatformKind::CUDA> queue;
  compute::BLASHandle<PlatformKind::CUDA> h_blas(queue);

  const double alpha = 1.0;
  const double beta  = 0.0;

  compute::BLASPolicy native_policy;
  native_policy.fp64_emulation_mode = compute::FP64EmulationMode::NATIVE;
  compute::BLAS::gemm_batched(h_blas, 'N', 'N', M, N, K, alpha, Aarr.device_data(), M, Barr.device_data(), K, beta,
                              Carr.device_data(), M, batch_count, native_policy);

  queue.sync();
  C0.updateFrom();
  C1.updateFrom();

  mat_t C0_ref(N, M), C1_ref(N, M);
  BLAS::gemm('N', 'N', M, N, K, alpha, A0.data(), M, B0.data(), K, beta, C0_ref.data(), M);
  BLAS::gemm('N', 'N', M, N, K, alpha, A1.data(), M, B1.data(), K, beta, C1_ref.data(), M);

  for (int j = 0; j < N; j++)
    for (int i = 0; i < M; i++)
    {
      CHECK(C0[j][i] == Approx(C0_ref[j][i]));
      CHECK(C1[j][i] == Approx(C1_ref[j][i]));
    }
}

TEST_CASE("AccelBLAS_CUDA_DGEMM_policy_emulation_request", "[CUDA][BLAS]")
{
  constexpr int M = 8;
  constexpr int N = 8;
  constexpr int K = 8;

  mat_t A(K, M);
  mat_t B(N, K);
  mat_t C(N, M);
  fill_mat(A, 2.0);
  fill_mat(B, 3.0);
  A.updateTo();
  B.updateTo();
  C.updateTo();

  compute::Queue<PlatformKind::CUDA> queue;
  compute::BLASHandle<PlatformKind::CUDA> h_blas(queue);

  compute::BLASPolicy emu_policy;
  emu_policy.fp64_emulation_mode = compute::FP64EmulationMode::FIXED_POINT;
  emu_policy.min_workspace_bytes = 64ULL * 1024ULL * 1024ULL;
  emu_policy.max_mantissa_bits   = 55;

  CHECK_NOTHROW(compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, 1.0, A.device_data(), M, B.device_data(), K, 0.0,
                                    C.device_data(), M, emu_policy));
  queue.sync();
}

TEST_CASE("AccelBLAS_CUDA_ZGEMM_policy", "[CUDA][BLAS]")
{
  constexpr int M = 23;
  constexpr int N = 19;
  constexpr int K = 17;

  cmat_t A(K, M);
  cmat_t B(N, K);
  cmat_t C_native(N, M);
  cmat_t C_policy_native(N, M);
  cmat_t C_ref(N, M);

  fill_mat(A, 2.0);
  fill_mat(B, 3.0);

  A.updateTo();
  B.updateTo();
  C_native.updateTo();
  C_policy_native.updateTo();

  compute::Queue<PlatformKind::CUDA> queue;
  compute::BLASHandle<PlatformKind::CUDA> h_blas(queue);

  const std::complex<double> alpha(1.0, 0.0);
  const std::complex<double> beta(0.0, 0.0);

  compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                      C_native.device_data(), M);

  compute::BLASPolicy native_policy;
  native_policy.fp64_emulation_mode = compute::FP64EmulationMode::NATIVE;
  compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                      C_policy_native.device_data(), M, native_policy);

  queue.sync();
  C_native.updateFrom();
  C_policy_native.updateFrom();

  BLAS::gemm('N', 'N', M, N, K, alpha, A.data(), M, B.data(), K, beta, C_ref.data(), M);

  for (int j = 0; j < N; j++)
    for (int i = 0; i < M; i++)
    {
      CHECK(C_native[j][i].real() == Approx(C_ref[j][i].real()));
      CHECK(C_native[j][i].imag() == Approx(C_ref[j][i].imag()));
      CHECK(C_policy_native[j][i].real() == Approx(C_ref[j][i].real()));
      CHECK(C_policy_native[j][i].imag() == Approx(C_ref[j][i].imag()));
    }
}

TEST_CASE("AccelBLAS_CUDA_ZGEMM_policy_batched", "[CUDA][BLAS]")
{
  constexpr int M           = 17;
  constexpr int N           = 13;
  constexpr int K           = 11;
  constexpr int batch_count = 2;

  cmat_t A0(K, M), A1(K, M);
  cmat_t B0(N, K), B1(N, K);
  cmat_t C0(N, M), C1(N, M);

  fill_mat(A0, 2.0);
  fill_mat(A1, 4.0);
  fill_mat(B0, 1.5);
  fill_mat(B1, 3.5);

  A0.updateTo();
  A1.updateTo();
  B0.updateTo();
  B1.updateTo();
  C0.updateTo();
  C1.updateTo();

  Vector<const std::complex<double>*, PinnedDualAllocator<const std::complex<double>*>> Aarr(batch_count),
      Barr(batch_count);
  Vector<std::complex<double>*, PinnedDualAllocator<std::complex<double>*>> Carr(batch_count);
  Aarr[0] = A0.device_data();
  Aarr[1] = A1.device_data();
  Barr[0] = B0.device_data();
  Barr[1] = B1.device_data();
  Carr[0] = C0.device_data();
  Carr[1] = C1.device_data();

  Aarr.updateTo();
  Barr.updateTo();
  Carr.updateTo();

  compute::Queue<PlatformKind::CUDA> queue;
  compute::BLASHandle<PlatformKind::CUDA> h_blas(queue);

  const std::complex<double> alpha(1.0, 0.0);
  const std::complex<double> beta(0.0, 0.0);

  compute::BLASPolicy native_policy;
  native_policy.fp64_emulation_mode = compute::FP64EmulationMode::NATIVE;
  compute::BLAS::gemm_batched(h_blas, 'N', 'N', M, N, K, alpha, Aarr.device_data(), M, Barr.device_data(), K, beta,
                              Carr.device_data(), M, batch_count, native_policy);

  queue.sync();
  C0.updateFrom();
  C1.updateFrom();

  cmat_t C0_ref(N, M), C1_ref(N, M);
  BLAS::gemm('N', 'N', M, N, K, alpha, A0.data(), M, B0.data(), K, beta, C0_ref.data(), M);
  BLAS::gemm('N', 'N', M, N, K, alpha, A1.data(), M, B1.data(), K, beta, C1_ref.data(), M);

  for (int j = 0; j < N; j++)
    for (int i = 0; i < M; i++)
    {
      CHECK(C0[j][i].real() == Approx(C0_ref[j][i].real()));
      CHECK(C0[j][i].imag() == Approx(C0_ref[j][i].imag()));
      CHECK(C1[j][i].real() == Approx(C1_ref[j][i].real()));
      CHECK(C1[j][i].imag() == Approx(C1_ref[j][i].imag()));
    }
}

TEST_CASE("AccelBLAS_CUDA_ZGEMM_policy_emulation_request", "[CUDA][BLAS]")
{
  constexpr int M = 8;
  constexpr int N = 8;
  constexpr int K = 8;

  cmat_t A(K, M);
  cmat_t B(N, K);
  cmat_t C(N, M);
  fill_mat(A, 2.0);
  fill_mat(B, 3.0);
  A.updateTo();
  B.updateTo();
  C.updateTo();

  compute::Queue<PlatformKind::CUDA> queue;
  compute::BLASHandle<PlatformKind::CUDA> h_blas(queue);

  compute::BLASPolicy emu_policy;
  emu_policy.fp64_emulation_mode = compute::FP64EmulationMode::FIXED_POINT;
  emu_policy.min_workspace_bytes = 64ULL * 1024ULL * 1024ULL;
  emu_policy.max_mantissa_bits   = 55;

  const std::complex<double> alpha(1.0, 0.0);
  const std::complex<double> beta(0.0, 0.0);
  CHECK_NOTHROW(compute::BLAS::gemm(h_blas, 'N', 'N', M, N, K, alpha, A.device_data(), M, B.device_data(), K, beta,
                                    C.device_data(), M, emu_policy));
  queue.sync();
}

#else

TEST_CASE("AccelBLAS_CUDA_DGEMM_policy_disabled", "[CUDA][BLAS]") { SUCCEED("QMC_BLAS_FP64_EMULATION is OFF"); }

#endif

} // namespace qmcplusplus
