#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2019 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////

set(SRC_DIR CUDA)
set(UTEST_EXE test_${SRC_DIR})
set(UTEST_NAME deterministic-unit_test_${SRC_DIR})

add_executable(${UTEST_EXE} test_CUDAallocator.cpp)
target_link_libraries(${UTEST_EXE} platform_runtime containers catch_main)

add_unit_test(${UTEST_NAME} 1 1 $<TARGET_FILE:${UTEST_EXE}>)

if(QMC_BLAS_FP64_EMULATION)
  set(UTEST_EXE test_${SRC_DIR}_accelblas)
  set(UTEST_NAME deterministic-unit_${UTEST_EXE})

  add_executable(${UTEST_EXE} test_AccelBLAS_CUDA_FP64Emulation.cpp)
  target_link_libraries(${UTEST_EXE} platform_runtime platform_LA containers catch_main)

  add_unit_test(${UTEST_NAME} 1 1 $<TARGET_FILE:${UTEST_EXE}>)

  if(BUILD_MICRO_BENCHMARKS)
    set(UTEST_EXE benchmark_${SRC_DIR}_accelblas_fp64)
    set(UTEST_NAME deterministic-unit_${UTEST_EXE})

    add_executable(${UTEST_EXE} benchmark_AccelBLAS_CUDA_FP64Emulation.cpp)
    target_link_libraries(${UTEST_EXE} platform_runtime platform_LA containers catch_main)

    add_unit_test(${UTEST_NAME} 1 1 $<TARGET_FILE:${UTEST_EXE}>)
  endif()
endif()
