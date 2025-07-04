//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <iostream>

#include <mpi.h>
#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_IMPL_MPIEXT_H)
#include <mpi-ext.h>
#endif

namespace {

using namespace KokkosComm::mpi;

namespace {

void view_access_via_mpi_extension() {
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI) && defined(KOKKOS_ENABLE_CUDA)

#if defined(KOKKOSCOMM_IMPL_MPIEXT_H)
  EXPECT_EQ(1, MPIX_Query_cuda_support());
#else
  GTEST_SKIP() << "mpi-ext.h not available";
#endif

#elif defined(KOKKOSCOMM_IMPL_MPI_IS_MPICH)

#if defined(KOKKOS_ENABLE_CUDA)
  EXPECT_EQ(1, MPIX_Query_cuda_support());
#else
  GTEST_SKIP() << "CUDA not enabled";
#endif

#else
  GTEST_SKIP() << "unable to check for CUDA-aware MPI";
#endif
}

void doit() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    GTEST_SKIP() << "requires at least 2 ranks";
  }

  const int n = 1024 * 1024;
  Kokkos::View<double *> a("a", n);

  if (0 == rank) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) { a(i) = i; });
    Kokkos::fence();

    std::cerr << "sending buffer is " << a.data() << "-" << a.data() + n << std::endl;
    MPI_Send(a.data(), n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else if (1 == rank) {
    std::cerr << "recving buffer is " << a.data() << "-" << a.data() + n << std::endl;
    MPI_Recv(a.data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  auto a_h = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, a);
  Kokkos::fence();

  for (int i = 0; i < n; ++i) {
    if (a_h[i] != i) {
      ASSERT_EQ(a_h[i], i);
    }
  }
}
}  // namespace

TEST(MpiViewAccess, MpiExtension) { view_access_via_mpi_extension(); }
TEST(MpiViewAccess, Basic) { doit(); }

}  // namespace
