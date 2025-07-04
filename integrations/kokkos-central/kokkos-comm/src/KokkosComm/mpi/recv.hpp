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

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include "impl/pack_traits.hpp"
#include "impl/types.hpp"

namespace KokkosComm::mpi {

template <KokkosView RecvView>
void recv(const RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Status *status) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::recv");

  if (KokkosComm::is_contiguous(rv)) {
    using ScalarType = typename RecvView::non_const_value_type;
    MPI_Recv(KokkosComm::data_handle(rv), KokkosComm::span(rv), KokkosComm::Impl::mpi_type_v<ScalarType>, src, tag,
             comm, status);
  } else {
    throw std::runtime_error("only contiguous views supported for low-level recv");
  }

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::recv");

  using KCPT   = KokkosComm::PackTraits<RecvView>;
  using Packer = typename KCPT::packer_type;
  using Args   = typename Packer::args_type;

  if (!KokkosComm::is_contiguous(rv)) {
    Args args = Packer::allocate_packed_for(space, "packed", rv);
    space.fence("Fence after allocation before MPI_Recv");
    MPI_Recv(KokkosComm::data_handle(args.view), args.count, args.datatype, src, tag, comm, MPI_STATUS_IGNORE);
    Packer::unpack_into(space, rv, args.view);
  } else {
    using RecvScalar = typename RecvView::value_type;
    space.fence("Fence before MPI_Recv");  // prevent work in `space` from writing to recv buffer
    MPI_Recv(KokkosComm::data_handle(rv), KokkosComm::span(rv), KokkosComm::Impl::mpi_type_v<RecvScalar>, src, tag,
             comm, MPI_STATUS_IGNORE);
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::mpi
