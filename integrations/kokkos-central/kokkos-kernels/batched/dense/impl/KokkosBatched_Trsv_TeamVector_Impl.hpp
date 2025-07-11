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
#ifndef KOKKOSBATCHED_TRSV_TEAMVECTOR_IMPL_HPP
#define KOKKOSBATCHED_TRSV_TEAMVECTOR_IMPL_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Trsv_TeamVector_Internal.hpp"

namespace KokkosBatched {
///
/// Team Impl
/// ===========

///
/// Implemented:
/// L/NT, U/NT, L/T, U/T
///
/// Not yet implemented
/// L/CT, U/CT

///
/// L/NT
///

template <typename MemberType, typename ArgDiag>
struct TeamVectorTrsv<MemberType, Uplo::Lower, Trans::NoTranspose, ArgDiag, Algo::Trsv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename bViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const bViewType &b) {
    return TeamVectorTrsvInternalLower<Algo::Trsv::Unblocked>::invoke(
        member, ArgDiag::use_unit_diag, A.extent(0), alpha, A.data(), A.stride(0), A.stride(1), b.data(), b.stride(0));
  }
};

///
/// L/T
///

template <typename MemberType, typename ArgDiag>
struct TeamVectorTrsv<MemberType, Uplo::Lower, Trans::Transpose, ArgDiag, Algo::Trsv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename bViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const bViewType &b) {
    return TeamVectorTrsvInternalUpper<Algo::Trsv::Unblocked>::invoke(
        member, ArgDiag::use_unit_diag, A.extent(1), alpha, A.data(), A.stride(1), A.stride(0), b.data(), b.stride(0));
  }
};

///
/// U/NT
///

template <typename MemberType, typename ArgDiag>
struct TeamVectorTrsv<MemberType, Uplo::Upper, Trans::NoTranspose, ArgDiag, Algo::Trsv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename bViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const bViewType &b) {
    return TeamVectorTrsvInternalUpper<Algo::Trsv::Unblocked>::invoke(
        member, ArgDiag::use_unit_diag, A.extent(0), alpha, A.data(), A.stride(0), A.stride(1), b.data(), b.stride(0));
  }
};

///
/// U/T
///

template <typename MemberType, typename ArgDiag>
struct TeamVectorTrsv<MemberType, Uplo::Upper, Trans::Transpose, ArgDiag, Algo::Trsv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename bViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const bViewType &b) {
    return TeamVectorTrsvInternalLower<Algo::Trsv::Unblocked>::invoke(
        member, ArgDiag::use_unit_diag, A.extent(1), alpha, A.data(), A.stride(1), A.stride(0), b.data(), b.stride(0));
  }
};

}  // namespace KokkosBatched

#endif
