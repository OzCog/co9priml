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
#ifndef KOKKOSBATCHED_GEMV_TEAM_IMPL_HPP
#define KOKKOSBATCHED_GEMV_TEAM_IMPL_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Gemv_Team_Internal.hpp"
#include "KokkosBlas2_team_gemv.hpp"
#include <Kokkos_DynRankView.hpp>

namespace KokkosBatched {

///
/// Team Impl
/// =========

///
/// Implemented:
/// NT, T
///
/// Not yet implemented
/// CT

///
/// NT
///

template <typename MemberType>
struct TeamGemv<MemberType, Trans::NoTranspose, Algo::Gemv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename xViewType, typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const xViewType &x, const ScalarType beta, const yViewType &y) {
    if constexpr (Kokkos::is_dyn_rank_view<AViewType>::value) {
      assert(A.rank_dynamic() == 3 &&
             "Batched TeamGemv requires rank-3 A matrix (use "
             "KokkosBlas::TeamGemv for regular rank-2 matrix)");
    } else {
      static_assert(AViewType::rank == 3,
                    "Batched TeamGemv requires rank-3 A matrix (use "
                    "KokkosBlas::TeamGemv for regular rank-2 matrix)");
    }

    if (A.extent(0) == 1) {
      KokkosBlas::TeamGemv<MemberType, Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
          member, alpha, Kokkos::subview(A, 0, Kokkos::ALL, Kokkos::ALL), Kokkos::subview(x, 0, Kokkos::ALL), beta,
          Kokkos::subview(y, 0, Kokkos::ALL));
      return 0;
    }
    return TeamGemvInternal<Algo::Gemv::Unblocked>::template invoke<
        MemberType, ScalarType, typename AViewType::array_layout, typename AViewType::non_const_value_type>(
        member, A.extent(0), A.extent(1), A.extent(2), alpha, A.data(), A.stride(0), A.stride(1), A.stride(2), x.data(),
        x.stride(0), x.stride(1), beta, y.data(), y.stride(0), y.stride(1));
  }
};

template <typename MemberType>
struct TeamGemv<MemberType, Trans::NoTranspose, Algo::Gemv::Blocked> {
  template <typename ScalarType, typename AViewType, typename xViewType, typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType & /*member*/, const ScalarType /*alpha*/,
                                           const AViewType & /*A*/, const xViewType & /*x*/, const ScalarType /*beta*/,
                                           const yViewType & /*y*/) {
    /*     if constexpr (Kokkos::is_dyn_rank_view<AViewType>::value) {
          assert(A.rank_dynamic() == 3 &&
                 "Batched TeamGemv requires rank-3 A matrix (use "
                 "KokkosBlas::TeamGemv for regular rank-2 matrix)");
        } else {
          static_assert(AViewType::rank == 3,
                        "Batched TeamGemv requires rank-3 A matrix (use "
                        "KokkosBlas::TeamGemv for regular rank-2 matrix)");
        } */
    Kokkos::abort(
        "KokkosBlas::TeamGemv<Algo::Gemv::Blocked> for rank-3 matrix is NOT "
        "implemented");
  }
};

///
/// T
///

template <typename MemberType>
struct TeamGemv<MemberType, Trans::Transpose, Algo::Gemv::Unblocked> {
  template <typename ScalarType, typename AViewType, typename xViewType, typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const xViewType &x, const ScalarType beta, const yViewType &y) {
    if constexpr (Kokkos::is_dyn_rank_view<AViewType>::value) {
      assert(A.rank_dynamic() == 3 &&
             "Batched TeamGemv requires rank-3 A matrix (use "
             "KokkosBlas::TeamGemv for regular rank-2 matrix)");
    } else {
      static_assert(AViewType::rank == 3,
                    "Batched TeamGemv requires rank-3 A matrix (use "
                    "KokkosBlas::TeamGemv for regular rank-2 matrix)");
    }
    if (A.extent(0) == 1) {
      KokkosBlas::TeamGemv<MemberType, Trans::Transpose, Algo::Gemv::Unblocked>::invoke(
          member, alpha, Kokkos::subview(A, 0, Kokkos::ALL, Kokkos::ALL), Kokkos::subview(x, 0, Kokkos::ALL), beta,
          Kokkos::subview(y, 0, Kokkos::ALL));
      return 0;
    }
    return TeamGemvInternal<Algo::Gemv::Unblocked>::template invoke<
        MemberType, ScalarType, typename AViewType::array_layout, typename AViewType::non_const_value_type>(
        member, A.extent(0), A.extent(2), A.extent(1), alpha, A.data(), A.stride(0), A.stride(2), A.stride(1), x.data(),
        x.stride(0), x.stride(1), beta, y.data(), y.stride(0), y.stride(1));
  }
};

template <typename MemberType>
struct TeamGemv<MemberType, Trans::Transpose, Algo::Gemv::Blocked> {
  template <typename ScalarType, typename AViewType, typename xViewType, typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType & /*member*/, const ScalarType /*alpha*/,
                                           const AViewType & /*A*/, const xViewType & /*x*/, const ScalarType /*beta*/,
                                           const yViewType & /*y*/) {
    /*     if constexpr (Kokkos::is_dyn_rank_view<AViewType>::value) {
          assert(A.rank_dynamic() == 3 &&
                 "Batched TeamGemv requires rank-3 A matrix (use "
                 "KokkosBlas::TeamGemv for regular rank-2 matrix)");
        } else {
          static_assert(AViewType::rank == 3,
                        "Batched TeamGemv requires rank-3 A matrix (use "
                        "KokkosBlas::TeamGemv for regular rank-2 matrix)");
        } */
    Kokkos::abort(
        "KokkosBlas::TeamGemv<Algo::Gemv::Blocked> for rank-3 matrix is NOT "
        "implemented");
  }
};

}  // namespace KokkosBatched

#endif
