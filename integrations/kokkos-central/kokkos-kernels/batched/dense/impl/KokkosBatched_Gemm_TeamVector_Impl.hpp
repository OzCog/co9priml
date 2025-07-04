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
#ifndef KOKKOSBATCHED_GEMM_TEAMVECTOR_IMPL_HPP
#define KOKKOSBATCHED_GEMM_TEAMVECTOR_IMPL_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Gemm_TeamVector_Internal.hpp"

namespace KokkosBatched {

///
/// Team Impl
/// =========

///
/// Implemented:
/// NT/NT, T/NT, NT/T, T/T
///
/// Not yet implemented (ConjTranspose)
/// CT/NT, NT/CT, CT/CT
///

///
/// NT/NT
///

template <typename MemberType>
struct TeamVectorGemm<MemberType, Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // C = beta C + alpha A B
    // C (m x n), A(m x k), B(k x n)
    return TeamVectorGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, C.extent(0), C.extent(1), A.extent(1), alpha, A.data(), A.stride(0), A.stride(1), B.data(), B.stride(0),
        B.stride(1), beta, C.data(), C.stride(0), C.stride(1));
  }
};

///
/// T/NT
///

template <typename MemberType>
struct TeamVectorGemm<MemberType, Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // C = beta C + alpha A B
    // C (m x n), A(m x k), B(k x n)
    return TeamVectorGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, C.extent(0), C.extent(1), A.extent(0), alpha, A.data(), A.stride(1), A.stride(0), B.data(), B.stride(0),
        B.stride(1), beta, C.data(), C.stride(0), C.stride(1));
  }
};

///
/// NT/T
///

template <typename MemberType>
struct TeamVectorGemm<MemberType, Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // C = beta C + alpha A B
    // C (m x n), A(m x k), B(k x n)
    return TeamVectorGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, C.extent(0), C.extent(1), A.extent(1), alpha, A.data(), A.stride(0), A.stride(1), B.data(), B.stride(1),
        B.stride(0), beta, C.data(), C.stride(0), C.stride(1));
  }
};

///
/// T/T
///

template <typename MemberType>
struct TeamVectorGemm<MemberType, Trans::Transpose, Trans::Transpose, Algo::Gemm::Unblocked> {
  template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                           const BViewType &B, const ScalarType beta, const CViewType &C) {
    // C = beta C + alpha A B
    // C (m x n), A(m x k), B(k x n)
    return TeamVectorGemmInternal<Algo::Gemm::Unblocked>::invoke(
        member, C.extent(0), C.extent(1), A.extent(0), alpha, A.data(), A.stride(1), A.stride(0), B.data(), B.stride(1),
        B.stride(0), beta, C.data(), C.stride(0), C.stride(1));
  }
};

}  // namespace KokkosBatched

#endif
