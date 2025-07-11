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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_COPYVIEWS_HPP_
#define KOKKOS_COPYVIEWS_HPP_
#include <string>
#include <sstream>
#include <Kokkos_Parallel.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <Kokkos_Layout.hpp>
#include <impl/Kokkos_HostSpace_ZeroMemset.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class Layout>
struct ViewFillLayoutSelector {};

template <>
struct ViewFillLayoutSelector<Kokkos::LayoutLeft> {
  static const Kokkos::Iterate iterate = Kokkos::Iterate::Left;
};

template <>
struct ViewFillLayoutSelector<Kokkos::LayoutRight> {
  static const Kokkos::Iterate iterate = Kokkos::Iterate::Right;
};

}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill<ViewType, Layout, ExecSpace, 0, iType> {
  using ST = typename ViewType::non_const_value_type;
  ViewFill(const ViewType& a, const ST& val, const ExecSpace& space) {
    Kokkos::Impl::DeepCopy<typename ViewType::memory_space, Kokkos::HostSpace,
                           ExecSpace>(space, a.data(), &val, sizeof(ST));
  }
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill<ViewType, Layout, ExecSpace, 1, iType> {
  ViewType a;
  typename ViewType::const_value_type val;
  using policy_type = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_,
           const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-1D",
                         policy_type(space, 0, a.extent(0)), *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i) const { a(i) = val; }
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill<ViewType, Layout, ExecSpace, 2, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<2, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_,
           const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-2D",
                         policy_type(space, {0, 0}, {a.extent(0), a.extent(1)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1) const { a(i0, i1) = val; }
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill<ViewType, Layout, ExecSpace, 3, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<3, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_,
           const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for(
        "Kokkos::ViewFill-3D",
        policy_type(space, {0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}),
        *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2) const {
    a(i0, i1, i2) = val;
  }
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill<ViewType, Layout, ExecSpace, 4, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<4, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_,
           const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for(
        "Kokkos::ViewFill-4D",
        policy_type(space, {0, 0, 0, 0},
                    {a.extent(0), a.extent(1), a.extent(2), a.extent(3)}),
        *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3) const {
    a(i0, i1, i2, i3) = val;
  }
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill<ViewType, Layout, ExecSpace, 5, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<5, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_,
           const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-5D",
                         policy_type(space, {0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(2),
                                      a.extent(3), a.extent(4)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3, const iType& i4) const {
    a(i0, i1, i2, i3, i4) = val;
  }
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill<ViewType, Layout, ExecSpace, 6, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<6, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_,
           const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-6D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(2),
                                      a.extent(3), a.extent(4), a.extent(5)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3, const iType& i4, const iType& i5) const {
    a(i0, i1, i2, i3, i4, i5) = val;
  }
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill<ViewType, Layout, ExecSpace, 7, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<6, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_,
           const ExecSpace& space)
      : a(a_), val(val_) {
    // MDRangePolicy is not supported for 7D views
    // Iterate separately over extent(2)
    Kokkos::parallel_for("Kokkos::ViewFill-7D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(3),
                                      a.extent(4), a.extent(5), a.extent(6)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i3,
                  const iType& i4, const iType& i5, const iType& i6) const {
    for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
      a(i0, i1, i2, i3, i4, i5, i6) = val;
  }
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill<ViewType, Layout, ExecSpace, 8, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<6, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_,
           const ExecSpace& space)
      : a(a_), val(val_) {
    // MDRangePolicy is not supported for 8D views
    // Iterate separately over extent(2) and extent(4)
    Kokkos::parallel_for("Kokkos::ViewFill-8D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(3),
                                      a.extent(5), a.extent(6), a.extent(7)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i3,
                  const iType& i5, const iType& i6, const iType& i7) const {
    for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
      for (iType i4 = 0; i4 < iType(a.extent(4)); i4++)
        a(i0, i1, i2, i3, i4, i5, i6, i7) = val;
  }
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy<ViewTypeA, ViewTypeB, Layout, ExecSpace, 1, iType> {
  ViewTypeA a;
  ViewTypeB b;

  using policy_type = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;
  using value_type  = typename ViewTypeA::value_type;

  ViewCopy(const ViewTypeA& a_, const ViewTypeB& b_,
           const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy-1D",
                         policy_type(space, 0, a.extent(0)), *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0) const {
    a(i0) = static_cast<value_type>(b(i0));
  }
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy<ViewTypeA, ViewTypeB, Layout, ExecSpace, 2, iType> {
  ViewTypeA a;
  ViewTypeB b;
  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<2, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;
  using value_type = typename ViewTypeA::value_type;

  ViewCopy(const ViewTypeA& a_, const ViewTypeB& b_,
           const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy-2D",
                         policy_type(space, {0, 0}, {a.extent(0), a.extent(1)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1) const {
    a(i0, i1) = static_cast<value_type>(b(i0, i1));
  }
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy<ViewTypeA, ViewTypeB, Layout, ExecSpace, 3, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<3, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;
  using value_type = typename ViewTypeA::value_type;

  ViewCopy(const ViewTypeA& a_, const ViewTypeB& b_,
           const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for(
        "Kokkos::ViewCopy-3D",
        policy_type(space, {0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}),
        *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2) const {
    a(i0, i1, i2) = static_cast<value_type>(b(i0, i1, i2));
  }
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy<ViewTypeA, ViewTypeB, Layout, ExecSpace, 4, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<4, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewCopy(const ViewTypeA& a_, const ViewTypeB& b_,
           const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for(
        "Kokkos::ViewCopy-4D",
        policy_type(space, {0, 0, 0, 0},
                    {a.extent(0), a.extent(1), a.extent(2), a.extent(3)}),
        *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3) const {
    a(i0, i1, i2, i3) = b(i0, i1, i2, i3);
  }
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy<ViewTypeA, ViewTypeB, Layout, ExecSpace, 5, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<5, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewCopy(const ViewTypeA& a_, const ViewTypeB& b_,
           const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy-5D",
                         policy_type(space, {0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(2),
                                      a.extent(3), a.extent(4)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3, const iType& i4) const {
    a(i0, i1, i2, i3, i4) = b(i0, i1, i2, i3, i4);
  }
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy<ViewTypeA, ViewTypeB, Layout, ExecSpace, 6, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<6, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewCopy(const ViewTypeA& a_, const ViewTypeB& b_,
           const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy-6D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(2),
                                      a.extent(3), a.extent(4), a.extent(5)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3, const iType& i4, const iType& i5) const {
    a(i0, i1, i2, i3, i4, i5) = b(i0, i1, i2, i3, i4, i5);
  }
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy<ViewTypeA, ViewTypeB, Layout, ExecSpace, 7, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<6, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewCopy(const ViewTypeA& a_, const ViewTypeB& b_,
           const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    // MDRangePolicy is not supported for 7D views
    // Iterate separately over extent(2)
    Kokkos::parallel_for("Kokkos::ViewCopy-7D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(3),
                                      a.extent(4), a.extent(5), a.extent(6)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i3,
                  const iType& i4, const iType& i5, const iType& i6) const {
    for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
      a(i0, i1, i2, i3, i4, i5, i6) = b(i0, i1, i2, i3, i4, i5, i6);
  }
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy<ViewTypeA, ViewTypeB, Layout, ExecSpace, 8, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<6, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewCopy(const ViewTypeA& a_, const ViewTypeB& b_,
           const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    // MDRangePolicy is not supported for 8D views
    // Iterate separately over extent(2) and extent(4)
    Kokkos::parallel_for("Kokkos::ViewCopy-8D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(3),
                                      a.extent(5), a.extent(6), a.extent(7)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i3,
                  const iType& i5, const iType& i6, const iType& i7) const {
    for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
      for (iType i4 = 0; i4 < iType(a.extent(4)); i4++)
        a(i0, i1, i2, i3, i4, i5, i6, i7) = b(i0, i1, i2, i3, i4, i5, i6, i7);
  }
};

}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <class DstType>
Kokkos::Iterate get_iteration_order(const DstType& dst) {
  int64_t strides[DstType::rank + 1];
  dst.stride(strides);
  Kokkos::Iterate iterate;
  if (std::is_same_v<typename DstType::array_layout, Kokkos::LayoutRight>) {
    iterate = Kokkos::Iterate::Right;
  } else if (std::is_same_v<typename DstType::array_layout,
                            Kokkos::LayoutLeft>) {
    iterate = Kokkos::Iterate::Left;
  } else if (std::is_same_v<typename DstType::array_layout,
                            Kokkos::LayoutStride>) {
    if (strides[0] > strides[DstType::rank - 1])
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  } else {
    if (std::is_same_v<typename DstType::execution_space::array_layout,
                       Kokkos::LayoutRight>)
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  }
  return iterate;
}

template <class ExecutionSpace, class DstType, class SrcType>
void view_copy(const ExecutionSpace& space, const DstType& dst,
               const SrcType& src) {
  using dst_memory_space = typename DstType::memory_space;
  using src_memory_space = typename SrcType::memory_space;

  constexpr bool ExecCanAccessSrc =
      Kokkos::SpaceAccessibility<ExecutionSpace, src_memory_space>::accessible;
  constexpr bool ExecCanAccessDst =
      Kokkos::SpaceAccessibility<ExecutionSpace, dst_memory_space>::accessible;

  if (!(ExecCanAccessSrc && ExecCanAccessDst)) {
    Kokkos::Impl::throw_runtime_exception(
        "Kokkos::Impl::view_copy called with invalid execution space");
  } else {
    // Figure out iteration order in case we need it
    Kokkos::Iterate iterate = get_iteration_order(dst);

    if ((dst.span() >= size_t(std::numeric_limits<int>::max())) ||
        (src.span() >= size_t(std::numeric_limits<int>::max()))) {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy<
            typename DstType::uniform_runtime_nomemspace_type,
            typename SrcType::uniform_runtime_const_nomemspace_type,
            Kokkos::LayoutRight, ExecutionSpace, DstType::rank, int64_t>(
            dst, src, space);
      else
        Kokkos::Impl::ViewCopy<
            typename DstType::uniform_runtime_nomemspace_type,
            typename SrcType::uniform_runtime_const_nomemspace_type,
            Kokkos::LayoutLeft, ExecutionSpace, DstType::rank, int64_t>(
            dst, src, space);
    } else {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy<
            typename DstType::uniform_runtime_nomemspace_type,
            typename SrcType::uniform_runtime_const_nomemspace_type,
            Kokkos::LayoutRight, ExecutionSpace, DstType::rank, int>(dst, src,
                                                                     space);
      else
        Kokkos::Impl::ViewCopy<
            typename DstType::uniform_runtime_nomemspace_type,
            typename SrcType::uniform_runtime_const_nomemspace_type,
            Kokkos::LayoutLeft, ExecutionSpace, DstType::rank, int>(dst, src,
                                                                    space);
    }
  }
}

template <class DstType, class SrcType>
void view_copy(const DstType& dst, const SrcType& src) {
  using dst_execution_space = typename DstType::execution_space;
  using src_execution_space = typename SrcType::execution_space;
  using dst_memory_space    = typename DstType::memory_space;
  using src_memory_space    = typename SrcType::memory_space;

  constexpr bool DstExecCanAccessSrc =
      Kokkos::SpaceAccessibility<dst_execution_space,
                                 src_memory_space>::accessible;

  constexpr bool SrcExecCanAccessDst =
      Kokkos::SpaceAccessibility<src_execution_space,
                                 dst_memory_space>::accessible;

  if (!DstExecCanAccessSrc && !SrcExecCanAccessDst) {
    std::ostringstream ss;
    ss << "Error: Kokkos::deep_copy with no available copy mechanism: "
       << "from source view (\"" << src.label() << "\") to destination view (\""
       << dst.label() << "\").\n"
       << "There is no common execution space that can access both source's "
          "space\n"
       << "(" << src_memory_space().name() << ") and destination's space ("
       << dst_memory_space().name() << "), "
       << "so source and destination\n"
       << "must be contiguous and have the same layout.\n";
    Kokkos::Impl::throw_runtime_exception(ss.str());
  }

  using ExecutionSpace =
      std::conditional_t<DstExecCanAccessSrc, dst_execution_space,
                         src_execution_space>;

  // Figure out iteration order in case we need it
  Kokkos::Iterate iterate = get_iteration_order(dst);

  if ((dst.span() >= size_t(std::numeric_limits<int>::max())) ||
      (src.span() >= size_t(std::numeric_limits<int>::max()))) {
    if (iterate == Kokkos::Iterate::Right)
      Kokkos::Impl::ViewCopy<
          typename DstType::uniform_runtime_nomemspace_type,
          typename SrcType::uniform_runtime_const_nomemspace_type,
          Kokkos::LayoutRight, ExecutionSpace, DstType::rank, int64_t>(dst,
                                                                       src);
    else
      Kokkos::Impl::ViewCopy<
          typename DstType::uniform_runtime_nomemspace_type,
          typename SrcType::uniform_runtime_const_nomemspace_type,
          Kokkos::LayoutLeft, ExecutionSpace, DstType::rank, int64_t>(dst, src);
  } else {
    if (iterate == Kokkos::Iterate::Right)
      Kokkos::Impl::ViewCopy<
          typename DstType::uniform_runtime_nomemspace_type,
          typename SrcType::uniform_runtime_const_nomemspace_type,
          Kokkos::LayoutRight, ExecutionSpace, DstType::rank, int>(dst, src);
    else
      Kokkos::Impl::ViewCopy<
          typename DstType::uniform_runtime_nomemspace_type,
          typename SrcType::uniform_runtime_const_nomemspace_type,
          Kokkos::LayoutLeft, ExecutionSpace, DstType::rank, int>(dst, src);
  }
}

template <class DstType, class SrcType, class... Args>
struct CommonSubview {
  using dst_subview_type = typename Kokkos::Subview<DstType, Args...>;
  using src_subview_type = typename Kokkos::Subview<SrcType, Args...>;
  dst_subview_type dst_sub;
  src_subview_type src_sub;
  CommonSubview(const DstType& dst, const SrcType& src, const Args&... args)
      : dst_sub(dst, args...), src_sub(src, args...) {}
};

template <class DstType, class SrcType, int Rank = DstType::rank>
struct ViewRemap;

template <class DstType, class SrcType>
struct ViewRemap<DstType, SrcType, 1> {
  using p_type = Kokkos::pair<int64_t, int64_t>;

  template <typename... OptExecSpace>
  ViewRemap(const DstType& dst, const SrcType& src,
            const OptExecSpace&... exec_space) {
    static_assert(
        sizeof...(OptExecSpace) <= 1,
        "OptExecSpace must be either empty or be an execution space!");

    if (dst.extent(0) == src.extent(0)) {
      view_copy(exec_space..., dst, src);
    } else {
      p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
      CommonSubview common_subview(dst, src, ext0);
      view_copy(exec_space..., common_subview.dst_sub, common_subview.src_sub);
    }
  }
};

template <class DstType, class SrcType, std::size_t... I>
auto create_common_subview_first_and_last_match(const DstType& dst,
                                                const SrcType& src,
                                                std::index_sequence<I...>) {
  using p_type = Kokkos::pair<int64_t, int64_t>;
  CommonSubview common_subview(
      dst, src, Kokkos::ALL,
      (p_type(0, std::min(dst.extent(I + 1), src.extent(I + 1))))...,
      Kokkos::ALL);
  return common_subview;
}

template <class DstType, class SrcType, std::size_t... I>
auto create_common_subview_first_match(const DstType& dst, const SrcType& src,
                                       std::index_sequence<I...>) {
  using p_type = Kokkos::pair<int64_t, int64_t>;
  CommonSubview common_subview(
      dst, src, Kokkos::ALL,
      (p_type(0, std::min(dst.extent(I + 1), src.extent(I + 1))))...);
  return common_subview;
}

template <class DstType, class SrcType, std::size_t... I>
auto create_common_subview_last_match(const DstType& dst, const SrcType& src,
                                      std::index_sequence<I...>) {
  using p_type = Kokkos::pair<int64_t, int64_t>;
  CommonSubview common_subview(
      dst, src, (p_type(0, std::min(dst.extent(I), src.extent(I))))...,
      Kokkos::ALL);
  return common_subview;
}

template <class DstType, class SrcType, std::size_t... I>
auto create_common_subview_no_match(const DstType& dst, const SrcType& src,
                                    std::index_sequence<I...>) {
  using p_type = Kokkos::pair<int64_t, int64_t>;
  CommonSubview common_subview(
      dst, src, (p_type(0, std::min(dst.extent(I), src.extent(I))))...);
  return common_subview;
}

template <class DstType, class SrcType, int Rank>
struct ViewRemap {
  using p_type = Kokkos::pair<int64_t, int64_t>;

  template <typename... OptExecSpace>
  ViewRemap(const DstType& dst, const SrcType& src,
            const OptExecSpace&... exec_space) {
    static_assert(
        sizeof...(OptExecSpace) <= 1,
        "OptExecSpace must be either empty or be an execution space!");

    if (dst.extent(0) == src.extent(0)) {
      if (dst.extent(Rank - 1) == src.extent(Rank - 1)) {
        if constexpr (Rank < 3)
          view_copy(exec_space..., dst, src);
        else {
          auto common_subview = create_common_subview_first_and_last_match(
              dst, src, std::make_index_sequence<Rank - 2>{});
          view_copy(exec_space..., common_subview.dst_sub,
                    common_subview.src_sub);
        }
      } else {
        auto common_subview = create_common_subview_first_match(
            dst, src, std::make_index_sequence<Rank - 1>{});
        view_copy(exec_space..., common_subview.dst_sub,
                  common_subview.src_sub);
      }
    } else {
      if (dst.extent(Rank - 1) == src.extent(Rank - 1)) {
        auto common_subview = create_common_subview_last_match(
            dst, src, std::make_index_sequence<Rank - 1>{});
        view_copy(exec_space..., common_subview.dst_sub,
                  common_subview.src_sub);
      } else {
        auto common_subview = create_common_subview_no_match(
            dst, src, std::make_index_sequence<Rank>{});
        view_copy(exec_space..., common_subview.dst_sub,
                  common_subview.src_sub);
      }
    }
  }
};

template <typename ExecutionSpace, class DT, class... DP>
inline void contiguous_fill(
    const ExecutionSpace& exec_space, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value) {
  using ViewType     = View<DT, DP...>;
  using ViewTypeFlat = Kokkos::View<
      typename ViewType::value_type*, Kokkos::LayoutRight,
      Kokkos::Device<typename ViewType::execution_space,
                     std::conditional_t<ViewType::rank == 0,
                                        typename ViewType::memory_space,
                                        Kokkos::AnonymousSpace>>,
      Kokkos::MemoryTraits<>>;

  ViewTypeFlat dst_flat(dst.data(), dst.size());
  if (dst.span() < static_cast<size_t>(std::numeric_limits<int>::max())) {
    Kokkos::Impl::ViewFill<ViewTypeFlat, Kokkos::LayoutRight, ExecutionSpace,
                           ViewTypeFlat::rank, int>(dst_flat, value,
                                                    exec_space);
  } else
    Kokkos::Impl::ViewFill<ViewTypeFlat, Kokkos::LayoutRight, ExecutionSpace,
                           ViewTypeFlat::rank, int64_t>(dst_flat, value,
                                                        exec_space);
}

// Default implementation for execution spaces that don't provide a definition
template <typename ExecutionSpace>
struct ZeroMemset {
  ZeroMemset(const ExecutionSpace& exec_space, void* dst, size_t cnt) {
    contiguous_fill(
        exec_space,
        Kokkos::View<std::byte*, ExecutionSpace, Kokkos::MemoryUnmanaged>(
            static_cast<std::byte*>(dst), cnt),
        std::byte{});
  }
};

// Returns true when we can safely determine that the object has all 0 bits,
// false otherwise.  It is intended to determine whether to perform zero memset
// as an optimization.
template <typename T>
bool has_all_zero_bits(const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);

  if constexpr (std::is_scalar_v<T>) {
    return value == T();
  }

  KOKKOS_IMPL_DISABLE_UNREACHABLE_WARNINGS_PUSH()
  if constexpr (std::is_standard_layout_v<T> &&
                std::has_unique_object_representations_v<T>) {
    constexpr std::byte all_zeroes[sizeof(T)] = {};
    return std::memcmp(&value, all_zeroes, sizeof(T)) == 0;
  }

  return false;
  KOKKOS_IMPL_DISABLE_UNREACHABLE_WARNINGS_POP()
}

template <typename ExecutionSpace, class DT, class... DP>
inline std::enable_if_t<
    std::is_trivially_copyable_v<typename ViewTraits<DT, DP...>::value_type> &&
    !ViewTraits<DT, DP...>::impl_is_customized>
contiguous_fill_or_memset(
    const ExecutionSpace& exec_space, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value) {
  // With OpenMP, using memset has significant performance issues.
  if (has_all_zero_bits(value)
#ifdef KOKKOS_ENABLE_OPENMP
      && !std::is_same_v<ExecutionSpace, Kokkos::OpenMP>
#endif
  )
    ZeroMemset(exec_space, dst.data(),
               dst.size() * sizeof(typename ViewTraits<DT, DP...>::value_type));
  else
    contiguous_fill(exec_space, dst, value);
}

template <typename ExecutionSpace, class DT, class... DP>
inline std::enable_if_t<
    !std::is_trivially_copyable_v<typename ViewTraits<DT, DP...>::value_type> ||
    ViewTraits<DT, DP...>::impl_is_customized>
contiguous_fill_or_memset(
    const ExecutionSpace& exec_space, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value) {
  contiguous_fill(exec_space, dst, value);
}

template <class DT, class... DP>
void contiguous_fill_or_memset(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value) {
  using ViewType        = View<DT, DP...>;
  using exec_space_type = typename ViewType::execution_space;

  contiguous_fill_or_memset(exec_space_type(), dst, value);
}
}  // namespace Impl

/** \brief  Deep copy a value from Host memory into a view.  */
template <class DT, class... DP>
inline void deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<std::is_same_v<typename ViewTraits<DT, DP...>::specialize,
                                    void>>* = nullptr) {
  using ViewType        = View<DT, DP...>;
  using exec_space_type = typename ViewType::execution_space;

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(ViewType::memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(Kokkos::HostSpace::name()),
        "Scalar", &value, dst.span() * sizeof(typename ViewType::value_type));
  }

  if (dst.data() == nullptr) {
    Kokkos::fence(
        "Kokkos::deep_copy: scalar copy, fence because destination is null");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  Kokkos::fence("Kokkos::deep_copy: scalar copy, pre copy fence");
  static_assert(std::is_same_v<typename ViewType::non_const_value_type,
                               typename ViewType::value_type>,
                "deep_copy requires non-const type");

  // If contiguous we can simply do a 1D flat loop or use memset
  // Do not use shortcut if there is a custom accessor
  if (dst.span_is_contiguous() && !ViewType::traits::impl_is_customized) {
    Impl::contiguous_fill_or_memset(dst, value);
    Kokkos::fence("Kokkos::deep_copy: scalar copy, post copy fence");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  // Figure out iteration order to do the ViewFill
  int64_t strides[ViewType::rank + 1];
  dst.stride(strides);
  Kokkos::Iterate iterate;
  if (std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutRight>) {
    iterate = Kokkos::Iterate::Right;
  } else if (std::is_same_v<typename ViewType::array_layout,
                            Kokkos::LayoutLeft>) {
    iterate = Kokkos::Iterate::Left;
  } else if (std::is_same_v<typename ViewType::array_layout,
                            Kokkos::LayoutStride>) {
    if (strides[0] > strides[ViewType::rank > 0 ? ViewType::rank - 1 : 0])
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  } else {
    if (std::is_same_v<typename ViewType::execution_space::array_layout,
                       Kokkos::LayoutRight>)
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  }

  // Lets call the right ViewFill functor based on integer space needed and
  // iteration type
  using ViewTypeUniform =
      std::conditional_t<ViewType::rank == 0,
                         typename ViewType::uniform_runtime_type,
                         typename ViewType::uniform_runtime_nomemspace_type>;
  if (dst.span() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    if (iterate == Kokkos::Iterate::Right)
      Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutRight,
                             exec_space_type, ViewType::rank, int64_t>(
          dst, value, exec_space_type());
    else
      Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutLeft,
                             exec_space_type, ViewType::rank, int64_t>(
          dst, value, exec_space_type());
  } else {
    if (iterate == Kokkos::Iterate::Right)
      Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutRight,
                             exec_space_type, ViewType::rank, int>(
          dst, value, exec_space_type());
    else
      Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutLeft,
                             exec_space_type, ViewType::rank, int>(
          dst, value, exec_space_type());
  }
  Kokkos::fence("Kokkos::deep_copy: scalar copy, post copy fence");

  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

/** \brief  Deep copy into a value in Host memory from a view.  */
template <class ST, class... SP>
inline void deep_copy(
    typename ViewTraits<ST, SP...>::non_const_value_type& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<std::is_same_v<typename ViewTraits<ST, SP...>::specialize,
                                    void>>* = nullptr) {
  using src_traits       = ViewTraits<ST, SP...>;
  using src_memory_space = typename src_traits::memory_space;

  static_assert(src_traits::rank == 0,
                "ERROR: Non-rank-zero view in deep_copy( value , View )");

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(Kokkos::HostSpace::name()),
        "Scalar", &dst,
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(),
        src.span() * sizeof(typename src_traits::value_type));
  }

  if (src.data() == nullptr) {
    Kokkos::fence("Kokkos::deep_copy: copy into scalar, src is null");
  } else {
    Kokkos::fence("Kokkos::deep_copy: copy into scalar, pre copy fence");
    Kokkos::Impl::DeepCopy<HostSpace, src_memory_space>(&dst, src.data(),
                                                        sizeof(ST));
    Kokkos::fence("Kokkos::deep_copy: copy into scalar, post copy fence");
  }

  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible type, and rank zero.  */
template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<
        (std::is_void_v<typename ViewTraits<DT, DP...>::specialize> &&
         std::is_void_v<typename ViewTraits<ST, SP...>::specialize> &&
         (unsigned(ViewTraits<DT, DP...>::rank) == unsigned(0) &&
          unsigned(ViewTraits<ST, SP...>::rank) == unsigned(0)))>* = nullptr) {
  using dst_type = View<DT, DP...>;
  using src_type = View<ST, SP...>;

  using value_type       = typename dst_type::value_type;
  using dst_memory_space = typename dst_type::memory_space;
  using src_memory_space = typename src_type::memory_space;

  static_assert(std::is_same_v<typename dst_type::value_type,
                               typename src_type::non_const_value_type>,
                "deep_copy requires matching non-const destination type");

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(),
        src.span() * sizeof(typename dst_type::value_type));
  }

  if (dst.data() == nullptr && src.data() == nullptr) {
    Kokkos::fence(
        "Kokkos::deep_copy: scalar to scalar copy, both pointers null");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  Kokkos::fence("Kokkos::deep_copy: scalar to scalar copy, pre copy fence");
  if (dst.data() != src.data()) {
    Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
        dst.data(), src.data(), sizeof(value_type));
    Kokkos::fence("Kokkos::deep_copy: scalar to scalar copy, post copy fence");
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of the default specialization, compatible
 * type, same non-zero rank, same contiguous layout.
 */
template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<
        (std::is_void_v<typename ViewTraits<DT, DP...>::specialize> &&
         std::is_void_v<typename ViewTraits<ST, SP...>::specialize> &&
         (unsigned(ViewTraits<DT, DP...>::rank) != 0 ||
          unsigned(ViewTraits<ST, SP...>::rank) != 0))>* = nullptr) {
  using dst_type         = View<DT, DP...>;
  using src_type         = View<ST, SP...>;
  using dst_memory_space = typename dst_type::memory_space;
  using src_memory_space = typename src_type::memory_space;
  using dst_ptr_type     = decltype(dst.data());
  using src_ptr_type     = decltype(src.data());

  static_assert(std::is_same_v<typename dst_type::value_type,
                               typename dst_type::non_const_value_type>,
                "deep_copy requires non-const destination type");

  static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                "deep_copy requires Views of equal rank");

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(),
        src.span() * sizeof(typename dst_type::value_type));
  }

  if (dst.data() == nullptr || src.data() == nullptr) {
    // throw if dimension mismatch
    if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
        (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
        (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
        (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
      std::string message(
          "Deprecation Error: Kokkos::deep_copy extents of views don't "
          "match: ");
      message += dst.label();
      message += "(";
      message += std::to_string(dst.extent(0));
      for (size_t r = 1; r < dst_type::rank; r++) {
        message += ",";
        message += std::to_string(dst.extent(r));
      }
      message += ") ";
      message += src.label();
      message += "(";
      message += std::to_string(src.extent(0));
      for (size_t r = 1; r < src_type::rank; r++) {
        message += ",";
        message += std::to_string(src.extent(r));
      }
      message += ") ";

      Kokkos::Impl::throw_runtime_exception(message);
    }
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, fence due to null "
        "argument");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  // Checking for Overlapping Views.
  dst_ptr_type dst_start = dst.data();
  src_ptr_type src_start = src.data();
#ifndef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
  dst_ptr_type dst_end = dst.data() + allocation_size_from_mapping_and_accessor(
                                          dst.mapping(), dst.accessor());
  src_ptr_type src_end = src.data() + allocation_size_from_mapping_and_accessor(
                                          src.mapping(), src.accessor());
#else
  dst_ptr_type dst_end = dst.data() + dst.span();
  src_ptr_type src_end = src.data() + src.span();
#endif
  if (((std::ptrdiff_t)dst_start == (std::ptrdiff_t)src_start) &&
      ((std::ptrdiff_t)dst_end == (std::ptrdiff_t)src_end) &&
      (dst.span_is_contiguous() && src.span_is_contiguous())) {
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, fence due to same "
        "spans");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  if ((((std::ptrdiff_t)dst_start < (std::ptrdiff_t)src_end) &&
       ((std::ptrdiff_t)dst_end > (std::ptrdiff_t)src_start)) &&
      ((dst.span_is_contiguous() && src.span_is_contiguous()))) {
    std::string message("Error: Kokkos::deep_copy of overlapping views: ");
    message += dst.label();
    message += "(";
    message += std::to_string((std::ptrdiff_t)dst_start);
    message += ",";
    message += std::to_string((std::ptrdiff_t)dst_end);
    message += ") ";
    message += src.label();
    message += "(";
    message += std::to_string((std::ptrdiff_t)src_start);
    message += ",";
    message += std::to_string((std::ptrdiff_t)src_end);
    message += ") ";
    Kokkos::Impl::throw_runtime_exception(message);
  }

  // Check for same extents
  if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
      (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
      (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
      (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
    std::string message(
        "Deprecation Error: Kokkos::deep_copy extents of views don't match: ");
    message += dst.label();
    message += "(";
    message += std::to_string(dst.extent(0));
    for (size_t r = 1; r < dst_type::rank; r++) {
      message += ",";
      message += std::to_string(dst.extent(r));
    }
    message += ") ";
    message += src.label();
    message += "(";
    message += std::to_string(src.extent(0));
    for (size_t r = 1; r < src_type::rank; r++) {
      message += ",";
      message += std::to_string(src.extent(r));
    }
    message += ") ";

    Kokkos::Impl::throw_runtime_exception(message);
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous
  // memory then can byte-wise copy

  if (std::is_same_v<typename dst_type::value_type,
                     typename src_type::non_const_value_type> &&
      (std::is_same_v<typename dst_type::array_layout,
                      typename src_type::array_layout> ||
       (dst_type::rank == 1 && src_type::rank == 1)) &&
      dst.span_is_contiguous() && src.span_is_contiguous() &&
      ((dst_type::rank < 1) || (dst.stride_0() == src.stride_0())) &&
      ((dst_type::rank < 2) || (dst.stride_1() == src.stride_1())) &&
      ((dst_type::rank < 3) || (dst.stride_2() == src.stride_2())) &&
      ((dst_type::rank < 4) || (dst.stride_3() == src.stride_3())) &&
      ((dst_type::rank < 5) || (dst.stride_4() == src.stride_4())) &&
      ((dst_type::rank < 6) || (dst.stride_5() == src.stride_5())) &&
      ((dst_type::rank < 7) || (dst.stride_6() == src.stride_6())) &&
      ((dst_type::rank < 8) || (dst.stride_7() == src.stride_7()))) {
#ifndef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
    const size_t nbytes = allocation_size_from_mapping_and_accessor(
                              src.mapping(), src.accessor()) *
                          sizeof(std::remove_pointer_t<dst_ptr_type>);
#else
    const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();
#endif
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, pre view equality "
        "check");
    if ((void*)dst.data() != (void*)src.data() && 0 < nbytes) {
      Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
          dst.data(), src.data(), nbytes);
      Kokkos::fence(
          "Kokkos::deep_copy: copy between contiguous views, post deep copy "
          "fence");
    }
  } else {
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, pre copy fence");
    Impl::view_copy(dst, src);
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, post copy fence");
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
namespace Experimental {
/** \brief  A local deep copy between views of the default specialization,
 * compatible type, same non-zero rank.
 */
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION
local_deep_copy_contiguous(const TeamType& team, const View<DT, DP...>& dst,
                           const View<ST, SP...>& src) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, src.span()),
                       [&](const int& i) { dst.data()[i] = src.data()[i]; });
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy_contiguous(
    const View<DT, DP...>& dst, const View<ST, SP...>& src) {
  for (size_t i = 0; i < src.span(); ++i) {
    dst.data()[i] = src.data()[i];
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 1 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 1)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N),
                       [&](const int& i) { dst(i) = src(i); });
  team.team_barrier();
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 2 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 2)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0      = i % dst.extent(0);
      int i1      = i / dst.extent(0);
      dst(i0, i1) = src(i0, i1);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 3 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 3)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0          = i % dst.extent(0);
      int itmp        = i / dst.extent(0);
      int i1          = itmp % dst.extent(1);
      int i2          = itmp / dst.extent(1);
      dst(i0, i1, i2) = src(i0, i1, i2);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 4 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 4)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N =
      dst.extent(0) * dst.extent(1) * dst.extent(2) * dst.extent(3);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0              = i % dst.extent(0);
      int itmp            = i / dst.extent(0);
      int i1              = itmp % dst.extent(1);
      itmp                = itmp / dst.extent(1);
      int i2              = itmp % dst.extent(2);
      int i3              = itmp / dst.extent(2);
      dst(i0, i1, i2, i3) = src(i0, i1, i2, i3);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 5 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 5)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0                  = i % dst.extent(0);
      int itmp                = i / dst.extent(0);
      int i1                  = itmp % dst.extent(1);
      itmp                    = itmp / dst.extent(1);
      int i2                  = itmp % dst.extent(2);
      itmp                    = itmp / dst.extent(2);
      int i3                  = itmp % dst.extent(3);
      int i4                  = itmp / dst.extent(3);
      dst(i0, i1, i2, i3, i4) = src(i0, i1, i2, i3, i4);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 6 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 6)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0                      = i % dst.extent(0);
      int itmp                    = i / dst.extent(0);
      int i1                      = itmp % dst.extent(1);
      itmp                        = itmp / dst.extent(1);
      int i2                      = itmp % dst.extent(2);
      itmp                        = itmp / dst.extent(2);
      int i3                      = itmp % dst.extent(3);
      itmp                        = itmp / dst.extent(3);
      int i4                      = itmp % dst.extent(4);
      int i5                      = itmp / dst.extent(4);
      dst(i0, i1, i2, i3, i4, i5) = src(i0, i1, i2, i3, i4, i5);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 7 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 7)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5) *
                   dst.extent(6);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0                          = i % dst.extent(0);
      int itmp                        = i / dst.extent(0);
      int i1                          = itmp % dst.extent(1);
      itmp                            = itmp / dst.extent(1);
      int i2                          = itmp % dst.extent(2);
      itmp                            = itmp / dst.extent(2);
      int i3                          = itmp % dst.extent(3);
      itmp                            = itmp / dst.extent(3);
      int i4                          = itmp % dst.extent(4);
      itmp                            = itmp / dst.extent(4);
      int i5                          = itmp % dst.extent(5);
      int i6                          = itmp / dst.extent(5);
      dst(i0, i1, i2, i3, i4, i5, i6) = src(i0, i1, i2, i3, i4, i5, i6);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 1 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 1)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  for (size_t i = 0; i < N; ++i) {
    dst(i) = src(i);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 2 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 2)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1) dst(i0, i1) = src(i0, i1);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 3 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 3)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          dst(i0, i1, i2) = src(i0, i1, i2);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 4 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 4)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            dst(i0, i1, i2, i3) = src(i0, i1, i2, i3);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 5 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 5)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              dst(i0, i1, i2, i3, i4) = src(i0, i1, i2, i3, i4);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 6 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 6)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                dst(i0, i1, i2, i3, i4, i5) = src(i0, i1, i2, i3, i4, i5);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 7 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 7)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                for (size_t i6 = 0; i6 < dst.extent(6); ++i6)
                  dst(i0, i1, i2, i3, i4, i5, i6) =
                      src(i0, i1, i2, i3, i4, i5, i6);
  }
}
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/** \brief  Deep copy a value into a view.  */
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy_contiguous(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<std::is_same_v<typename ViewTraits<DT, DP...>::specialize,
                                    void>>* = nullptr) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, dst.span()),
                       [&](const int& i) { dst.data()[i] = value; });
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy_contiguous(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<std::is_same_v<typename ViewTraits<DT, DP...>::specialize,
                                    void>>* = nullptr) {
  for (size_t i = 0; i < dst.span(); ++i) {
    dst.data()[i] = value;
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 1)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N),
                       [&](const int& i) { dst(i) = value; });
  team.team_barrier();
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 2)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1);

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      team.team_barrier();
      local_deep_copy_contiguous(team, dst, value);
      team.team_barrier();
      return;
    }
  }
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
    int i0      = i % dst.extent(0);
    int i1      = i / dst.extent(0);
    dst(i0, i1) = value;
  });
  team.team_barrier();
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 3)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2);

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      team.team_barrier();
      local_deep_copy_contiguous(team, dst, value);
      team.team_barrier();
      return;
    }
  }
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
    int i0          = i % dst.extent(0);
    int itmp        = i / dst.extent(0);
    int i1          = itmp % dst.extent(1);
    int i2          = itmp / dst.extent(1);
    dst(i0, i1, i2) = value;
  });
  team.team_barrier();
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 4)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N =
      dst.extent(0) * dst.extent(1) * dst.extent(2) * dst.extent(3);

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      team.team_barrier();
      local_deep_copy_contiguous(team, dst, value);
      team.team_barrier();
      return;
    }
  }
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
    int i0              = i % dst.extent(0);
    int itmp            = i / dst.extent(0);
    int i1              = itmp % dst.extent(1);
    itmp                = itmp / dst.extent(1);
    int i2              = itmp % dst.extent(2);
    int i3              = itmp / dst.extent(2);
    dst(i0, i1, i2, i3) = value;
  });
  team.team_barrier();
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 5)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4);

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      team.team_barrier();
      local_deep_copy_contiguous(team, dst, value);
      team.team_barrier();
      return;
    }
  }
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
    int i0                  = i % dst.extent(0);
    int itmp                = i / dst.extent(0);
    int i1                  = itmp % dst.extent(1);
    itmp                    = itmp / dst.extent(1);
    int i2                  = itmp % dst.extent(2);
    itmp                    = itmp / dst.extent(2);
    int i3                  = itmp % dst.extent(3);
    int i4                  = itmp / dst.extent(3);
    dst(i0, i1, i2, i3, i4) = value;
  });
  team.team_barrier();
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 6)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5);

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      team.team_barrier();
      local_deep_copy_contiguous(team, dst, value);
      team.team_barrier();
      return;
    }
  }
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
    int i0                      = i % dst.extent(0);
    int itmp                    = i / dst.extent(0);
    int i1                      = itmp % dst.extent(1);
    itmp                        = itmp / dst.extent(1);
    int i2                      = itmp % dst.extent(2);
    itmp                        = itmp / dst.extent(2);
    int i3                      = itmp % dst.extent(3);
    itmp                        = itmp / dst.extent(3);
    int i4                      = itmp % dst.extent(4);
    int i5                      = itmp / dst.extent(4);
    dst(i0, i1, i2, i3, i4, i5) = value;
  });
  team.team_barrier();
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 7)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5) *
                   dst.extent(6);

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      team.team_barrier();
      local_deep_copy_contiguous(team, dst, value);
      team.team_barrier();
      return;
    }
  }
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
    int i0                          = i % dst.extent(0);
    int itmp                        = i / dst.extent(0);
    int i1                          = itmp % dst.extent(1);
    itmp                            = itmp / dst.extent(1);
    int i2                          = itmp % dst.extent(2);
    itmp                            = itmp / dst.extent(2);
    int i3                          = itmp % dst.extent(3);
    itmp                            = itmp / dst.extent(3);
    int i4                          = itmp % dst.extent(4);
    itmp                            = itmp / dst.extent(4);
    int i5                          = itmp % dst.extent(5);
    int i6                          = itmp / dst.extent(5);
    dst(i0, i1, i2, i3, i4, i5, i6) = value;
  });
  team.team_barrier();
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 1)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  for (size_t i = 0; i < N; ++i) {
    dst(i) = value;
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 2)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      local_deep_copy_contiguous(dst, value);
      return;
    }
  }
  for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
    for (size_t i1 = 0; i1 < dst.extent(1); ++i1) dst(i0, i1) = value;
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 3)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      local_deep_copy_contiguous(dst, value);
      return;
    }
  }
  for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
      for (size_t i2 = 0; i2 < dst.extent(2); ++i2) dst(i0, i1, i2) = value;
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 4)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      local_deep_copy_contiguous(dst, value);
      return;
    }
  }
  for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
      for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
        for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
          dst(i0, i1, i2, i3) = value;
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 5)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      local_deep_copy_contiguous(dst, value);
      return;
    }
  }
  for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
      for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
        for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
          for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
            dst(i0, i1, i2, i3, i4) = value;
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 6)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      local_deep_copy_contiguous(dst, value);
      return;
    }
  }
  for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
      for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
        for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
          for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
            for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
              dst(i0, i1, i2, i3, i4, i5) = value;
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 7)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    // FIXME We might want to check the traits for customization here but we
    // aren't aware of a use case where that is necessary.
    if constexpr (std::is_same_v<decltype(dst.data()),
                                 typename View<DT, DP...>::element_type*>) {
      local_deep_copy_contiguous(dst, value);
      return;
    }
  }
  for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
      for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
        for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
          for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
            for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
              for (size_t i6 = 0; i6 < dst.extent(6); ++i6)
                dst(i0, i1, i2, i3, i4, i5, i6) = value;
}
} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

/** \brief  Deep copy a value from Host memory into a view. ExecSpace can access
 * dst */
template <class ExecSpace, class DT, class... DP>
inline void deep_copy(
    const ExecSpace& space, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<
        Kokkos::is_execution_space<ExecSpace>::value &&
        std::is_void_v<typename ViewTraits<DT, DP...>::specialize> &&
        Kokkos::SpaceAccessibility<ExecSpace, typename ViewTraits<DT, DP...>::
                                                  memory_space>::accessible>* =
        nullptr) {
  using dst_traits = ViewTraits<DT, DP...>;
  static_assert(std::is_same_v<typename dst_traits::non_const_value_type,
                               typename dst_traits::value_type>,
                "deep_copy requires non-const type");
  using dst_memory_space = typename dst_traits::memory_space;
  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(Kokkos::HostSpace::name()),
        "(none)", &value, dst.span() * sizeof(typename dst_traits::value_type));
  }
  if (dst.data() == nullptr) {
    space.fence("Kokkos::deep_copy: scalar copy on space, dst data is null");
  } else if (dst.span_is_contiguous() &&
             !ViewTraits<DT, DP...>::impl_is_customized) {
    Impl::contiguous_fill_or_memset(space, dst, value);
  } else {
    using ViewType = View<DT, DP...>;
    // Figure out iteration order to do the ViewFill
    int64_t strides[ViewType::rank + 1];
    dst.stride(strides);
    Kokkos::Iterate iterate;
    if (std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutRight>) {
      iterate = Kokkos::Iterate::Right;
    } else if (std::is_same_v<typename ViewType::array_layout,
                              Kokkos::LayoutLeft>) {
      iterate = Kokkos::Iterate::Left;
    } else if (std::is_same_v<typename ViewType::array_layout,
                              Kokkos::LayoutStride>) {
      if (strides[0] > strides[ViewType::rank > 0 ? ViewType::rank - 1 : 0])
        iterate = Kokkos::Iterate::Right;
      else
        iterate = Kokkos::Iterate::Left;
    } else {
      if (std::is_same_v<typename ViewType::execution_space::array_layout,
                         Kokkos::LayoutRight>)
        iterate = Kokkos::Iterate::Right;
      else
        iterate = Kokkos::Iterate::Left;
    }

    // Lets call the right ViewFill functor based on integer space needed and
    // iteration type
    using ViewTypeUniform =
        std::conditional_t<ViewType::rank == 0,
                           typename ViewType::uniform_runtime_type,
                           typename ViewType::uniform_runtime_nomemspace_type>;
    if (dst.span() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutRight, ExecSpace,
                               ViewType::rank, int64_t>(dst, value, space);
      else
        Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutLeft, ExecSpace,
                               ViewType::rank, int64_t>(dst, value, space);
    } else {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutRight, ExecSpace,
                               ViewType::rank, int32_t>(dst, value, space);
      else
        Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutLeft, ExecSpace,
                               ViewType::rank, int32_t>(dst, value, space);
    }
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

/** \brief  Deep copy a value from Host memory into a view. ExecSpace can not
 * access dst */
template <class ExecSpace, class DT, class... DP>
inline void deep_copy(
    const ExecSpace& space, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<
        Kokkos::is_execution_space<ExecSpace>::value &&
        std::is_void_v<typename ViewTraits<DT, DP...>::specialize> &&
        !Kokkos::SpaceAccessibility<ExecSpace, typename ViewTraits<DT, DP...>::
                                                   memory_space>::accessible>* =
        nullptr) {
  using dst_traits = ViewTraits<DT, DP...>;
  static_assert(std::is_same_v<typename dst_traits::non_const_value_type,
                               typename dst_traits::value_type>,
                "deep_copy requires non-const type");
  using dst_memory_space = typename dst_traits::memory_space;
  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(Kokkos::HostSpace::name()),
        "(none)", &value, dst.span() * sizeof(typename dst_traits::value_type));
  }
  if (dst.data() == nullptr) {
    space.fence(
        "Kokkos::deep_copy: scalar-to-view copy on space, dst data is null");
  } else {
    space.fence("Kokkos::deep_copy: scalar-to-view copy on space, pre copy");
    using fill_exec_space = typename dst_traits::memory_space::execution_space;
    if (dst.span_is_contiguous() &&
        !ViewTraits<DT, DP...>::impl_is_customized) {
      Impl::contiguous_fill_or_memset(fill_exec_space(), dst, value);
    } else {
      using ViewTypeUniform = std::conditional_t<
          View<DT, DP...>::rank == 0,
          typename View<DT, DP...>::uniform_runtime_type,
          typename View<DT, DP...>::uniform_runtime_nomemspace_type>;
      Kokkos::Impl::ViewFill<ViewTypeUniform, typename dst_traits::array_layout,
                             fill_exec_space>(dst, value, fill_exec_space());
    }
    fill_exec_space().fence(
        "Kokkos::deep_copy: scalar-to-view copy on space, fence after fill");
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

/** \brief  Deep copy into a value in Host memory from a view.  */
template <class ExecSpace, class ST, class... SP>
inline void deep_copy(
    const ExecSpace& exec_space,
    typename ViewTraits<ST, SP...>::non_const_value_type& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<Kokkos::is_execution_space<ExecSpace>::value &&
                     std::is_same_v<typename ViewTraits<ST, SP...>::specialize,
                                    void>>* = nullptr) {
  using src_traits       = ViewTraits<ST, SP...>;
  using src_memory_space = typename src_traits::memory_space;
  static_assert(src_traits::rank == 0,
                "ERROR: Non-rank-zero view in deep_copy( value , View )");
  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(Kokkos::HostSpace::name()),
        "(none)", &dst,
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(), sizeof(ST));
  }

  if (src.data() == nullptr) {
    exec_space.fence(
        "Kokkos::deep_copy: view-to-scalar copy on space, src data is null");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  Kokkos::Impl::DeepCopy<HostSpace, src_memory_space, ExecSpace>(
      exec_space, &dst, src.data(), sizeof(ST));
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible type, and rank zero.  */
template <class ExecSpace, class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const ExecSpace& exec_space, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<
        (Kokkos::is_execution_space<ExecSpace>::value &&
         std::is_void_v<typename ViewTraits<DT, DP...>::specialize> &&
         std::is_void_v<typename ViewTraits<ST, SP...>::specialize> &&
         (unsigned(ViewTraits<DT, DP...>::rank) == unsigned(0) &&
          unsigned(ViewTraits<ST, SP...>::rank) == unsigned(0)))>* = nullptr) {
  using src_traits = ViewTraits<ST, SP...>;
  using dst_traits = ViewTraits<DT, DP...>;

  using src_memory_space = typename src_traits::memory_space;
  using dst_memory_space = typename dst_traits::memory_space;
  static_assert(std::is_same_v<typename dst_traits::value_type,
                               typename src_traits::non_const_value_type>,
                "deep_copy requires matching non-const destination type");

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(), sizeof(DT));
  }

  if (dst.data() == nullptr && src.data() == nullptr) {
    exec_space.fence(
        "Kokkos::deep_copy: view-to-view copy on space, data is null");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  if (dst.data() != src.data()) {
    Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space, ExecSpace>(
        exec_space, dst.data(), src.data(),
        sizeof(typename dst_traits::value_type));
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of the default specialization, compatible
 * type, same non-zero rank
 */
template <class ExecSpace, class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const ExecSpace& exec_space, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<
        (Kokkos::is_execution_space<ExecSpace>::value &&
         std::is_void_v<typename ViewTraits<DT, DP...>::specialize> &&
         std::is_void_v<typename ViewTraits<ST, SP...>::specialize> &&
         (unsigned(ViewTraits<DT, DP...>::rank) != 0 ||
          unsigned(ViewTraits<ST, SP...>::rank) != 0))>* = nullptr) {
  using dst_type = View<DT, DP...>;
  using src_type = View<ST, SP...>;

  static_assert(std::is_same_v<typename dst_type::value_type,
                               typename dst_type::non_const_value_type>,
                "deep_copy requires non-const destination type");

  static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                "deep_copy requires Views of equal rank");

  using dst_execution_space = typename dst_type::execution_space;
  using src_execution_space = typename src_type::execution_space;
  using dst_memory_space    = typename dst_type::memory_space;
  using src_memory_space    = typename src_type::memory_space;
  using dst_value_type      = typename dst_type::value_type;
  using src_value_type      = typename src_type::value_type;

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(), dst.span() * sizeof(dst_value_type));
  }

  dst_value_type* dst_start = dst.data();
  dst_value_type* dst_end   = dst.data() + dst.span();
  src_value_type* src_start = src.data();
  src_value_type* src_end   = src.data() + src.span();

  // Early dropout if identical range
  if ((dst_start == nullptr || src_start == nullptr) ||
      ((std::ptrdiff_t(dst_start) == std::ptrdiff_t(src_start)) &&
       (std::ptrdiff_t(dst_end) == std::ptrdiff_t(src_end)))) {
    // throw if dimension mismatch
    if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
        (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
        (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
        (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
      std::string message(
          "Deprecation Error: Kokkos::deep_copy extents of views don't "
          "match: ");
      message += dst.label();
      message += "(";
      message += std::to_string(dst.extent(0));
      for (size_t r = 1; r < dst_type::rank; r++) {
        message += ",";
        message += std::to_string(dst.extent(r));
      }
      message += ") ";
      message += src.label();
      message += "(";
      message += std::to_string(src.extent(0));
      for (size_t r = 1; r < src_type::rank; r++) {
        message += ",";
        message += std::to_string(src.extent(r));
      }
      message += ") ";

      Kokkos::Impl::throw_runtime_exception(message);
    }
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  // Error out for non-identical overlapping views.
  if ((((std::ptrdiff_t)dst_start < (std::ptrdiff_t)src_end) &&
       ((std::ptrdiff_t)dst_end > (std::ptrdiff_t)src_start)) &&
      ((dst.span_is_contiguous() && src.span_is_contiguous()))) {
    std::string message("Error: Kokkos::deep_copy of overlapping views: ");
    message += dst.label();
    message += "(";
    message += std::to_string((std::ptrdiff_t)dst_start);
    message += ",";
    message += std::to_string((std::ptrdiff_t)dst_end);
    message += ") ";
    message += src.label();
    message += "(";
    message += std::to_string((std::ptrdiff_t)src_start);
    message += ",";
    message += std::to_string((std::ptrdiff_t)src_end);
    message += ") ";
    Kokkos::Impl::throw_runtime_exception(message);
  }

  // Check for same extents
  if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
      (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
      (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
      (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
    std::string message(
        "Deprecation Error: Kokkos::deep_copy extents of views don't match: ");
    message += dst.label();
    message += "(";
    message += std::to_string(dst.extent(0));
    for (size_t r = 1; r < dst_type::rank; r++) {
      message += ",";
      message += std::to_string(dst.extent(r));
    }
    message += ") ";
    message += src.label();
    message += "(";
    message += std::to_string(src.extent(0));
    for (size_t r = 1; r < src_type::rank; r++) {
      message += ",";
      message += std::to_string(src.extent(r));
    }
    message += ") ";

    Kokkos::Impl::throw_runtime_exception(message);
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous
  // memory then can byte-wise copy

  if (std::is_same_v<typename dst_type::value_type,
                     typename src_type::non_const_value_type> &&
      (std::is_same_v<typename dst_type::array_layout,
                      typename src_type::array_layout> ||
       (dst_type::rank == 1 && src_type::rank == 1)) &&
      dst.span_is_contiguous() && src.span_is_contiguous() &&
      ((dst_type::rank < 1) || (dst.stride_0() == src.stride_0())) &&
      ((dst_type::rank < 2) || (dst.stride_1() == src.stride_1())) &&
      ((dst_type::rank < 3) || (dst.stride_2() == src.stride_2())) &&
      ((dst_type::rank < 4) || (dst.stride_3() == src.stride_3())) &&
      ((dst_type::rank < 5) || (dst.stride_4() == src.stride_4())) &&
      ((dst_type::rank < 6) || (dst.stride_5() == src.stride_5())) &&
      ((dst_type::rank < 7) || (dst.stride_6() == src.stride_6())) &&
      ((dst_type::rank < 8) || (dst.stride_7() == src.stride_7()))) {
    const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();
    if ((void*)dst.data() != (void*)src.data() && 0 < nbytes) {
      Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space, ExecSpace>(
          exec_space, dst.data(), src.data(), nbytes);
    }
  } else {
    // Copying data between views in accessible memory spaces and either
    // non-contiguous or incompatible shape.

    constexpr bool ExecCanAccessSrcDst =
        Kokkos::SpaceAccessibility<ExecSpace, dst_memory_space>::accessible &&
        Kokkos::SpaceAccessibility<ExecSpace, src_memory_space>::accessible;
    constexpr bool DstExecCanAccessSrc =
        Kokkos::SpaceAccessibility<dst_execution_space,
                                   src_memory_space>::accessible;
    constexpr bool SrcExecCanAccessDst =
        Kokkos::SpaceAccessibility<src_execution_space,
                                   dst_memory_space>::accessible;

    if constexpr (ExecCanAccessSrcDst) {
      Impl::view_copy(exec_space, dst, src);
    } else if constexpr (DstExecCanAccessSrc || SrcExecCanAccessDst) {
      using cpy_exec_space =
          std::conditional_t<DstExecCanAccessSrc, dst_execution_space,
                             src_execution_space>;
      exec_space.fence(
          "Kokkos::deep_copy: view-to-view noncontiguous copy on space, pre "
          "copy");
      Impl::view_copy(cpy_exec_space(), dst, src);
      cpy_exec_space().fence(
          "Kokkos::deep_copy: view-to-view noncontiguous copy on space, post "
          "copy");
    } else {
      Kokkos::Impl::throw_runtime_exception(
          "deep_copy given views that would require a temporary allocation");
    }
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {
template <typename ViewType>
bool size_mismatch(const ViewType& view, unsigned int max_extent,
                   const size_t new_extents[8]) {
  for (unsigned int dim = 0; dim < max_extent; ++dim)
    if (new_extents[dim] != view.extent(dim)) {
      return true;
    }
  for (unsigned int dim = max_extent; dim < 8; ++dim)
    if (new_extents[dim] != KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
      return true;
    }
  return false;
}

}  // namespace Impl

/** \brief  Resize a view with copying old data to new data at the corresponding
 * indices. */
template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutLeft> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutRight>>
impl_resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
            Kokkos::View<T, P...>& v, const size_t n0, const size_t n1,
            const size_t n2, const size_t n3, const size_t n4, const size_t n5,
            const size_t n6, const size_t n7) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only resize managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::resize "
                "must not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a memory space instance!");

  // TODO (mfh 27 Jun 2017) If the old View has enough space but just
  // different dimensions (e.g., if the product of the dimensions,
  // including extra space for alignment, will not change), then
  // consider just reusing storage.  For now, Kokkos always
  // reallocates if any of the dimensions change, even if the old View
  // has enough space.

  const size_t new_extents[8] = {n0, n1, n2, n3, n4, n5, n6, n7};
  const bool sizeMismatch = Impl::size_mismatch(v, v.rank_dynamic, new_extents);

  if (sizeMismatch) {
    auto prop_copy = [&]() {
      if constexpr (view_type::traits::impl_is_customized) {
        // FIXME SACADO: this is specializing for sacado, might need a better
        // thing
        Kokkos::Impl::AccessorArg_t acc_arg{new_extents[view_type::rank()]};
        return Impl::with_properties_if_unset(
            arg_prop, acc_arg, typename view_type::execution_space{},
            v.label());
      } else
        return Impl::with_properties_if_unset(
            arg_prop, typename view_type::execution_space{}, v.label());
      ;
    }();

    view_type v_resized;
    if constexpr (view_type::rank() == 0) {
      v_resized = view_type(prop_copy);
    } else if constexpr (view_type::rank() == 1) {
      v_resized = view_type(prop_copy, n0);
    } else if constexpr (view_type::rank() == 2) {
      v_resized = view_type(prop_copy, n0, n1);
    } else if constexpr (view_type::rank() == 3) {
      v_resized = view_type(prop_copy, n0, n1, n2);
    } else if constexpr (view_type::rank() == 4) {
      v_resized = view_type(prop_copy, n0, n1, n2, n3);
    } else if constexpr (view_type::rank() == 5) {
      v_resized = view_type(prop_copy, n0, n1, n2, n3, n4);
    } else if constexpr (view_type::rank() == 6) {
      v_resized = view_type(prop_copy, n0, n1, n2, n3, n4, n5);
    } else if constexpr (view_type::rank() == 7) {
      v_resized = view_type(prop_copy, n0, n1, n2, n3, n4, n5, n6);
    } else {
      v_resized = view_type(prop_copy, n0, n1, n2, n3, n4, n5, n6, n7);
    }

    if constexpr (alloc_prop_input::has_execution_space)
      Kokkos::Impl::ViewRemap<view_type, view_type>(
          v_resized, v, Impl::get_property<Impl::ExecutionSpaceTag>(prop_copy));
    else {
      Kokkos::Impl::ViewRemap<view_type, view_type>(v_resized, v);
      Kokkos::fence("Kokkos::resize(View)");
    }

    v = v_resized;
  }
}

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutLeft> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutRight>>
resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
       Kokkos::View<T, P...>& v, const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_resize(arg_prop, v, n0, n1, n2, n3, n4, n5, n6, n7);
}

template <class T, class... P>
inline std::enable_if_t<
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutLeft> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutRight>>
resize(Kokkos::View<T, P...>& v, const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_resize(Impl::ViewCtorProp<>{}, v, n0, n1, n2, n3, n4, n5, n6, n7);
}

template <class I, class T, class... P>
inline std::enable_if_t<
    (Impl::is_view_ctor_property<I>::value ||
     Kokkos::is_execution_space<I>::value) &&
    (std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                    Kokkos::LayoutLeft> ||
     std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                    Kokkos::LayoutRight>)>
resize(const I& arg_prop, Kokkos::View<T, P...>& v,
       const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_resize(Kokkos::view_alloc(arg_prop), v, n0, n1, n2, n3, n4, n5, n6, n7);
}

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutLeft> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutRight> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutStride>>
impl_resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
            Kokkos::View<T, P...>& v,
            const typename Kokkos::View<T, P...>::array_layout& layout) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only resize managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::resize "
                "must not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a memory space instance!");

  if (v.layout() != layout) {
    auto prop_copy = Impl::with_properties_if_unset(arg_prop, v.label());

    view_type v_resized(prop_copy, layout);

    if constexpr (alloc_prop_input::has_execution_space)
      Kokkos::Impl::ViewRemap<view_type, view_type>(
          v_resized, v, Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop));
    else {
      Kokkos::Impl::ViewRemap<view_type, view_type>(v_resized, v);
      Kokkos::fence("Kokkos::resize(View)");
    }

    v = v_resized;
  }
}

// FIXME User-provided (custom) layouts are not required to have a comparison
// operator. Hence, there is no way to check if the requested layout is actually
// the same as the existing one.
template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    !(std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                     Kokkos::LayoutLeft> ||
      std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                     Kokkos::LayoutRight> ||
      std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                     Kokkos::LayoutStride>)>
impl_resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
            Kokkos::View<T, P...>& v,
            const typename Kokkos::View<T, P...>::array_layout& layout) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only resize managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::resize "
                "must not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a memory space instance!");

  auto prop_copy = Impl::with_properties_if_unset(arg_prop, v.label());

  view_type v_resized(prop_copy, layout);

  if constexpr (alloc_prop_input::has_execution_space)
    Kokkos::Impl::ViewRemap<view_type, view_type>(
        v_resized, v, Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop));
  else {
    Kokkos::Impl::ViewRemap<view_type, view_type>(v_resized, v);
    Kokkos::fence("Kokkos::resize(View)");
  }

  v = v_resized;
}

template <class T, class... P, class... ViewCtorArgs>
inline void resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
                   Kokkos::View<T, P...>& v,
                   const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_resize(arg_prop, v, layout);
}

template <class I, class T, class... P>
inline std::enable_if_t<Impl::is_view_ctor_property<I>::value ||
                        Kokkos::is_execution_space<I>::value>
resize(const I& arg_prop, Kokkos::View<T, P...>& v,
       const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_resize(arg_prop, v, layout);
}

template <class ExecutionSpace, class T, class... P>
inline void resize(const ExecutionSpace& exec_space, Kokkos::View<T, P...>& v,
                   const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_resize(Impl::ViewCtorProp<>(), exec_space, v, layout);
}

template <class T, class... P>
inline void resize(Kokkos::View<T, P...>& v,
                   const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_resize(Impl::ViewCtorProp<>{}, v, layout);
}

/** \brief  Resize a view with discarding old data. */
template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutLeft> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutRight>>
impl_realloc(Kokkos::View<T, P...>& v, const size_t n0, const size_t n1,
             const size_t n2, const size_t n3, const size_t n4, const size_t n5,
             const size_t n6, const size_t n7,
             const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only realloc managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a memory space instance!");

  const size_t new_extents[8] = {n0, n1, n2, n3, n4, n5, n6, n7};
  const bool sizeMismatch = Impl::size_mismatch(v, v.rank_dynamic, new_extents);

  if (sizeMismatch) {
    auto arg_prop_copy = Impl::with_properties_if_unset(arg_prop, v.label());
    v = view_type();  // Best effort to deallocate in case no other view refers
                      // to the shared allocation
    v = view_type(arg_prop_copy, n0, n1, n2, n3, n4, n5, n6, n7);
    return;
  }

  if constexpr (alloc_prop_input::initialize) {
    if constexpr (alloc_prop_input::has_execution_space) {
      const auto& exec_space =
          Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop);
      Kokkos::deep_copy(exec_space, v, typename view_type::value_type{});
    } else
      Kokkos::deep_copy(v, typename view_type::value_type{});
  }
}

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutLeft> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutRight>>
realloc(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
        Kokkos::View<T, P...>& v,
        const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_realloc(v, n0, n1, n2, n3, n4, n5, n6, n7, arg_prop);
}

template <class T, class... P>
inline std::enable_if_t<
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutLeft> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutRight>>
realloc(Kokkos::View<T, P...>& v,
        const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_realloc(v, n0, n1, n2, n3, n4, n5, n6, n7, Impl::ViewCtorProp<>{});
}

template <class I, class T, class... P>
inline std::enable_if_t<
    Impl::is_view_ctor_property<I>::value &&
    (std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                    Kokkos::LayoutLeft> ||
     std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                    Kokkos::LayoutRight>)>
realloc(const I& arg_prop, Kokkos::View<T, P...>& v,
        const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_realloc(v, n0, n1, n2, n3, n4, n5, n6, n7, Kokkos::view_alloc(arg_prop));
}

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutLeft> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutRight> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutStride>>
impl_realloc(Kokkos::View<T, P...>& v,
             const typename Kokkos::View<T, P...>::array_layout& layout,
             const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only realloc managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a memory space instance!");

  if (v.layout() != layout) {
    v = view_type();  // Deallocate first, if the only view to allocation
    v = view_type(arg_prop, layout);
    return;
  }

  if constexpr (alloc_prop_input::initialize) {
    if constexpr (alloc_prop_input::has_execution_space) {
      const auto& exec_space =
          Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop);
      Kokkos::deep_copy(exec_space, v, typename view_type::value_type{});
    } else
      Kokkos::deep_copy(v, typename view_type::value_type{});
  }
}

// FIXME User-provided (custom) layouts are not required to have a comparison
// operator. Hence, there is no way to check if the requested layout is actually
// the same as the existing one.
template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    !(std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                     Kokkos::LayoutLeft> ||
      std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                     Kokkos::LayoutRight> ||
      std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                     Kokkos::LayoutStride>)>
impl_realloc(Kokkos::View<T, P...>& v,
             const typename Kokkos::View<T, P...>::array_layout& layout,
             const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only realloc managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a memory space instance!");

  auto arg_prop_copy = Impl::with_properties_if_unset(arg_prop, v.label());

  v = view_type();  // Deallocate first, if the only view to allocation
  v = view_type(arg_prop_copy, layout);
}

template <class T, class... P, class... ViewCtorArgs>
inline void realloc(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    Kokkos::View<T, P...>& v,
    const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_realloc(v, layout, arg_prop);
}

template <class I, class T, class... P>
inline std::enable_if_t<Impl::is_view_ctor_property<I>::value> realloc(
    const I& arg_prop, Kokkos::View<T, P...>& v,
    const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_realloc(v, layout, Kokkos::view_alloc(arg_prop));
}

template <class T, class... P>
inline void realloc(
    Kokkos::View<T, P...>& v,
    const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_realloc(v, layout, Impl::ViewCtorProp<>{});
}

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

// Deduce Mirror Types
template <class Space, class T, class... P>
struct MirrorViewType {
  // The incoming view_type
  using src_view_type = typename Kokkos::View<T, P...>;
  // The memory space for the mirror view
  using memory_space = typename Space::memory_space;
  // Check whether it is the same memory space
  static constexpr bool is_same_memspace =
      std::is_same_v<memory_space, typename src_view_type::memory_space>;
  // The array_layout
  using array_layout = typename src_view_type::array_layout;
  // The data type (we probably want it non-const since otherwise we can't even
  // deep_copy to it.
  using data_type = typename src_view_type::non_const_data_type;
  // The destination view type if it is not the same memory space
  using dest_view_type = Kokkos::View<data_type, array_layout, Space>;
  // If it is the same memory_space return the existsing view_type
  // This will also keep the unmanaged trait if necessary
  using view_type =
      std::conditional_t<is_same_memspace, src_view_type, dest_view_type>;
};

// collection of static asserts for create_mirror and create_mirror_view
template <class... ViewCtorArgs>
void check_view_ctor_args_create_mirror() {
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(
      !alloc_prop_input::has_label,
      "The view constructor arguments passed to Kokkos::create_mirror[_view] "
      "must not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to "
                "Kokkos::create_mirror[_view] must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::allow_padding,
                "The view constructor arguments passed to "
                "Kokkos::create_mirror[_view] must "
                "not explicitly allow padding!");
}

// create a mirror
// private interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs>
inline auto create_mirror(const Kokkos::View<T, P...>& src,
                          const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  check_view_ctor_args_create_mirror<ViewCtorArgs...>();

  auto prop_copy = Impl::with_properties_if_unset(
      arg_prop, std::string(src.label()).append("_mirror"));

  if constexpr (Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space) {
    using memory_space = typename decltype(prop_copy)::memory_space;
    using dst_type =
        typename Impl::MirrorViewType<memory_space, T, P...>::dest_view_type;
#ifndef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
    // This is necessary because constructing non-const element type from
    // const element type accessors is not generally supported
    // We don't construct from the src accessor generally because our accessors
    // aren't generally constructible from each other.
    // We could change that, but all our internal accessors are stateless anyway
    // right now. So for now if you have custom accessors that need to carry
    // forward information you just have to make the conversion constructors
    // work.
    if constexpr (std::is_constructible_v<
                      typename dst_type::accessor_type,
                      typename Kokkos::View<T, P...>::accessor_type>)
      return dst_type(prop_copy, src.mapping(), src.accessor());
    else
      return dst_type(prop_copy, src.layout());
#else
    return dst_type(prop_copy, src.layout());
#endif
  } else {
    using dst_type = typename View<T, P...>::HostMirror;
#ifndef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
    // This is necessary because constructing non-const element type from
    // const element type accessors is not generally supported
    if constexpr (std::is_constructible_v<
                      typename dst_type::accessor_type,
                      typename Kokkos::View<T, P...>::accessor_type>)
      return dst_type(prop_copy, src.mapping(), src.accessor());
    else
      return dst_type(prop_copy, src.layout());
#else
    return dst_type(prop_copy, src.layout());
#endif
  }
#if defined(KOKKOS_COMPILER_NVCC) && KOKKOS_COMPILER_NVCC >= 1130 && \
    !defined(KOKKOS_COMPILER_MSVC)
  __builtin_unreachable();
#endif
}
}  // namespace Impl

// public interface
template <class T, class... P,
          typename = std::enable_if_t<
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
auto create_mirror(Kokkos::View<T, P...> const& src) {
  return Impl::create_mirror(src, Impl::ViewCtorProp<>{});
}

// public interface that accepts a without initializing flag
template <class T, class... P,
          typename = std::enable_if_t<
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
auto create_mirror(Kokkos::Impl::WithoutInitializing_t wi,
                   Kokkos::View<T, P...> const& src) {
  return Impl::create_mirror(src, view_alloc(wi));
}

// public interface that accepts a space
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<
              Kokkos::is_space<Space>::value &&
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
auto create_mirror(Space const&, Kokkos::View<T, P...> const& src) {
  return Impl::create_mirror(src, view_alloc(typename Space::memory_space{}));
}

// public interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs,
          typename = std::enable_if_t<
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
auto create_mirror(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
                   Kokkos::View<T, P...> const& src) {
  return Impl::create_mirror(src, arg_prop);
}

// public interface that accepts a space and a without initializing flag
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<
              Kokkos::is_space<Space>::value &&
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
auto create_mirror(Kokkos::Impl::WithoutInitializing_t wi, Space const&,
                   Kokkos::View<T, P...> const& src) {
  return Impl::create_mirror(src,
                             view_alloc(typename Space::memory_space{}, wi));
}

namespace Impl {

// choose a `Kokkos::create_mirror` adapted for the provided view and the
// provided arguments
template <class View, class... ViewCtorArgs>
inline auto choose_create_mirror(
    const View& src, const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  // Due to the fact that users can overload `Kokkos::create_mirror`, but also
  // that they may not have implemented all of its different possible
  // variations, this function chooses the correct private or public version of
  // it to call.
  // This helper should be used by any overload of
  // `Kokkos::Impl::create_mirror_view`.

  if constexpr (std::is_void_v<typename View::traits::specialize>) {
    // if the view is not specialized, just call the Impl function

    // using ADL to find the later defined overload of the function
    using namespace Kokkos::Impl;

    return create_mirror(src, arg_prop);
  } else {
    // otherwise, recreate the public call
    using ViewProp = Impl::ViewCtorProp<ViewCtorArgs...>;

    // using ADL to find the later defined overload of the function
    using namespace Kokkos;

    if constexpr (sizeof...(ViewCtorArgs) == 0) {
      // if there are no view constructor args, call the specific public
      // function
      return create_mirror(src);
    } else if constexpr (sizeof...(ViewCtorArgs) == 1 &&
                         ViewProp::has_memory_space) {
      // if there is one view constructor arg and it has a memory space, call
      // the specific public function
      return create_mirror(typename ViewProp::memory_space{}, src);
    } else if constexpr (sizeof...(ViewCtorArgs) == 1 &&
                         !ViewProp::initialize) {
      // if there is one view constructor arg and it has a without initializing
      // mark, call the specific public function
      return create_mirror(typename Kokkos::Impl::WithoutInitializing_t{}, src);
    } else if constexpr (sizeof...(ViewCtorArgs) == 2 &&
                         ViewProp::has_memory_space && !ViewProp::initialize) {
      // if there is two view constructor args and they have a memory space and
      // a without initializing mark, call the specific public function
      return create_mirror(typename Kokkos::Impl::WithoutInitializing_t{},
                           typename ViewProp::memory_space{}, src);
    } else {
      // if there are other constructor args, call the generic public function

      // Beware, there are some libraries using Kokkos that don't implement
      // this overload (hence the reason for this present function to exist).
      return create_mirror(arg_prop, src);
    }
  }

#if defined(KOKKOS_COMPILER_NVCC) && KOKKOS_COMPILER_NVCC >= 1130 && \
    !defined(KOKKOS_COMPILER_MSVC)
  __builtin_unreachable();
#endif
}

// create a mirror view
// private interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs>
inline auto create_mirror_view(
    const Kokkos::View<T, P...>& src,
    [[maybe_unused]] const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  if constexpr (!Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space) {
    if constexpr (std::is_same_v<typename Kokkos::View<T, P...>::memory_space,
                                 typename Kokkos::View<
                                     T, P...>::HostMirror::memory_space> &&
                  std::is_same_v<
                      typename Kokkos::View<T, P...>::data_type,
                      typename Kokkos::View<T, P...>::HostMirror::data_type>) {
      check_view_ctor_args_create_mirror<ViewCtorArgs...>();
      return typename Kokkos::View<T, P...>::HostMirror(src);
    } else {
      return Kokkos::Impl::choose_create_mirror(src, arg_prop);
    }
  } else {
    if constexpr (Impl::MirrorViewType<typename Impl::ViewCtorProp<
                                           ViewCtorArgs...>::memory_space,
                                       T, P...>::is_same_memspace) {
      check_view_ctor_args_create_mirror<ViewCtorArgs...>();
      return typename Impl::MirrorViewType<
          typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space, T,
          P...>::view_type(src);
    } else {
      return Kokkos::Impl::choose_create_mirror(src, arg_prop);
    }
  }
#if defined(KOKKOS_COMPILER_NVCC) && KOKKOS_COMPILER_NVCC >= 1130 && \
    !defined(KOKKOS_COMPILER_MSVC)
  __builtin_unreachable();
#endif
}
}  // namespace Impl

// public interface
template <class T, class... P>
auto create_mirror_view(const Kokkos::View<T, P...>& src) {
  return Impl::create_mirror_view(src, view_alloc());
}

// public interface that accepts a without initializing flag
template <class T, class... P>
auto create_mirror_view(Kokkos::Impl::WithoutInitializing_t wi,
                        Kokkos::View<T, P...> const& src) {
  return Impl::create_mirror_view(src, view_alloc(wi));
}

// public interface that accepts a space
template <class Space, class T, class... P,
          class Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
auto create_mirror_view(const Space&, const Kokkos::View<T, P...>& src) {
  return Impl::create_mirror_view(src,
                                  view_alloc(typename Space::memory_space()));
}

// public interface that accepts a space and a without initializing flag
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
auto create_mirror_view(Kokkos::Impl::WithoutInitializing_t wi, Space const&,
                        Kokkos::View<T, P...> const& src) {
  return Impl::create_mirror_view(
      src, view_alloc(typename Space::memory_space{}, wi));
}

// public interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs,
          typename = std::enable_if_t<
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
auto create_mirror_view(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
                        const Kokkos::View<T, P...>& src) {
  return Impl::create_mirror_view(src, arg_prop);
}

namespace Impl {

// collection of static asserts for create_mirror_view_and_copy
template <class... ViewCtorArgs>
void check_view_ctor_args_create_mirror_view_and_copy() {
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(
      alloc_prop_input::has_memory_space,
      "The view constructor arguments passed to "
      "Kokkos::create_mirror_view_and_copy must include a memory space!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to "
                "Kokkos::create_mirror_view_and_copy must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::allow_padding,
                "The view constructor arguments passed to "
                "Kokkos::create_mirror_view_and_copy must "
                "not explicitly allow padding!");
}

}  // namespace Impl

// create a mirror view and deep copy it
// public interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class... ViewCtorArgs, class T, class... P,
          class Enable = std::enable_if_t<
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
auto create_mirror_view_and_copy(
    [[maybe_unused]] const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    const Kokkos::View<T, P...>& src) {
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  Impl::check_view_ctor_args_create_mirror_view_and_copy<ViewCtorArgs...>();

  if constexpr (Impl::MirrorViewType<typename alloc_prop_input::memory_space, T,
                                     P...>::is_same_memspace) {
    // same behavior as deep_copy(src, src)
    if constexpr (!alloc_prop_input::has_execution_space)
      fence(
          "Kokkos::create_mirror_view_and_copy: fence before returning src "
          "view");
    return src;
  } else {
    using Space  = typename alloc_prop_input::memory_space;
    using Mirror = typename Impl::MirrorViewType<Space, T, P...>::view_type;

    auto arg_prop_copy = Impl::with_properties_if_unset(
        arg_prop, std::string{}, WithoutInitializing,
        typename Space::execution_space{});

    std::string& label = Impl::get_property<Impl::LabelTag>(arg_prop_copy);
    if (label.empty()) label = src.label();
    auto mirror = typename Mirror::non_const_type{arg_prop_copy, src.layout()};
    if constexpr (alloc_prop_input::has_execution_space) {
      deep_copy(Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop_copy),
                mirror, src);
    } else
      deep_copy(mirror, src);
    return mirror;
  }
#if defined(KOKKOS_COMPILER_NVCC) && KOKKOS_COMPILER_NVCC >= 1130 && \
    !defined(KOKKOS_COMPILER_MSVC)
  __builtin_unreachable();
#endif
}

// Previously when using auto here, the intel compiler 19.3 would
// sometimes not create a symbol, guessing that it somehow is a combination
// of auto and just forwarding arguments (see issue #5196)
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
typename Impl::MirrorViewType<Space, T, P...>::view_type
create_mirror_view_and_copy(
    const Space&, const Kokkos::View<T, P...>& src,
    std::string const& name = "",
    std::enable_if_t<
        std::is_void_v<typename ViewTraits<T, P...>::specialize>>* = nullptr) {
  return create_mirror_view_and_copy(
      Kokkos::view_alloc(typename Space::memory_space{}, name), src);
}

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
