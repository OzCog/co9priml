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

#ifndef KOKKOS_VECTOR_HPP
#define KOKKOS_VECTOR_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_VECTOR
#endif

#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
namespace {
[[deprecated("Deprecated <Kokkos_Vector.hpp> header is included")]] int
emit_warning_kokkos_vector_deprecated() {
  return 0;
}
static auto do_not_include = emit_warning_kokkos_vector_deprecated();
}  // namespace
#endif
#else
#error "Deprecated <Kokkos_Vector.hpp> header is included"
#endif

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_DualView.hpp>

/* Drop in replacement for std::vector based on Kokkos::DualView
 * Most functions only work on the host (it will not compile if called from
 * device kernel)
 *
 */
namespace Kokkos {

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
template <class Scalar, class Arg1Type = void>
class KOKKOS_DEPRECATED vector
    : public DualView<Scalar*, LayoutLeft, Arg1Type> {
 public:
  using value_type      = Scalar;
  using pointer         = Scalar*;
  using const_pointer   = const Scalar*;
  using reference       = Scalar&;
  using const_reference = const Scalar&;
  using iterator        = Scalar*;
  using const_iterator  = const Scalar*;
  using size_type       = size_t;

 private:
  size_t _size;
  float _extra_storage;
  using DV = DualView<Scalar*, LayoutLeft, Arg1Type>;

 public:
#ifdef KOKKOS_ENABLE_CUDA_UVM
  KOKKOS_INLINE_FUNCTION reference operator()(int i) const {
    return DV::view_host()(i);
  };
  KOKKOS_INLINE_FUNCTION reference operator[](int i) const {
    return DV::view_host()(i);
  };
#else
  inline reference operator()(int i) const { return DV::view_host()(i); }
  inline reference operator[](int i) const { return DV::view_host()(i); }
#endif

  /* Member functions which behave like std::vector functions */

  vector() : DV() {
    _size          = 0;
    _extra_storage = 1.1;
  }

  vector(int n, Scalar val = Scalar())
      : DualView<Scalar*, LayoutLeft, Arg1Type>("Vector", size_t(n * (1.1))) {
    _size                 = n;
    _extra_storage        = 1.1;
    DV::modified_flags(0) = 1;

    assign(n, val);
  }

  void resize(size_t n) {
    if (n >= span()) DV::resize(size_t(n * _extra_storage));
    _size = n;
  }

  void resize(size_t n, const Scalar& val) { assign(n, val); }

  void assign(size_t n, const Scalar& val) {
    /* Resize if necessary (behavior of std:vector) */

    if (n > span()) DV::resize(size_t(n * _extra_storage));
    _size = n;

    /* Assign value either on host or on device */

    if (DV::template need_sync<typename DV::t_dev::device_type>()) {
      set_functor_host f(DV::view_host(), val);
      parallel_for("Kokkos::vector::assign", n, f);
      typename DV::t_host::execution_space().fence(
          "Kokkos::vector::assign: fence after assigning values");
      DV::template modify<typename DV::t_host::device_type>();
    } else {
      set_functor f(DV::view_device(), val);
      parallel_for("Kokkos::vector::assign", n, f);
      typename DV::t_dev::execution_space().fence(
          "Kokkos::vector::assign: fence after assigning values");
      DV::template modify<typename DV::t_dev::device_type>();
    }
  }

  void reserve(size_t n) { DV::resize(size_t(n * _extra_storage)); }

  void push_back(Scalar val) {
    if (_size == span()) {
      size_t new_size = _size * _extra_storage;
      if (new_size == _size) new_size++;
      DV::resize(new_size);
    }

    DV::sync_host();
    DV::view_host()(_size) = val;
    _size++;
    DV::modify_host();
  }

  void pop_back() { _size--; }

  void clear() { _size = 0; }

  iterator insert(iterator it, const value_type& val) {
    return insert(it, 1, val);
  }

  iterator insert(iterator it, size_type count, const value_type& val) {
    if ((size() == 0) && (it == begin())) {
      resize(count, val);
      DV::sync_host();
      return begin();
    }
    DV::sync_host();
    DV::modify_host();
    if (std::less<>()(it, begin()) || std::less<>()(end(), it))
      Kokkos::abort("Kokkos::vector::insert : invalid insert iterator");
    if (count == 0) return it;
    ptrdiff_t start = std::distance(begin(), it);
    auto org_size   = size();
    resize(size() + count);

    std::copy_backward(begin() + start, begin() + org_size,
                       begin() + org_size + count);
    std::fill_n(begin() + start, count, val);

    return begin() + start;
  }

 private:
  template <class T>
  struct impl_is_input_iterator : /* TODO replace this */ std::bool_constant<
                                      !std::is_convertible_v<T, size_type>> {};

 public:
  // TODO: can use detection idiom to generate better error message here later
  template <typename InputIterator>
  std::enable_if_t<impl_is_input_iterator<InputIterator>::value, iterator>
  insert(iterator it, InputIterator b, InputIterator e) {
    ptrdiff_t count = std::distance(b, e);

    DV::sync_host();
    DV::modify_host();
    if (std::less<>()(it, begin()) || std::less<>()(end(), it))
      Kokkos::abort("Kokkos::vector::insert : invalid insert iterator");

    ptrdiff_t start = std::distance(begin(), it);
    auto org_size   = size();

    // Note: resize(...) invalidates it; use begin() + start instead
    resize(size() + count);

    std::copy_backward(begin() + start, begin() + org_size,
                       begin() + org_size + count);
    std::copy(b, e, begin() + start);

    return begin() + start;
  }

  KOKKOS_INLINE_FUNCTION constexpr bool is_allocated() const {
    return DV::is_allocated();
  }

  size_type size() const { return _size; }
  size_type max_size() const { return 2000000000; }
  size_type span() const { return DV::span(); }
  bool empty() const { return _size == 0; }

  pointer data() const { return DV::view_host().data(); }

  iterator begin() const { return DV::view_host().data(); }

  const_iterator cbegin() const { return DV::view_host().data(); }

  iterator end() const {
    return _size > 0 ? DV::view_host().data() + _size : DV::view_host().data();
  }

  const_iterator cend() const {
    return _size > 0 ? DV::view_host().data() + _size : DV::view_host().data();
  }

  reference front() { return DV::view_host()(0); }

  reference back() { return DV::view_host()(_size - 1); }

  const_reference front() const { return DV::view_host()(0); }

  const_reference back() const { return DV::view_host()(_size - 1); }

  /* std::algorithms which work originally with iterators, here they are
   * implemented as member functions */

  size_t lower_bound(const size_t& start, const size_t& theEnd,
                     const Scalar& comp_val) const {
    int lower = start;  // FIXME (mfh 24 Apr 2014) narrowing conversion
    int upper =
        _size > theEnd
            ? theEnd
            : _size - 1;  // FIXME (mfh 24 Apr 2014) narrowing conversion
    if (upper <= lower) {
      return theEnd;
    }

    Scalar lower_val = DV::view_host()(lower);
    Scalar upper_val = DV::view_host()(upper);
    size_t idx       = (upper + lower) / 2;
    Scalar val       = DV::view_host()(idx);
    if (val > upper_val) return upper;
    if (val < lower_val) return start;

    while (upper > lower) {
      if (comp_val > val) {
        lower = ++idx;
      } else {
        upper = idx;
      }
      idx = (upper + lower) / 2;
      val = DV::view_host()(idx);
    }
    return idx;
  }

  bool is_sorted() {
    for (int i = 0; i < _size - 1; i++) {
      if (DV::view_host()(i) > DV::view_host()(i + 1)) return false;
    }
    return true;
  }

  iterator find(Scalar val) const {
    if (_size == 0) return end();

    int upper, lower, current;
    current = _size / 2;
    upper   = _size - 1;
    lower   = 0;

    if ((val < DV::view_host()(0)) || (val > DV::view_host()(_size - 1)))
      return end();

    while (upper > lower) {
      if (val > DV::view_host()(current))
        lower = current + 1;
      else
        upper = current;
      current = (upper + lower) / 2;
    }

    if (val == DV::view_host()(current))
      return &DV::view_host()(current);
    else
      return end();
  }

  /* Additional functions for data management */

  void device_to_host() { deep_copy(DV::view_host(), DV::view_device()); }
  void host_to_device() const { deep_copy(DV::view_device(), DV::view_host()); }

  void on_host() { DV::template modify<typename DV::t_host::device_type>(); }
  void on_device() { DV::template modify<typename DV::t_dev::device_type>(); }

  void set_overallocation(float extra) { _extra_storage = 1.0 + extra; }

 public:
  struct set_functor {
    using execution_space = typename DV::t_dev::execution_space;
    typename DV::t_dev _data;
    Scalar _val;

    set_functor(typename DV::t_dev data, Scalar val) : _data(data), _val(val) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& i) const { _data(i) = _val; }
  };

  struct set_functor_host {
    using execution_space = typename DV::t_host::execution_space;
    typename DV::t_host _data;
    Scalar _val;

    set_functor_host(typename DV::t_host data, Scalar val)
        : _data(data), _val(val) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& i) const { _data(i) = _val; }
  };
};
#endif

}  // namespace Kokkos
#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_VECTOR
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_VECTOR
#endif
#endif
