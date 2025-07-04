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

#include "config.hpp"

#include <cstdio>
#include <cstdlib>
#include <type_traits> // std::is_void
#if defined(MDSPAN_IMPL_HAS_CUDA) || defined(MDSPAN_IMPL_HAS_HIP) || defined(MDSPAN_IMPL_HAS_SYCL)
#include "assert.h"
#endif

#ifndef MDSPAN_IMPL_HOST_DEVICE
#  if defined(MDSPAN_IMPL_HAS_CUDA) || defined(MDSPAN_IMPL_HAS_HIP)
#    define MDSPAN_IMPL_HOST_DEVICE __host__ __device__
#  else
#    define MDSPAN_IMPL_HOST_DEVICE
#  endif
#endif

#ifndef MDSPAN_FORCE_INLINE_FUNCTION
#  ifdef MDSPAN_IMPL_COMPILER_MSVC // Microsoft compilers
#    define MDSPAN_FORCE_INLINE_FUNCTION __forceinline MDSPAN_IMPL_HOST_DEVICE
#  else
#    define MDSPAN_FORCE_INLINE_FUNCTION __attribute__((always_inline)) MDSPAN_IMPL_HOST_DEVICE
#  endif
#endif

#ifndef MDSPAN_INLINE_FUNCTION
#  define MDSPAN_INLINE_FUNCTION inline MDSPAN_IMPL_HOST_DEVICE
#endif

#ifndef MDSPAN_FUNCTION
#  define MDSPAN_FUNCTION MDSPAN_IMPL_HOST_DEVICE
#endif

#ifdef MDSPAN_IMPL_HAS_HIP
#  define MDSPAN_DEDUCTION_GUIDE MDSPAN_IMPL_HOST_DEVICE
#else
#  define MDSPAN_DEDUCTION_GUIDE
#endif

// In CUDA defaulted functions do not need host device markup
#ifndef MDSPAN_INLINE_FUNCTION_DEFAULTED
#  define MDSPAN_INLINE_FUNCTION_DEFAULTED
#endif

//==============================================================================
// <editor-fold desc="Preprocessor helpers"> {{{1

#define MDSPAN_PP_COUNT(...) \
  MDSPAN_IMPL_PP_INTERNAL_EXPAND_ARGS( \
    MDSPAN_IMPL_PP_INTERNAL_ARGS_AUGMENTER(__VA_ARGS__) \
  )

#define MDSPAN_IMPL_PP_INTERNAL_ARGS_AUGMENTER(...) unused, __VA_ARGS__
#define MDSPAN_IMPL_PP_INTERNAL_EXPAND(x) x
#define MDSPAN_IMPL_PP_INTERNAL_EXPAND_ARGS(...) \
  MDSPAN_IMPL_PP_INTERNAL_EXPAND( \
    MDSPAN_IMPL_PP_INTERNAL_COUNT( \
      __VA_ARGS__, 69, 68, 67, 66, 65, 64, 63, 62, 61, \
      60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,  \
      48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37,  \
      36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25,  \
      24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,  \
      12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 \
    ) \
  )
# define MDSPAN_IMPL_PP_INTERNAL_COUNT( \
         _1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, \
    _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
    _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, \
    _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, \
    _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, \
    _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, \
    _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, \
    _70, count, ...) count \
    /**/

#define MDSPAN_PP_STRINGIFY_IMPL(x) #x
#define MDSPAN_PP_STRINGIFY(x) MDSPAN_PP_STRINGIFY_IMPL(x)

#define MDSPAN_PP_CAT_IMPL(x, y) x ## y
#define MDSPAN_PP_CAT(x, y) MDSPAN_PP_CAT_IMPL(x, y)

#define MDSPAN_PP_EVAL(X, ...) X(__VA_ARGS__)

#define MDSPAN_PP_REMOVE_PARENS_IMPL(...) __VA_ARGS__
#define MDSPAN_PP_REMOVE_PARENS(...) MDSPAN_PP_REMOVE_PARENS_IMPL __VA_ARGS__

#define MDSPAN_IMPL_STANDARD_NAMESPACE_STRING MDSPAN_PP_STRINGIFY(MDSPAN_IMPL_STANDARD_NAMESPACE)
#define MDSPAN_IMPL_PROPOSED_NAMESPACE_STRING MDSPAN_PP_STRINGIFY(MDSPAN_IMPL_STANDARD_NAMESPACE) "::" MDSPAN_PP_STRINGIFY(MDSPAN_IMPL_PROPOSED_NAMESPACE)

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

#if defined(MDSPAN_IMPL_HAS_CUDA) || defined(MDSPAN_IMPL_HAS_HIP)
MDSPAN_FUNCTION inline void default_precondition_violation_handler(const char* cond, const char* file, unsigned line)
{
  ::printf("%s:%u: precondition failure: `%s`\n", file, line, cond);
  assert(0);
}
#elif defined(MDSPAN_IMPL_HAS_SYCL)
MDSPAN_FUNCTION inline void default_precondition_violation_handler(const char* cond, const char* file, unsigned line)
{
#ifdef __INTEL_LLVM_COMPILER
  sycl::ext::oneapi::experimental::printf("%s:%u: precondition failure: `%s`\n", file, line, cond);
#else
  (void) cond;
  (void) file;
  (void) line;
#endif
  assert(0);
}
#else
MDSPAN_FUNCTION inline void default_precondition_violation_handler(const char* cond, const char* file, unsigned line)
{
  std::fprintf(stderr, "%s:%u: precondition failure: `%s`\n", file, line, cond);
  std::abort();
}
#endif

} // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#ifndef MDSPAN_IMPL_PRECONDITION_VIOLATION_HANDLER
#define MDSPAN_IMPL_PRECONDITION_VIOLATION_HANDLER(cond, file, line) \
  MDSPAN_IMPL_STANDARD_NAMESPACE::detail::default_precondition_violation_handler(cond, file, line)
#endif

#ifndef MDSPAN_IMPL_CHECK_PRECONDITION
  #ifndef NDEBUG
    #define MDSPAN_IMPL_CHECK_PRECONDITION 0
  #else
    #define MDSPAN_IMPL_CHECK_PRECONDITION 1
  #endif
#endif

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

template <bool check = MDSPAN_IMPL_CHECK_PRECONDITION>
MDSPAN_FUNCTION constexpr void precondition(const char* cond, const char* file, unsigned line)
{
  if (!check) { return; }
  // in case the macro doesn't use the arguments for custom macros
  (void) cond;
  (void) file;
  (void) line;
  MDSPAN_IMPL_PRECONDITION_VIOLATION_HANDLER(cond, file, line);
}

} // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#define MDSPAN_IMPL_PRECONDITION(...) \
  do { \
    if (!(__VA_ARGS__)) { \
      MDSPAN_IMPL_STANDARD_NAMESPACE::detail::precondition(#__VA_ARGS__, __FILE__, __LINE__); \
    } \
  } while (0)

// </editor-fold> end Preprocessor helpers }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Concept emulation"> {{{1

// These compatibility macros don't help with partial ordering, but they should do the trick
// for what we need to do with concepts in mdspan
#ifdef MDSPAN_IMPL_USE_CONCEPTS
#  define MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) > requires REQ
#  define MDSPAN_FUNCTION_REQUIRES(PAREN_PREQUALS, FNAME, PAREN_PARAMS, QUALS, REQ) \
     MDSPAN_PP_REMOVE_PARENS(PAREN_PREQUALS) FNAME PAREN_PARAMS QUALS requires REQ \
     /**/
#else
#  define MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) , typename ::std::enable_if<(REQ), int>::type = 0>
#  define MDSPAN_FUNCTION_REQUIRES(PAREN_PREQUALS, FNAME, PAREN_PARAMS, QUALS, REQ) \
     MDSPAN_TEMPLATE_REQUIRES( \
       class function_requires_ignored=void, \
       (std::is_void<function_requires_ignored>::value && REQ) \
     ) MDSPAN_PP_REMOVE_PARENS(PAREN_PREQUALS) FNAME PAREN_PARAMS QUALS \
     /**/
#endif

#if defined(MDSPAN_IMPL_COMPILER_MSVC) && (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL)
#  define MDSPAN_TEMPLATE_REQUIRES(...) \
      MDSPAN_PP_CAT( \
        MDSPAN_PP_CAT(MDSPAN_TEMPLATE_REQUIRES_, MDSPAN_PP_COUNT(__VA_ARGS__))\
        (__VA_ARGS__), \
      ) \
    /**/
#else
#  define MDSPAN_TEMPLATE_REQUIRES(...) \
    MDSPAN_PP_EVAL( \
        MDSPAN_PP_CAT(MDSPAN_TEMPLATE_REQUIRES_, MDSPAN_PP_COUNT(__VA_ARGS__)), \
        __VA_ARGS__ \
    ) \
    /**/
#endif

#define MDSPAN_TEMPLATE_REQUIRES_2(TP1, REQ) \
  template<TP1 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_3(TP1, TP2, REQ) \
  template<TP1, TP2 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_4(TP1, TP2, TP3, REQ) \
  template<TP1, TP2, TP3 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_5(TP1, TP2, TP3, TP4, REQ) \
  template<TP1, TP2, TP3, TP4 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_6(TP1, TP2, TP3, TP4, TP5, REQ) \
  template<TP1, TP2, TP3, TP4, TP5 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_7(TP1, TP2, TP3, TP4, TP5, TP6, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_8(TP1, TP2, TP3, TP4, TP5, TP6, TP7, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_9(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_10(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_11(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_12(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_13(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_14(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_15(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_16(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_17(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_18(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_19(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define MDSPAN_TEMPLATE_REQUIRES_20(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18, TP19, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18, TP19 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/

#define MDSPAN_INSTANTIATE_ONLY_IF_USED \
  MDSPAN_TEMPLATE_REQUIRES( \
    class instantiate_only_if_used_tparam=void, \
    ( MDSPAN_IMPL_TRAIT(std::is_void, instantiate_only_if_used_tparam) ) \
  ) \
  /**/

// </editor-fold> end Concept emulation }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="inline variables"> {{{1

#ifdef MDSPAN_IMPL_USE_INLINE_VARIABLES
#  define MDSPAN_IMPL_INLINE_VARIABLE inline
#else
#  define MDSPAN_IMPL_INLINE_VARIABLE
#endif

// </editor-fold> end inline variables }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Return type deduction"> {{{1

#if MDSPAN_IMPL_USE_RETURN_TYPE_DEDUCTION
#  define MDSPAN_IMPL_DEDUCE_RETURN_TYPE_SINGLE_LINE(SIGNATURE, BODY) \
    auto MDSPAN_PP_REMOVE_PARENS(SIGNATURE) { return MDSPAN_PP_REMOVE_PARENS(BODY); }
#  define MDSPAN_IMPL_DEDUCE_DECLTYPE_AUTO_RETURN_TYPE_SINGLE_LINE(SIGNATURE, BODY) \
    decltype(auto) MDSPAN_PP_REMOVE_PARENS(SIGNATURE) { return MDSPAN_PP_REMOVE_PARENS(BODY); }
#else
#  define MDSPAN_IMPL_DEDUCE_RETURN_TYPE_SINGLE_LINE(SIGNATURE, BODY) \
    auto MDSPAN_PP_REMOVE_PARENS(SIGNATURE) \
      -> std::remove_cv_t<std::remove_reference_t<decltype(BODY)>> \
    { return MDSPAN_PP_REMOVE_PARENS(BODY); }
#  define MDSPAN_IMPL_DEDUCE_DECLTYPE_AUTO_RETURN_TYPE_SINGLE_LINE(SIGNATURE, BODY) \
    auto MDSPAN_PP_REMOVE_PARENS(SIGNATURE) \
      -> decltype(BODY) \
    { return MDSPAN_PP_REMOVE_PARENS(BODY); }

#endif

// </editor-fold> end Return type deduction }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="fold expressions"> {{{1

struct enable_fold_comma { };

#ifdef MDSPAN_IMPL_USE_FOLD_EXPRESSIONS
#  define MDSPAN_IMPL_FOLD_AND(...) ((__VA_ARGS__) && ...)
#  define MDSPAN_IMPL_FOLD_AND_TEMPLATE(...) ((__VA_ARGS__) && ...)
#  define MDSPAN_IMPL_FOLD_OR(...) ((__VA_ARGS__) || ...)
#  define MDSPAN_IMPL_FOLD_ASSIGN_LEFT(INIT, ...) (INIT = ... = (__VA_ARGS__))
#  define MDSPAN_IMPL_FOLD_ASSIGN_RIGHT(PACK, ...) (PACK = ... = (__VA_ARGS__))
#  define MDSPAN_IMPL_FOLD_TIMES_RIGHT(PACK, ...) (PACK * ... * (__VA_ARGS__))
#  define MDSPAN_IMPL_FOLD_PLUS_RIGHT(PACK, ...) (PACK + ... + (__VA_ARGS__))
#  define MDSPAN_IMPL_FOLD_COMMA(...) ((__VA_ARGS__), ...)
#else

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

namespace fold_compatibility_impl {

// We could probably be more clever here, but at the (small) risk of losing some compiler understanding.  For the
// few operations we need, it's not worth generalizing over the operation

#if MDSPAN_IMPL_USE_RETURN_TYPE_DEDUCTION

MDSPAN_FORCE_INLINE_FUNCTION
constexpr decltype(auto) fold_right_and_impl() {
  return true;
}

template <class Arg, class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr decltype(auto) fold_right_and_impl(Arg&& arg, Args&&... args) {
  return ((Arg&&)arg) && fold_compatibility_impl::fold_right_and_impl((Args&&)args...);
}

MDSPAN_FORCE_INLINE_FUNCTION
constexpr decltype(auto) fold_right_or_impl() {
  return false;
}

template <class Arg, class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto fold_right_or_impl(Arg&& arg, Args&&... args) {
  return ((Arg&&)arg) || fold_compatibility_impl::fold_right_or_impl((Args&&)args...);
}

template <class Arg1>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto fold_left_assign_impl(Arg1&& arg1) {
  return (Arg1&&)arg1;
}

template <class Arg1, class Arg2, class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto fold_left_assign_impl(Arg1&& arg1, Arg2&& arg2, Args&&... args) {
  return fold_compatibility_impl::fold_left_assign_impl((((Arg1&&)arg1) = ((Arg2&&)arg2)), (Args&&)args...);
}

template <class Arg1>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto fold_right_assign_impl(Arg1&& arg1) {
  return (Arg1&&)arg1;
}

template <class Arg1, class Arg2, class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto fold_right_assign_impl(Arg1&& arg1, Arg2&& arg2,  Args&&... args) {
  return ((Arg1&&)arg1) = fold_compatibility_impl::fold_right_assign_impl((Arg2&&)arg2, (Args&&)args...);
}

template <class Arg1>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto fold_right_plus_impl(Arg1&& arg1) {
  return (Arg1&&)arg1;
}

template <class Arg1, class Arg2, class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto fold_right_plus_impl(Arg1&& arg1, Arg2&& arg2, Args&&... args) {
  return ((Arg1&&)arg1) + fold_compatibility_impl::fold_right_plus_impl((Arg2&&)arg2, (Args&&)args...);
}

template <class Arg1>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto fold_right_times_impl(Arg1&& arg1) {
  return (Arg1&&)arg1;
}

template <class Arg1, class Arg2, class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto fold_right_times_impl(Arg1&& arg1, Arg2&& arg2, Args&&... args) {
  return ((Arg1&&)arg1) * fold_compatibility_impl::fold_right_times_impl((Arg2&&)arg2, (Args&&)args...);
}

#else

//------------------------------------------------------------------------------
// <editor-fold desc="right and"> {{{2

template <class... Args>
struct fold_right_and_impl_;
template <>
struct fold_right_and_impl_<> {
  using rv = bool;
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl() noexcept {
    return true;
  }
};
template <class Arg, class... Args>
struct fold_right_and_impl_<Arg, Args...> {
  using next_t = fold_right_and_impl_<Args...>;
  using rv = decltype(std::declval<Arg>() && std::declval<typename next_t::rv>());
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl(Arg&& arg, Args&&... args) noexcept {
    return ((Arg&&)arg) && next_t::impl((Args&&)args...);
  }
};

template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename fold_right_and_impl_<Args...>::rv
fold_right_and_impl(Args&&... args) {
  return fold_right_and_impl_<Args...>::impl((Args&&)args...);
}

// </editor-fold> end right and }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right or"> {{{2

template <class... Args>
struct fold_right_or_impl_;
template <>
struct fold_right_or_impl_<> {
  using rv = bool;
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl() noexcept {
    return false;
  }
};
template <class Arg, class... Args>
struct fold_right_or_impl_<Arg, Args...> {
  using next_t = fold_right_or_impl_<Args...>;
  using rv = decltype(std::declval<Arg>() || std::declval<typename next_t::rv>());
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl(Arg&& arg, Args&&... args) noexcept {
    return ((Arg&&)arg) || next_t::impl((Args&&)args...);
  }
};

template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename fold_right_or_impl_<Args...>::rv
fold_right_or_impl(Args&&... args) {
  return fold_right_or_impl_<Args...>::impl((Args&&)args...);
}

// </editor-fold> end right or }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right plus"> {{{2

template <class... Args>
struct fold_right_plus_impl_;
template <class Arg>
struct fold_right_plus_impl_<Arg> {
  using rv = Arg&&;
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl(Arg&& arg) noexcept {
    return (Arg&&)arg;
  }
};
template <class Arg1, class Arg2, class... Args>
struct fold_right_plus_impl_<Arg1, Arg2, Args...> {
  using next_t = fold_right_plus_impl_<Arg2, Args...>;
  using rv = decltype(std::declval<Arg1>() + std::declval<typename next_t::rv>());
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl(Arg1&& arg, Arg2&& arg2, Args&&... args) noexcept {
    return ((Arg1&&)arg) + next_t::impl((Arg2&&)arg2, (Args&&)args...);
  }
};

template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename fold_right_plus_impl_<Args...>::rv
fold_right_plus_impl(Args&&... args) {
  return fold_right_plus_impl_<Args...>::impl((Args&&)args...);
}

// </editor-fold> end right plus }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right times"> {{{2

template <class... Args>
struct fold_right_times_impl_;
template <class Arg>
struct fold_right_times_impl_<Arg> {
  using rv = Arg&&;
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl(Arg&& arg) noexcept {
    return (Arg&&)arg;
  }
};
template <class Arg1, class Arg2, class... Args>
struct fold_right_times_impl_<Arg1, Arg2, Args...> {
  using next_t = fold_right_times_impl_<Arg2, Args...>;
  using rv = decltype(std::declval<Arg1>() * std::declval<typename next_t::rv>());
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl(Arg1&& arg, Arg2&& arg2, Args&&... args) noexcept {
    return ((Arg1&&)arg) * next_t::impl((Arg2&&)arg2, (Args&&)args...);
  }
};

template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename fold_right_times_impl_<Args...>::rv
fold_right_times_impl(Args&&... args) {
  return fold_right_times_impl_<Args...>::impl((Args&&)args...);
}

// </editor-fold> end right times }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right assign"> {{{2

template <class... Args>
struct fold_right_assign_impl_;
template <class Arg>
struct fold_right_assign_impl_<Arg> {
  using rv = Arg&&;
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl(Arg&& arg) noexcept {
    return (Arg&&)arg;
  }
};
template <class Arg1, class Arg2, class... Args>
struct fold_right_assign_impl_<Arg1, Arg2, Args...> {
  using next_t = fold_right_assign_impl_<Arg2, Args...>;
  using rv = decltype(std::declval<Arg1>() = std::declval<typename next_t::rv>());
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl(Arg1&& arg, Arg2&& arg2, Args&&... args) noexcept {
    return ((Arg1&&)arg) = next_t::impl((Arg2&&)arg2, (Args&&)args...);
  }
};

template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename fold_right_assign_impl_<Args...>::rv
fold_right_assign_impl(Args&&... args) {
  return fold_right_assign_impl_<Args...>::impl((Args&&)args...);
}

// </editor-fold> end right assign }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="left assign"> {{{2

template <class... Args>
struct fold_left_assign_impl_;
template <class Arg>
struct fold_left_assign_impl_<Arg> {
  using rv = Arg&&;
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl(Arg&& arg) noexcept {
    return (Arg&&)arg;
  }
};
template <class Arg1, class Arg2, class... Args>
struct fold_left_assign_impl_<Arg1, Arg2, Args...> {
  using assign_result_t = decltype(std::declval<Arg1>() = std::declval<Arg2>());
  using next_t = fold_left_assign_impl_<assign_result_t, Args...>;
  using rv = typename next_t::rv;
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr rv
  impl(Arg1&& arg, Arg2&& arg2, Args&&... args) noexcept {
    return next_t::impl(((Arg1&&)arg) = (Arg2&&)arg2, (Args&&)args...);
  }
};

template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename fold_left_assign_impl_<Args...>::rv
fold_left_assign_impl(Args&&... args) {
  return fold_left_assign_impl_<Args...>::impl((Args&&)args...);
}

// </editor-fold> end left assign }}}2
//------------------------------------------------------------------------------

#endif


template <class... Args>
constexpr enable_fold_comma fold_comma_impl(Args&&...) noexcept { return { }; }

template <bool... Bs>
struct fold_bools;

} // fold_compatibility_impl

} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#  define MDSPAN_IMPL_FOLD_AND(...) MDSPAN_IMPL_STANDARD_NAMESPACE::fold_compatibility_impl::fold_right_and_impl((__VA_ARGS__)...)
#  define MDSPAN_IMPL_FOLD_OR(...) MDSPAN_IMPL_STANDARD_NAMESPACE::fold_compatibility_impl::fold_right_or_impl((__VA_ARGS__)...)
#  define MDSPAN_IMPL_FOLD_ASSIGN_LEFT(INIT, ...) MDSPAN_IMPL_STANDARD_NAMESPACE::fold_compatibility_impl::fold_left_assign_impl(INIT, (__VA_ARGS__)...)
#  define MDSPAN_IMPL_FOLD_ASSIGN_RIGHT(PACK, ...) MDSPAN_IMPL_STANDARD_NAMESPACE::fold_compatibility_impl::fold_right_assign_impl((PACK)..., __VA_ARGS__)
#  define MDSPAN_IMPL_FOLD_TIMES_RIGHT(PACK, ...) MDSPAN_IMPL_STANDARD_NAMESPACE::fold_compatibility_impl::fold_right_times_impl((PACK)..., __VA_ARGS__)
#  define MDSPAN_IMPL_FOLD_PLUS_RIGHT(PACK, ...) MDSPAN_IMPL_STANDARD_NAMESPACE::fold_compatibility_impl::fold_right_plus_impl((PACK)..., __VA_ARGS__)
#  define MDSPAN_IMPL_FOLD_COMMA(...) MDSPAN_IMPL_STANDARD_NAMESPACE::fold_compatibility_impl::fold_comma_impl((__VA_ARGS__)...)

#  define MDSPAN_IMPL_FOLD_AND_TEMPLATE(...) \
  MDSPAN_IMPL_TRAIT(std::is_same, fold_compatibility_impl::fold_bools<(__VA_ARGS__)..., true>, fold_compatibility_impl::fold_bools<true, (__VA_ARGS__)...>)

#endif

// </editor-fold> end fold expressions }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Variable template compatibility"> {{{1

#if MDSPAN_IMPL_USE_VARIABLE_TEMPLATES
#  define MDSPAN_IMPL_TRAIT(TRAIT, ...) TRAIT##_v<__VA_ARGS__>
#else
#  define MDSPAN_IMPL_TRAIT(TRAIT, ...) TRAIT<__VA_ARGS__>::value
#endif

// </editor-fold> end Variable template compatibility }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Pre-C++14 constexpr"> {{{1

#if MDSPAN_IMPL_USE_CONSTEXPR_14
#  define MDSPAN_IMPL_CONSTEXPR_14 constexpr
// Workaround for a bug (I think?) in EDG frontends
#  ifdef __EDG__
#    define MDSPAN_IMPL_CONSTEXPR_14_DEFAULTED
#  else
#    define MDSPAN_IMPL_CONSTEXPR_14_DEFAULTED constexpr
#  endif
#else
#  define MDSPAN_IMPL_CONSTEXPR_14
#  define MDSPAN_IMPL_CONSTEXPR_14_DEFAULTED
#endif

// </editor-fold> end Pre-C++14 constexpr }}}1
//==============================================================================

#if MDSPAN_IMPL_USE_IF_CONSTEXPR_17
#  define MDSPAN_IMPL_IF_CONSTEXPR_17 constexpr
#else
#  define MDSPAN_IMPL_IF_CONSTEXPR_17
#endif
