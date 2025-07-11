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

#ifndef KOKKOS_CORE_HPP
#define KOKKOS_CORE_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_CORE
#endif

//----------------------------------------------------------------------------
// In the case windows.h is included before Kokkos_Core.hpp there might be
// errors due to the potentially defined macros with name "min" and "max" in
// windows.h. These collide with the use of "min" and "max" in names inside
// Kokkos. The macros will be redefined at the end of Kokkos_Core.hpp
#if defined(min)
#pragma push_macro("min")
#undef min
#define KOKKOS_IMPL_PUSH_MACRO_MIN
#endif
#if defined(max)
#pragma push_macro("max")
#undef max
#define KOKKOS_IMPL_PUSH_MACRO_MAX
#endif

//----------------------------------------------------------------------------
// Include the execution space header files for the enabled execution spaces.

#include <Kokkos_Core_fwd.hpp>

#include <KokkosCore_Config_DeclareBackend.hpp>

#include <Kokkos_Half.hpp>
#include <Kokkos_AnonymousSpace.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Clamp.hpp>
#include <Kokkos_MinMax.hpp>
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_MathematicalSpecialFunctions.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Swap.hpp>
#include <Kokkos_MemoryPool.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_View.hpp>
#include <Kokkos_Vectorization.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_hwloc.hpp>
#include <Kokkos_Timer.hpp>
#include <Kokkos_Tuners.hpp>
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
#include <Kokkos_TaskScheduler.hpp>
#endif
#include <Kokkos_Complex.hpp>
#include <Kokkos_CopyViews.hpp>
#include <impl/Kokkos_TeamMDPolicy.hpp>
#include <impl/Kokkos_InitializationSettings.hpp>
#include <functional>
#include <iosfwd>
#include <memory>
#include <vector>

//----------------------------------------------------------------------------

namespace Kokkos {

void initialize(int& argc, char* argv[]);

void initialize(
    InitializationSettings const& settings = InitializationSettings());

namespace Impl {

void pre_initialize(const InitializationSettings& settings);

void post_initialize(const InitializationSettings& settings);

void pre_finalize();

void post_finalize();

void declare_configuration_metadata(const std::string& category,
                                    const std::string& key,
                                    const std::string& value);

}  // namespace Impl

[[nodiscard]] bool is_initialized() noexcept;
[[nodiscard]] bool is_finalized() noexcept;

[[nodiscard]] int device_id() noexcept;
[[nodiscard]] int num_devices() noexcept;
[[nodiscard]] int num_threads() noexcept;

bool show_warnings() noexcept;
bool tune_internals() noexcept;

/** \brief  Finalize the spaces that were initialized via Kokkos::initialize */
void finalize();

/**
 * \brief Push a user-defined function to be called in
 *   Kokkos::finalize, before any Kokkos state is finalized.
 *
 * \warning Only call this after Kokkos::initialize, but before
 *   Kokkos::finalize.
 *
 * This function is the Kokkos analog to std::atexit.  If you call
 * this with a function f, then your function will get called when
 * Kokkos::finalize is called.  Specifically, it will be called BEFORE
 * Kokkos does any finalization.  This means that all execution
 * spaces, memory spaces, etc. that were initialized will still be
 * initialized when your function is called.
 *
 * Just like std::atexit, if you call push_finalize_hook in sequence
 * with multiple functions (f, g, h), Kokkos::finalize will call them
 * in reverse order (h, g, f), as if popping a stack.  Furthermore,
 * just like std::atexit, if any of your functions throws but does not
 * catch an exception, Kokkos::finalize will call std::terminate.
 */
void push_finalize_hook(std::function<void()> f);

void fence(const std::string& name /*= "Kokkos::fence: Unnamed Global Fence"*/);

/** \brief Print "Bill of Materials" */
void print_configuration(std::ostream& os, bool verbose = false);

}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

static inline void check_init_final([[maybe_unused]] char const* func_name) {
// FIXME_THREADS: Checking for calls to kokkos_malloc, kokkos_realloc,
// kokkos_free before initialize or after finalize is currently disabled
// for the Threads backend. Refer issue #7944.
#if !defined(KOKKOS_ENABLE_THREADS)
  if (is_finalized()) {
    std::stringstream ss;
    ss << "Kokkos ERROR: attempting to perform C-style memory management "
          "via ";
    ss << func_name << "() **after** Kokkos::finalize() was called\n";
    Kokkos::abort(ss.str().c_str());
  } else if (!is_initialized()) {
    std::stringstream ss;
    ss << "Kokkos ERROR: attempting to perform C-style memory management "
          "via ";
    ss << func_name << "() **before** Kokkos::initialize() was called\n";
    Kokkos::abort(ss.str().c_str());
  }
#endif
}

}  // namespace Impl

/* Allocate memory from a memory space.
 * The allocation is tracked in Kokkos memory tracking system, so
 * leaked memory can be identified.
 */
template <class Space = Kokkos::DefaultExecutionSpace::memory_space>
inline void* kokkos_malloc(const std::string& arg_alloc_label,
                           const size_t arg_alloc_size) {
  Impl::check_init_final("kokkos_malloc");
  using MemorySpace = typename Space::memory_space;
  return Impl::SharedAllocationRecord<MemorySpace>::allocate_tracked(
      MemorySpace(), arg_alloc_label, arg_alloc_size);
}

template <class Space = Kokkos::DefaultExecutionSpace::memory_space>
inline void* kokkos_malloc(const size_t arg_alloc_size) {
  Impl::check_init_final("kokkos_malloc");
  using MemorySpace = typename Space::memory_space;
  return Impl::SharedAllocationRecord<MemorySpace>::allocate_tracked(
      MemorySpace(), "no-label", arg_alloc_size);
}

template <class Space = Kokkos::DefaultExecutionSpace::memory_space>
inline void kokkos_free(void* arg_alloc) {
  Impl::check_init_final("kokkos_free");
  using MemorySpace = typename Space::memory_space;
  return Impl::SharedAllocationRecord<MemorySpace>::deallocate_tracked(
      arg_alloc);
}

template <class Space = Kokkos::DefaultExecutionSpace::memory_space>
inline void* kokkos_realloc(void* arg_alloc, const size_t arg_alloc_size) {
  Impl::check_init_final("kokkos_realloc");
  using MemorySpace = typename Space::memory_space;
  return Impl::SharedAllocationRecord<MemorySpace>::reallocate_tracked(
      arg_alloc, arg_alloc_size);
}

}  // namespace Kokkos

namespace Kokkos {

/** \brief  ScopeGuard
 *  Some user scope issues have been identified with some Kokkos::finalize
 * calls; ScopeGuard aims to correct these issues.
 *
 *  Two requirements for ScopeGuard:
 *     if Kokkos::is_initialized() in the constructor, don't call
 * Kokkos::initialize or Kokkos::finalize it is not copyable or assignable
 */
namespace Impl {

inline std::string scopeguard_correct_usage() {
  return std::string(
      "Do instead:\n"
      "  std::unique_ptr<Kokkos::ScopeGuard> guard =\n"
      "    !Kokkos::is_initialized() && !Kokkos::is_finalized()?\n"
      "    new ScopeGuard(argc,argv) : nullptr;\n");
}

inline std::string scopeguard_create_while_initialized_warning() {
  return std::string(
             "Kokkos Error: Creating a ScopeGuard while Kokkos is initialized "
             "is illegal.\n")
      .append(scopeguard_correct_usage());
}

inline std::string scopeguard_create_after_finalize_warning() {
  return std::string(
             "Kokkos Error: Creating a ScopeGuard after Kokkos was finalized "
             "is illegal.\n")
      .append(scopeguard_correct_usage());
}

inline std::string scopeguard_destruct_after_finalize_warning() {
  return std::string(
             "Kokkos Error: Destroying a ScopeGuard after Kokkos was finalized "
             "is illegal.\n")
      .append(scopeguard_correct_usage());
}

}  // namespace Impl

class KOKKOS_ATTRIBUTE_NODISCARD ScopeGuard {
 public:
  template <class... Args>
#if defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201907
  [[nodiscard]]
#endif
  ScopeGuard(Args&&... args) {
    if (is_initialized()) {
      Kokkos::abort(
          Impl::scopeguard_create_while_initialized_warning().c_str());
    }
    if (is_finalized()) {
      Kokkos::abort(Impl::scopeguard_create_after_finalize_warning().c_str());
    }
    initialize(static_cast<Args&&>(args)...);
  }

  ~ScopeGuard() {
    if (is_finalized()) {
      Kokkos::abort(Impl::scopeguard_destruct_after_finalize_warning().c_str());
    }
    finalize();
  }

  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard& operator=(ScopeGuard&&)      = delete;
  ScopeGuard(const ScopeGuard&)            = delete;
  ScopeGuard(ScopeGuard&&)                 = delete;
};

}  // namespace Kokkos

namespace Kokkos {
namespace Experimental {
namespace Impl {
// Customization point for backends. Default behavior is to return the passed in
// instance, ignoring weights
template <class ExecSpace, class T>
std::vector<ExecSpace> impl_partition_space(const ExecSpace& base_instance,
                                            const std::vector<T>& weights) {
  std::vector<ExecSpace> instances;
  instances.reserve(weights.size());
  std::generate_n(std::back_inserter(instances), weights.size(),
                  [&base_instance]() { return base_instance; });

  return instances;
}
}  // namespace Impl

// Partitioning an Execution Space
// Input:
//   - Base execution space
//   - integer arguments for relative weight, either input per weight or vector
//   of weights
// Ouput:
//   - Array (or vector) of execution spaces partitioned based on weights
template <class ExecSpace, class... Args>
std::array<ExecSpace, sizeof...(Args)> partition_space(
    ExecSpace const& base_instance, Args... args) {
  static_assert(is_execution_space<ExecSpace>::value,
                "Kokkos Error: partition_space expects an Execution Space as "
                "first argument");
  static_assert(
      (... && std::is_arithmetic_v<Args>),
      "Kokkos Error: partitioning arguments must be integers or floats");

  // Get vector of instances from backend specific impl
  std::vector<std::common_type_t<Args...>> weights = {args...};
  auto instances_vec = Impl::impl_partition_space(base_instance, weights);

  // Convert to std::array and return
  std::array<ExecSpace, sizeof...(Args)> instances;
  std::copy(instances_vec.begin(), instances_vec.end(), instances.begin());
  return instances;
}

template <class ExecSpace, class T>
std::vector<ExecSpace> partition_space(ExecSpace const& base_instance,
                                       std::vector<T> const& weights) {
  static_assert(is_execution_space<ExecSpace>::value,
                "Kokkos Error: partition_space expects an Execution Space as "
                "first argument");
  static_assert(
      std::is_arithmetic_v<T>,
      "Kokkos Error: partitioning arguments must be integers or floats");

  // Return vector of instances from backend specific impl
  return Impl::impl_partition_space(base_instance, weights);
}
}  // namespace Experimental
}  // namespace Kokkos

#include <Kokkos_Crs.hpp>
#include <Kokkos_WorkGraphPolicy.hpp>
#include <Kokkos_HyperGraph.hpp>
#include <Kokkos_HyperGraph_ModuleMapper.hpp>
// Including this in Kokkos_Parallel_Reduce.hpp led to a circular dependency
// because Kokkos::Sum is used in Kokkos_Combined_Reducer.hpp and the default.
// The real answer is to finally break up Kokkos_Partial_Reduce.hpp into
// smaller parts...
#include <impl/Kokkos_Combined_Reducer.hpp>
// Yet another workaround to deal with circular dependency issues because the
// implementation of the RAII wrapper is using Kokkos::single.
#include <Kokkos_AcquireUniqueTokenImpl.hpp>

// ECAN-style attention allocation for agentic kernels
#include <Kokkos_AttentionAllocation.hpp>
#include <Kokkos_AgenticKernel.hpp>
#include <Kokkos_AttentionAwareParallelImpl.hpp>

//----------------------------------------------------------------------------
// Redefinition of the macros min and max if we pushed them at entry of
// Kokkos_Core.hpp
#if defined(KOKKOS_IMPL_PUSH_MACRO_MIN)
#pragma pop_macro("min")
#undef KOKKOS_IMPL_PUSH_MACRO_MIN
#endif
#if defined(KOKKOS_IMPL_PUSH_MACRO_MAX)
#pragma pop_macro("max")
#undef KOKKOS_IMPL_PUSH_MACRO_MAX
#endif

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_CORE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_CORE
#endif
#endif
