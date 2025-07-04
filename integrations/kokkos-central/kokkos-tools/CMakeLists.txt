cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(KokkosTools CXX)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_CXX_STANDARD LESS 17)
  message(FATAL_ERROR "KokkosTools requires C++17")
endif()


# Include utilities
include(cmake/utils.cmake)
include(cmake/configure_tpls.cmake)

# Set policies
cmake_policy(SET CMP0111 NEW) # error if library not found

# Disable in-source builds to prevent source tree corruption.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "FATAL: In-source builds are not allowed. You should create a separate directory for build files.")
endif()

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)

message(STATUS)
message(STATUS Configuring Kokkos-Tools)
message(STATUS)

# Common settings
set(BUILD_SHARED_LIBS "Build shared libraries" ON)
if(WIN32)
  set(BUILD_SHARED_LIBS OFF) # We need to add __declspec(dllexport/dllimport) for Windows DLLs
endif()

# Tools settings
option(KokkosTools_ENABLE_SINGLE  "Build single library interfacing all profilers and dispatching at runtime" OFF)
if(WIN32)
  set(KokkosTools_ENABLE_SINGLE ON)
endif()

option(KokkosTools_ENABLE_PAPI    "Enable PAPI support"             OFF)
option(KokkosTools_ENABLE_MPI     "Enable MPI support"              OFF)
option(KokkosTools_ENABLE_CALIPER "Enable building Caliper library" OFF)
option(KokkosTools_ENABLE_APEX    "Enable building Apex library"    OFF)
option(KokkosTools_ENABLE_EXAMPLES "Build examples"                 OFF)
option(KokkosTools_ENABLE_TESTS    "Build tests"                    OFF)

# Fetch Kokkos options:
acquire_kokkos_config()
if(DEFINED Kokkos_FOUND_MSG)
  message(STATUS "${Kokkos_FOUND_MSG}: ${Kokkos_INSTALL_DIR}\n"
    "\t\tDevices: ${Kokkos_DEVICES}\n"
    "\t\tArchitecture: ${Kokkos_ARCH}\n"
    "\t\tTPLs: ${Kokkos_TPLS}\n"
    "\t\tCompiler: ${Kokkos_CXX_COMPILER} (${Kokkos_CXX_COMPILER_ID})\n"
    "\t\tCMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}\n"
    "\t\tOptions: ${Kokkos_OPTIONS}")
else()
  message(STATUS "Kokkos NOT found")
endif()

# Libraries
if(KokkosTools_ENABLE_PAPI)
  find_package(PAPI REQUIRED) # TODO: papi-connector requires v6.0 or newer
  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.20")
    cmake_path(GET PAPI_INCLUDE_DIR PARENT_PATH PAPI_ROOT)
  else()
    get_filename_component(PAPI_ROOT ${PAPI_INCLUDE_DIR} DIRECTORY)
  endif()
  message(STATUS "Found PAPI ${PAPI_VERSION_STRING} at ${PAPI_ROOT}")
  set(KokkosTools_HAS_PAPI ON)
else()
  message(STATUS "PAPI support disabled")
  set(KokkosTools_HAS_PAPI OFF)
endif()

if(KokkosTools_ENABLE_MPI)
  find_package(MPI REQUIRED)
  message(STATUS "Found MPI ${MPI_CXX_VERSION}: ${MPI_CXX_LIBRARIES}")
  set(KOKKOSTOOLS_HAS_MPI 1)
else()
  message(STATUS "MPI not available. MPI disabled.")
  set(KOKKOSTOOLS_HAS_MPI 0)
endif()

include(cmake/configure_variorum.cmake)

set(KOKKOSTOOLS_HAS_CALIPER ${KokkosTools_ENABLE_CALIPER})
set(KOKKOSTOOLS_HAS_NVTX  ${Kokkos_ENABLE_CUDA}) # we assume that enabling CUDA for Kokkos program means nvtx should be available
set(KOKKOSTOOLS_HAS_ROCTX ${Kokkos_ENABLE_HIP})  # we assume that enabling HIP for Kokkos program means roctx should be available

if(DEFINED ENV{VTUNE_HOME})
  set(VTune_ROOT $ENV{VTUNE_HOME})
endif()
if(VTune_ROOT)
  find_package(ITT REQUIRED)
  set(KOKKOSTOOLS_HAS_VTUNE ON)
else()
  message(WARNING "Set VTUNE_HOME in environment or VTune_ROOT in build options to build VTune connectors")
  set(VTune_ROOT "" CACHE STRING "Path to VTune Intel compiler")
  set(KOKKOSTOOLS_HAS_VTUNE OFF)
endif()

# make Kokkos profiling interface available for native profilers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/profiling/all)

set(COMMON_HEADERS_PATH ${CMAKE_CURRENT_BINARY_DIR}/common)
include_directories(${COMMON_HEADERS_PATH})

# Allow all tools to include any file.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

set(SINGLELIB_PROFILERS "" CACHE STRING "" FORCE)

# Export settings
include(GNUInstallDirs)
set(EXPORT_NAME KokkosToolsConfig)
set(EXPORT_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(EXPORT_LIB_DIR ${CMAKE_INSTALL_LIBDIR})
set(EXPORT_TARGETS "" CACHE STRING "" FORCE)

if(WIN32)
  message(STATUS "Windows target detected - skipping Unix-only tools.")
endif()

if(APPLE)
  message(STATUS "Apple OSX target detected.")
endif()

# Utilities
if(NOT WIN32)
  add_subdirectory(common/kernel-filter)
  add_subdirectory(common/kokkos-sampler)
endif()
add_subdirectory(debugging/kernel-logger)

# Profilers
if(NOT WIN32)
  add_subdirectory(profiling/simple-kernel-timer)
  add_subdirectory(profiling/memory-hwm)
  if(KokkosTools_ENABLE_MPI)
    add_subdirectory(profiling/memory-hwm-mpi)
  else()
    message(STATUS "Skipping memory-hwm-mpi (MPI disabled)")
  endif()
  add_subdirectory(profiling/memory-events)
  add_subdirectory(profiling/memory-usage)
  add_subdirectory(profiling/chrome-tracing)
  add_subdirectory(profiling/space-time-stack)
  add_subdirectory(profiling/perfetto-connector)
endif()

# External lib connectors
if(KokkosTools_ENABLE_PAPI)
  add_subdirectory(profiling/papi-connector)
endif()

if(NOT WIN32 AND NOT APPLE)
  find_program(KOKKOSTOOLS_DTRACE_EXECUTABLE dtrace)
  if(KOKKOSTOOLS_DTRACE_EXECUTABLE)
    add_subdirectory(profiling/systemtap-connector)
    set(KOKKOSTOOLS_HAS_SYSTEMTAP ON)
  else()
    message(STATUS "Skipping systemtap-connector (dtrace executable wasn't found)")
    set(KOKKOSTOOLS_HAS_SYSTEMTAP OFF)
  endif()
else()
  set(KOKKOSTOOLS_HAS_SYSTEMTAP OFF)
endif()

if(KOKKOSTOOLS_HAS_VARIORUM)
  add_subdirectory(profiling/variorum-connector)
endif()

# GPU profilers
if(Kokkos_ENABLE_CUDA)
  add_subdirectory(profiling/nvtx-connector)
  add_subdirectory(profiling/nvtx-focused-connector)
endif()
if(Kokkos_ENABLE_HIP)
  add_subdirectory(profiling/roctx-connector)
endif()

if(KOKKOSTOOLS_HAS_VTUNE)
  add_subdirectory(profiling/vtune-connector)
  add_subdirectory(profiling/vtune-focused-connector)
endif()

# Find or build Caliper
if(KokkosTools_ENABLE_CALIPER)
  find_package(caliper QUIET)
  if(caliper_INCLUDE_DIR)
    cmake_path(GET caliper_INCLUDE_DIR PARENT_PATH Caliper_INSTALL_DIR)
    file(REAL_PATH ${Caliper_INSTALL_DIR} Caliper_INSTALL_DIR)
    message(STATUS "Caliper installation found in: ${Caliper_INSTALL_DIR}")
   list(APPEND SINGLELIB_PROFILERS caliper)
  else()
  # Don't support git submodules for Caliper. The Kokkos tools user has can try installing Apex and linking on their own if they don't have it.
    message(FATAL_ERROR "FATAL: Required Caliper installation not found! Exiting.")
  endif()
endif()

# Find or build Apex
if(KokkosTools_ENABLE_APEX)
  find_package(Apex QUIET)
  if(Apex_FOUND)
    message(STATUS "Apex installation found in: ${Apex_DIR}")
  list(APPEND SINGLELIB_PROFILERS "apex")
  else()
      include(cmake/AddGitSubmodule.cmake)
      # Build Binutils as part of APEX
      set(APEX_WITH_BFD TRUE CACHE BOOL "Include Binutils support for APEX")
      set(APEX_BUILD_BFD TRUE CACHE BOOL "Build Binutils support for APEX")
      # Build Tuning support as part of APEX
      set(APEX_WITH_PLUGINS TRUE CACHE BOOL "Include Plugins support for APEX")
      set(APEX_WITH_ACTIVEHARMONY TRUE CACHE BOOL "Include Active Harmony support for APEX")
      set(APEX_BUILD_ACTIVEHARMONY TRUE CACHE BOOL "Build Active Harmony support for APEX")
      # Include MPI support as part of APEX
      set(APEX_WITH_MPI ${KokkosTools_ENABLE_MPI} CACHE BOOL "Include MPI support for APEX")
      # Include Device-specific support as part of APEX
      if(Kokkos_DEVICES MATCHES "CUDA")
        set(APEX_WITH_CUDA TRUE CACHE BOOL "Include CUDA support for APEX")
      endif(Kokkos_DEVICES MATCHES "CUDA")
      if(Kokkos_DEVICES MATCHES "HIP")
        set(APEX_WITH_HIP TRUE CACHE BOOL "Include HIP support for APEX")
      endif(Kokkos_DEVICES MATCHES "HIP")
      if(Kokkos_DEVICES MATCHES "SYCL")
        set(APEX_WITH_LEVEL0 TRUE CACHE BOOL "Include LEVEL0 support for APEX")
      endif(Kokkos_DEVICES MATCHES "SYCL")
      # Add openmp support as long as the compiler supports OMPT - definitely not GNU though.
      if(Kokkos_DEVICES MATCHES "OPENMP")
        if ("${CMAKE_CXX_COMPILER_ID}" NOT STREQUAL "GNU")
          set(APEX_WITH_OMPT TRUE CACHE BOOL "Include OpenMP support for APEX")
        endif ("${CMAKE_CXX_COMPILER_ID}" NOT STREQUAL "GNU")
      endif(Kokkos_DEVICES MATCHES "OPENMP")
      add_git_submodule(tpls/apex)
  endif()
endif()

# Config file
configure_file(common/kp_config.hpp.in common/kp_config.hpp)

# Build single library interface (once we have everything set up)
if(KokkosTools_ENABLE_SINGLE)
  message(STATUS "Building Monolithic KokkosTools library with profilers: ${SINGLELIB_PROFILERS}")
  add_subdirectory(profiling/all)
else()
  message(STATUS "Monolithic KokkosTools library skipped")
endif()

# Build examples
if(KokkosTools_ENABLE_EXAMPLES)
  if(NOT KokkosTools_ENABLE_SINGLE)
    message(WARNING "This example requires KokkosTools built with monolothic library interface (KokkosTools_ENABLE_SINGLE=ON)")
  else()
    enable_testing()
    add_subdirectory(example)
  endif()
endif()

# Tests
if(KokkosTools_ENABLE_TESTS)
  enable_testing()
  include(cmake/BuildGTest.cmake)
  add_subdirectory(tests)
endif()

# Install exports
install(TARGETS ${EXPORT_TARGETS} EXPORT ${EXPORT_NAME})
install(EXPORT ${EXPORT_NAME}
  NAMESPACE KokkosTools::
  DESTINATION ${EXPORT_LIB_DIR}/cmake)
install(CODE "SET(KokkosTools_HAS_MPI ${USE_MPI})")

