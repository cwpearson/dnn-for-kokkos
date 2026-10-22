# KokkosDNN compiler warnings helper.
#
# Kokkos exposes Kokkos_ENABLE_COMPILER_WARNINGS, but those flags are
# deliberately NOT propagated downstream (see the "Not using
# COMPILER_SPECIFIC_FLAGS so the warning flags are not passed downstream"
# comment in kokkos/cmake/kokkos_arch.cmake): they only affect the Kokkos
# library build itself via CMAKE_CXX_FLAGS. There is therefore nothing to
# piggy-back on; KokkosDNN needs its own option.
#
# KokkosDNN_ENABLE_COMPILER_WARNINGS (OFF by default) enables a warning set
# mirroring Kokkos's COMMON_WARNINGS/GNU_WARNINGS lists, plus -Werror (MSVC
# /WX), so a clean build here means warnings-clean under the same bar
# Kokkos itself uses. Each flag is probed with check_cxx_compiler_flag so
# nvcc_wrapper/hipcc/AppleClang only receive flags they accept.
#
# Usage: include this module once (top-level CMakeLists does so), then call
#   kokkosdnn_enable_compiler_warnings(<target>)
# for every first-party target. INTERFACE libraries get INTERFACE scope so
# the flags propagate to consumers; anything else gets PRIVATE scope so
# third-party targets pulled in via FetchContent are unaffected.
include_guard(GLOBAL)

include(CheckCXXCompilerFlag)

# Probe once at include time; results are cached by CheckCXXCompilerFlag.
set(_KOKKOSDNN_WARNING_CANDIDATES
  -Wall
  -Wextra
  -Wextra-semi
  -Wunused-parameter
  -Wshadow
  -pedantic
  -Wsign-compare
  -Wtype-limits
  -Wuninitialized
  -Wsuggest-override
  -Wempty-body
  -Wignored-qualifiers
  -Wimplicit-fallthrough
)

set(KOKKOSDNN_SUPPORTED_WARNING_FLAGS "")
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  foreach(_flag IN LISTS _KOKKOSDNN_WARNING_CANDIDATES)
    string(MAKE_C_IDENTIFIER "${_flag}" _varname)
    string(REGEX REPLACE "^_" "" _varname "${_varname}")
    check_cxx_compiler_flag("${_flag}" "KOKKOSDNN_HAS_WARNING_${_varname}")
    if(KOKKOSDNN_HAS_WARNING_${_varname})
      list(APPEND KOKKOSDNN_SUPPORTED_WARNING_FLAGS "${_flag}")
    endif()
  endforeach()
  unset(_flag)
  unset(_varname)
endif()
unset(_KOKKOSDNN_WARNING_CANDIDATES)

# nvcc's extended-lambda machinery trips -Wextra-semi inside nvcc's own
# headers in every TU including Kokkos (same reason Kokkos drops it for
# the NVIDIA compiler ID). Omit it whenever the Kokkos backend compiles
# with nvcc (CUDA backend, where CMAKE_CXX_COMPILER is nvcc_wrapper).
if(Kokkos_ENABLE_CUDA)
  list(REMOVE_ITEM KOKKOSDNN_SUPPORTED_WARNING_FLAGS "-Wextra-semi")
endif()

function(kokkosdnn_enable_compiler_warnings target)
  if(NOT KokkosDNN_ENABLE_COMPILER_WARNINGS)
    return()
  endif()
  if(NOT TARGET ${target})
    message(WARNING "kokkosdnn_enable_compiler_warnings: no such target '${target}'")
    return()
  endif()
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    set(_scope INTERFACE)
  else()
    set(_scope PRIVATE)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${target} ${_scope} /W4 /WX)
  else()
    target_compile_options(${target} ${_scope}
      ${KOKKOSDNN_SUPPORTED_WARNING_FLAGS} -Werror)
  endif()
  unset(_type)
  unset(_scope)
endfunction()

# Project policy: warnings are enforced on first-party code only.
# Silence a third-party target completely (ALWAYS, independent of
# KokkosDNN_ENABLE_COMPILER_WARNINGS):
#   - compiled libraries get -w (/w on MSVC) privately, so their own
#     build cannot fail on warnings (e.g. googlebenchmark's own -Werror
#     plus unconditional -pedantic-errors, which hipcc trips over);
#   - their headers get SYSTEM treatment, so including them from our
#     warning-clean TUs cannot fail either (e.g. benchmark.h's
#     __COUNTER__ use under hipcc -pedantic).
# Pass the REAL target name (not an :: alias): ALIAS targets reject
# set_target_properties/target_compile_options.
function(kokkosdnn_suppress_third_party_warnings target)
  if(NOT TARGET ${target})
    return()
  endif()
  # Resolve an :: alias to the real target (set_target_properties and
  # target_compile_options reject aliases).
  get_target_property(_aliased ${target} ALIASED_TARGET)
  if(_aliased)
    set(target ${_aliased})
  endif()
  get_target_property(_type ${target} TYPE)
  # Prebuilt (IMPORTED) targets compile nothing here; only their headers
  # need SYSTEM treatment below.
  get_target_property(_imported ${target} IMPORTED)
  if(NOT _type STREQUAL "INTERFACE_LIBRARY" AND NOT _imported)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      target_compile_options(${target} PRIVATE /w)
    else()
      target_compile_options(${target} PRIVATE -w)
    endif()
  endif()
  # SYSTEM-mark every interface include dir in place (entries must also
  # list in INTERFACE_INCLUDE_DIRECTORIES to take effect, which they do
  # here since we copy that very list).
  get_target_property(_inc ${target} INTERFACE_INCLUDE_DIRECTORIES)
  if(_inc)
    get_target_property(_sys ${target} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
    if(_sys)
      list(APPEND _sys ${_inc})
    else()
      set(_sys ${_inc})
    endif()
    list(REMOVE_DUPLICATES _sys)
    set_target_properties(${target} PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_sys}")
    unset(_sys)
  endif()
  unset(_inc)
  unset(_type)
  unset(_imported)
  unset(_aliased)
endfunction()
