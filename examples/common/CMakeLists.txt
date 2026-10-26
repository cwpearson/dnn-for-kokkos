# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Shared example-REPL line reader (KokkosDNN::example_readline).
#
# Wraps GNU readline (editing + history + prefill) when the headers and
# library are found, and falls back to std::getline otherwise so every
# example configures and runs without new mandatory dependencies.
#
# NOTE (license): linking GNU readline pulls GPLv3 obligations into the
# resulting example binaries. Pass -DKokkosDNN_ENABLE_READLINE=OFF to force
# the dependency-free std::getline fallback.
option(KokkosDNN_ENABLE_READLINE "Use GNU readline for example REPLs when available" ON)

set(_KOKKOSDNN_HAVE_READLINE OFF)
if(KokkosDNN_ENABLE_READLINE)
  find_path(KOKKOSDNN_READLINE_INCLUDE_DIR readline/readline.h QUIET)
  find_library(KOKKOSDNN_READLINE_LIBRARY readline QUIET)
  if(KOKKOSDNN_READLINE_INCLUDE_DIR AND KOKKOSDNN_READLINE_LIBRARY)
    set(_KOKKOSDNN_HAVE_READLINE ON)
  endif()
endif()

add_library(kokkosdnn_example_readline STATIC KokkosDNN_ExampleReadline.cpp)
add_library(KokkosDNN::example_readline ALIAS kokkosdnn_example_readline)
target_include_directories(kokkosdnn_example_readline PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
if(_KOKKOSDNN_HAVE_READLINE)
  message(STATUS "KokkosDNN examples: GNU readline enabled (${KOKKOSDNN_READLINE_LIBRARY})")
  target_compile_definitions(kokkosdnn_example_readline PRIVATE KOKKOSDNN_EXAMPLE_HAVE_READLINE=1)
  target_include_directories(kokkosdnn_example_readline SYSTEM PRIVATE ${KOKKOSDNN_READLINE_INCLUDE_DIR})
  target_link_libraries(kokkosdnn_example_readline PUBLIC ${KOKKOSDNN_READLINE_LIBRARY})
else()
  message(STATUS "KokkosDNN examples: GNU readline not found or disabled; REPL uses std::getline fallback")
endif()
set_target_properties(kokkosdnn_example_readline PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(kokkosdnn_example_readline)
