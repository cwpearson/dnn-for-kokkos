// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_SAFETENSORS_LOAD_HPP
#define KOKKOSDNN_SAFETENSORS_LOAD_HPP

// Kokkos-dependent safetensors device loader (header-only).
//
// Why split: stream_into() below is the only helper in the safetensors
// reader that needs Kokkos (HostSpace staging view + deep_copy). The parser
// itself (KokkosDNN_Safetensors.hpp: JSON scanner, Entry table, MappedFile)
// is plain POSIX + STL on purpose, so parser-only tests and tools can
// include it without paying for a Kokkos dependency. Include this header
// only in translation units that actually stream tensors to device.

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "KokkosDNN_Safetensors.hpp"

namespace KokkosDNN {
namespace Safetensors {

// Stream n elements from a mapping through a bounded HostSpace chunk with a
// single deep_copy (no 2x device residency).
template <class ExecSpace, class Scalar>
void stream_into(const char* base, size_t n, size_t elsize, Scalar* dev_ptr) {
  using HostV = Kokkos::View<Scalar*, Kokkos::HostSpace>;
  HostV host("safetensors_load", n);
  constexpr size_t kChunk = size_t{1} << 18;
  for (size_t o = 0; o < n;) {
    const size_t m = std::min(n - o, kChunk);
    convert_chunk<Scalar>(base + o * elsize, elsize, host.data() + o, m);
    o += m;
  }
  Kokkos::View<Scalar*, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      dev(dev_ptr, n);
  Kokkos::deep_copy(dev, host);
}

}  // namespace Safetensors
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_SAFETENSORS_LOAD_HPP
