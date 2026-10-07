#ifndef KOKKOSDNN_TYPES_HPP
#define KOKKOSDNN_TYPES_HPP

#include <Kokkos_Core.hpp>

namespace KokkosDNN {

// FP32-only PoC (see CMake / README scope). File loaders convert
// BF16/FP16 safetensors weights to float on load.
using value_type = float;

// Row-major 2D tensors: [rows, cols]. This matches HF weight layout
// [out_features, in_features] and token-major activations [seq, dim].
template <class ExecSpace>
using Matrix = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
template <class ExecSpace>
using Vector = Kokkos::View<float*, ExecSpace>;
template <class ExecSpace>
using VectorInt = Kokkos::View<int*, ExecSpace>;

// Host mirrors (always HostSpace, LayoutRight for memcpy simplicity).
using HostMatrix = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using HostVector = Kokkos::View<float*, Kokkos::HostSpace>;

}  // namespace KokkosDNN

#endif
