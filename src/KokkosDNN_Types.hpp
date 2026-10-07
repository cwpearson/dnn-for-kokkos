#ifndef KOKKOSDNN_TYPES_HPP
#define KOKKOSDNN_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

namespace KokkosDNN {

// Scalar-agnostic PoC: core ops and the Llama example are templated on the
// scalar type. Supported: float and Kokkos::Experimental::bhalf_t (BF16,
// the Llama 3.3 8B checkpoint dtype). Accumulation inside kernels stays in
// FP32; the cuBLAS GEMM path covers both scalars, other TPL paths (cuDNN)
// are float-only and fall back to the native Kokkos kernel otherwise.
using value_type = float;
using bhalf_t = Kokkos::Experimental::bhalf_t;
using half_t = Kokkos::Experimental::half_t;

// Row-major 2D tensors: [rows, cols]. This matches HF weight layout
// [out_features, in_features] and token-major activations [seq, dim].
template <class ExecSpace, class Scalar = float>
using Matrix = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
template <class ExecSpace, class Scalar = float>
using Vector = Kokkos::View<Scalar*, ExecSpace>;
template <class ExecSpace>
using VectorInt = Kokkos::View<int*, ExecSpace>;

// Host mirrors (always HostSpace, LayoutRight for memcpy simplicity).
template <class Scalar = float>
using HostMatrix =
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace>;
template <class Scalar = float>
using HostVector = Kokkos::View<Scalar*, Kokkos::HostSpace>;

}  // namespace KokkosDNN

#endif
