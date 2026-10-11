#ifndef KOKKOSDNN_EMBEDDING_HPP
#define KOKKOSDNN_EMBEDDING_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// Gather rows: out[t,:] = table[token[t],:]. Scalar-agnostic.
template <class ExecSpace, class Scalar>
void embedding_gather(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& table,
    const Kokkos::View<int*, ExecSpace>& tokens,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::embed") {
  const size_t T = tokens.extent(0), D = table.extent(1);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) { out(t, d) = table(tokens(t), d); });
}

// Scaled gather (Gemma-style embeddings): out[t,:] = scale * table[token[t],:].
// The scale (e.g. sqrt(hidden_size)) is applied in FP32 and cast back to
// Scalar, so the op is scalar-agnostic (float, bhalf_t). In-place N/A
// (shapes differ); throws on dimension mismatch.
template <class ExecSpace, class Scalar>
void embedding_gather_scaled(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& table,
    const Kokkos::View<int*, ExecSpace>& tokens, float scale,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::embed_scaled") {
  const size_t T = tokens.extent(0), D = table.extent(1);
  if (out.extent(0) != T || out.extent(1) != D)
    throw std::invalid_argument("embedding_gather_scaled: dimension mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        out(t, d) = static_cast<Scalar>(scale *
                                        static_cast<float>(table(tokens(t), d)));
      });
}

}  // namespace KokkosDNN

#endif
