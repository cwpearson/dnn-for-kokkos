#ifndef KOKKOSDNN_EMBEDDING_HPP
#define KOKKOSDNN_EMBEDDING_HPP

#include <Kokkos_Core.hpp>
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

}  // namespace KokkosDNN

#endif
