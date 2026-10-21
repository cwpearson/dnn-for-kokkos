#ifndef KOKKOSDNN_KOKORO_WEIGHTS_HPP
#define KOKKOSDNN_KOKORO_WEIGHTS_HPP

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <map>
#include <string>

#include "KokkosDNN_KokoroConfig.hpp"

namespace KokkosDNN {
namespace Kokoro {

using DefaultScalar = float;

// Generic Kokoro weights: device views keyed by loader name.
// Loader keys are "section.key" with the DataParallel "module." prefix
// stripped, and weight_norm pairs ("*.weight_g" + "*.weight_v") fused into a
// single "*.weight" dense tensor. Ranks:
//   v1: biases, norms ([N]).
//   v2: linears [Out, In], embeddings [Vocab, Dim].
//   v3: convs [Out, In, K] (fused dense) and alphas [1, C, 1].
template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  Kokkos::View<Scalar*, ExecSpace> v1(const std::string& n) const {
    auto it = v1_.find(n);
    if (it == v1_.end())
      throw std::runtime_error("Kokoro weights: missing v1 '" + n + "'");
    return it->second;
  }
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> v2(
      const std::string& n) const {
    auto it = v2_.find(n);
    if (it == v2_.end())
      throw std::runtime_error("Kokoro weights: missing v2 '" + n + "'");
    return it->second;
  }
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> v3(
      const std::string& n) const {
    auto it = v3_.find(n);
    if (it == v3_.end())
      throw std::runtime_error("Kokoro weights: missing v3 '" + n + "'");
    return it->second;
  }
  bool has_v1(const std::string& n) const { return v1_.count(n) != 0; }
  bool has_v2(const std::string& n) const { return v2_.count(n) != 0; }
  bool has_v3(const std::string& n) const { return v3_.count(n) != 0; }
  std::map<std::string, Kokkos::View<Scalar*, ExecSpace>> v1_;
  std::map<std::string,
           Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>>
      v2_;
  std::map<std::string,
           Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>>
      v3_;
};

// Fill with small deterministic values for shape/smoke tests (no downloads).
template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Load kokoro-v1_0.pth into device views (dependency-free ZIP/pickle reader,
// weight_norm fused on host). Throws on missing/mismatched tensors.
template <class ExecSpace, class Scalar = DefaultScalar>
void load_kokoro_pth(const Config& cfg, const std::string& pth_path,
                      Weights<ExecSpace, Scalar>& w);

// Voice pack: [N, 1, 256] -> row idx (clamped) as [256] device vector.
template <class ExecSpace, class Scalar = DefaultScalar>
void load_voice_row(const std::string& voice_path, int idx,
                    Kokkos::View<Scalar*, ExecSpace>& ref_s);

}  // namespace Kokoro
}  // namespace KokkosDNN

#endif
