// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_OCCUPANCY_HPP
#define KOKKOSDNN_OCCUPANCY_HPP

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace KokkosDNN {
namespace Impl {

// Cached driver occupancy probe.
//
// `TeamPolicy::team_size_max(f, ParallelForTag{})` is a *driver* query: on
// CUDA/HIP it walks the block-size ladder (1024, 992, ... down to 1) asking
// the occupancy calculator how many blocks fit per SM for each candidate,
// and Kokkos recomputes it from a freshly built `TeamPolicy(1, 1)` at every
// launch site. In batch-1 decode the identical probe is repeated hundreds of
// times per generated token (one per rmsnorm call, two per MoE layer, two per
// quantized expert GEMM), even though the answer can only change with:
//
//   * the execution-space instance (a different device/driver context),
//   * the functor *type* (its register / static shmem footprint),
//   * the per-team and per-thread scratch reservations of the policy.
//
// League size, team size and vector length are not part of the key because
// Kokkos's query does not read them (TeamPolicyInternal::team_size_max only
// consults the vector length and the scratch sizes); the probe policy is the
// canonical `TeamPolicy(1, 1)` that every call site already builds.
//
// Deliberately backend-agnostic: no #ifdef, the probe itself is still the
// portable Kokkos query, and on host backends it is a trivial `return 1`
// (cached, so it also costs nothing there). Callers must pass *every* scratch
// reservation the real policy will use, because the answer depends on them.
// The functor's value is not inspected either: do not use this helper for
// functors that carry scratch-space views whose footprint varies with the
// value (none of the current call sites do; they pass scratch explicitly).
//
// Thread safety: a mutex guards a small per-instantiation table. Probes run
// on the host, so this never touches device code. The static table is per
// (ExecSpace, Functor) instantiation, so a cache hit costs an
// `ExecSpace()` construction (a shared-pointer copy on GPU backends), one
// lock and a linear scan of a handful of entries -- orders of magnitude
// cheaper than re-running the driver query.
template <class ExecSpace, class Functor>
int cached_team_size_max(const Functor& f, std::size_t team_scratch_bytes = 0,
                         std::size_t thread_scratch_bytes = 0) {
  struct Entry {
    uint32_t instance_id;
    std::size_t team_scratch;
    std::size_t thread_scratch;
    int value;
  };
  static std::mutex mutex;
  static std::vector<Entry> entries;

  // ExecSpace::impl_instance_id() is the backend-agnostic identity Kokkos
  // itself uses for execution-space instances (see Kokkos_Profiling).
  const uint32_t instance_id = ExecSpace().impl_instance_id();

  std::lock_guard<std::mutex> lock(mutex);
  for (const Entry& e : entries) {
    if (e.instance_id == instance_id && e.team_scratch == team_scratch_bytes &&
        e.thread_scratch == thread_scratch_bytes)
      return e.value;
  }

  // Miss: run the real query once and remember it.
  Kokkos::TeamPolicy<ExecSpace> probe(1, 1);
  if (team_scratch_bytes != 0)
    probe.set_scratch_size(0, Kokkos::PerTeam(team_scratch_bytes));
  if (thread_scratch_bytes != 0)
    probe.set_scratch_size(0, Kokkos::PerThread(thread_scratch_bytes));
  const int value = probe.team_size_max(f, Kokkos::ParallelForTag{});
  entries.push_back(
      Entry{instance_id, team_scratch_bytes, thread_scratch_bytes, value});
  return value;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_OCCUPANCY_HPP
