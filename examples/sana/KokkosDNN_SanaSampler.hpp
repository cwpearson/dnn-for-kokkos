// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_SANA_SAMPLER_HPP
#define KOKKOSDNN_SANA_SAMPLER_HPP

// Flow-matching Euler sampler with classifier-free guidance for the Sana
// example (NVlabs/Sana `FlowEuler` over diffusers'
// `FlowMatchEulerDiscreteScheduler`, Apache-2.0/MIT; yaml scheduler block:
// predict_flow_v=true, noise_schedule=linear_flow, flow_shift=4.0).
//
// Reference math (non-dynamic shifting, no karras/exponential/beta sigmas,
// deterministic Euler, no stochastic sampling):
//   train extremes : sigma_max = shift*1/(1+(shift-1)*1) = 1,
//                    sigma_min = shift*0.001/(1+(shift-1)*0.001)
//   inference      : u = linspace(1000*sigma_max, 1000*sigma_min, steps)/1000
//                    sig[i] = shift*u[i]/(1+(shift-1)*u[i]), sig[steps] = 0
//                    t[i] = 1000*sig[i]            (DiT timestep input)
//   loop           : v = DiT(x, t[i], cond/uncond); if guidance > 1,
//                    v = uncond + guidance*(cond - uncond)   (FlowEuler CFG)
//                    x += (sig[i+1] - sig[i]) * v            (Euler step)
//   init           : x_T ~ N(0,1), seeded std::mt19937 + Box-Muller
//                    (bit-portable: no std::normal_distribution, whose
//                    mapping is implementation-defined).
//
// Perturbed-attention guidance (PAG) is intentionally unsupported: Sana's
// FlowEuler path runs plain CFG (apg=None); PAG only exists on the DPM
// code path. run_flow_euler() therefore takes no pag_scale.
//
// All helpers are host-side (std::vector<float>) except run_flow_euler(),
// which bridges host latents/captions to the device Model::forward().
// Without guidance it runs one B=1 forward per step; with active CFG
// (guidance > 1 and cond != uncond) it batches both branches into one B=2
// forward per step (row 0 = cond, row 1 = uncond).

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_SanaConfig.hpp"
#include "KokkosDNN_SanaModel.hpp"

namespace KokkosDNN {
namespace Sana {

struct FlowEulerConfig {
  int steps = 18;               // inference steps (Sana default 18-20)
  float guidance_scale = 5.0f;  // CFG scale (> 1 enables guidance)
  float flow_shift = 4.0f;      // yaml scheduler flow_shift
  uint64_t seed = 42;           // latent init seed

  void validate() const {
    if (steps <= 0)
      throw std::invalid_argument("Sana FlowEuler: steps must be > 0");
    if (!(guidance_scale > 0.0f) || !std::isfinite(guidance_scale))
      throw std::invalid_argument(
          "Sana FlowEuler: guidance_scale must be positive and finite");
    if (!(flow_shift >= 1.0f) || !std::isfinite(flow_shift))
      throw std::invalid_argument(
          "Sana FlowEuler: flow_shift must be >= 1 and finite");
  }
};

namespace Impl {

// Train-schedule shifted sigma: shift*s/(1+(shift-1)*s).
inline float shift_sigma(float s, float shift) {
  return shift * s / (1.0f + (shift - 1.0f) * s);
}

// Portable uniform in [0,1) from explicit mt19937 output bits.
inline double mt_uniform01(std::mt19937& eng) {
  const uint64_t hi = eng(), lo = eng();
  const uint64_t v = ((hi << 32) | lo) >> 11;
  return (double)v * (1.0 / 9007199254740992.0);  // 2^53
}

}  // namespace Impl

// Sigma schedule, size steps+1: shifted linspace from 1 down, terminal 0.
inline std::vector<float> build_sigmas(int steps, float shift) {
  if (steps <= 0)
    throw std::invalid_argument("Sana build_sigmas: steps must be > 0");
  if (!(shift >= 1.0f) || !std::isfinite(shift))
    throw std::invalid_argument("Sana build_sigmas: shift must be >= 1");
  const float sigma_max = Impl::shift_sigma(1.0f, shift);    // == 1
  const float sigma_min = Impl::shift_sigma(0.001f, shift);  // train floor
  std::vector<float> sig((size_t)steps + 1);
  for (int i = 0; i < steps; ++i) {
    const float u =
        (steps == 1)
            ? sigma_max
            : sigma_max + (sigma_min - sigma_max) * ((float)i / (float)(steps - 1));
    sig[(size_t)i] = Impl::shift_sigma(u, shift);
  }
  sig[(size_t)steps] = 0.0f;
  return sig;
}

// DiT timestep inputs (FlowMatchEulerDiscreteScheduler timesteps = 1000*sig).
inline std::vector<float> build_model_timesteps(
    const std::vector<float>& sigmas) {
  std::vector<float> t(sigmas.size());
  for (size_t i = 0; i < sigmas.size(); ++i) t[i] = 1000.0f * sigmas[i];
  return t;
}

// Classifier-free guidance: out = uncond + scale*(cond - uncond).
inline void guided_predict(const std::vector<float>& cond,
                           const std::vector<float>& uncond, float scale,
                           std::vector<float>& out) {
  if (cond.size() != uncond.size())
    throw std::invalid_argument("Sana guided_predict: size mismatch");
  out.resize(cond.size());
  for (size_t i = 0; i < cond.size(); ++i)
    out[i] = uncond[i] + scale * (cond[i] - uncond[i]);
}

// Euler update: x += dt * v.
inline void euler_step(std::vector<float>& x, const std::vector<float>& v,
                       float dt) {
  if (x.size() != v.size())
    throw std::invalid_argument("Sana euler_step: size mismatch");
  for (size_t i = 0; i < x.size(); ++i) x[i] += dt * v[i];
}

// Seeded N(0,1) latents (Box-Muller over portable mt19937 bits).
inline std::vector<float> sample_gaussian(size_t n, uint64_t seed) {
  std::mt19937 eng((uint32_t)seed);
  // SplitMix-style: also mix the high word so close seeds decorrelate.
  eng.seed((uint32_t)seed ^ (uint32_t)(seed >> 32) ^ 0x9e3779b9u);
  std::vector<float> out(n);
  for (size_t i = 0; i < n; i += 2) {
    double u1 = 0.0;
    while (u1 <= 0.0) u1 = Impl::mt_uniform01(eng);
    const double u2 = Impl::mt_uniform01(eng);
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double a = 6.28318530717958647692 * u2;
    out[i] = (float)(r * std::cos(a));
    if (i + 1 < n) out[i + 1] = (float)(r * std::sin(a));
  }
  return out;
}

// Full FlowEuler loop around a Sana Model (B=1).
//
//   x_init : [C,H,W] initial latents (row-major C,H,W).
//   cond   : [L,Cc] conditional caption features (row-major).
//   uncond : [L,Cc] unconditional caption features (used iff guidance > 1;
//            pass {} to reuse cond).
// Returns the denoised [C,H,W] latents. on_step(i, t, sigma) observes each
// step (may be empty).
template <class ExecSpace, class Scalar>
std::vector<float> run_flow_euler(
    Model<ExecSpace, Scalar>& model, const Config& cfg,
    const std::vector<float>& x_init, const std::vector<float>& cond,
    const std::vector<float>& uncond, const FlowEulerConfig& s, int latent_h,
    int latent_w,
    const std::function<void(int, float, float)>& on_step = {}) {
  cfg.validate();
  s.validate();
  const size_t C = (size_t)cfg.in_channels;
  const size_t Cc = (size_t)cfg.caption_channels;
  const size_t H = (size_t)latent_h, W = (size_t)latent_w;
  if (H == 0 || W == 0)
    throw std::invalid_argument("Sana run_flow_euler: bad latent geometry");
  const size_t T = H * W;
  if (x_init.size() != C * H * W)
    throw std::invalid_argument("Sana run_flow_euler: x_init size mismatch");
  if (cond.size() % Cc != 0 || cond.empty())
    throw std::invalid_argument("Sana run_flow_euler: cond size mismatch");
  const size_t L = cond.size() / Cc;
  if (L > (size_t)cfg.model_max_length)
    throw std::invalid_argument("Sana run_flow_euler: caption too long");
  const bool do_cfg = s.guidance_scale > 1.0f;
  const std::vector<float>& unc = uncond.empty() ? cond : uncond;
  if (unc.size() != cond.size())
    throw std::invalid_argument("Sana run_flow_euler: uncond size mismatch");
  // Identical cond/uncond make guidance a no-op: run a single branch.
  bool same = (unc.data() == cond.data()) || (unc == cond);

  using LatR = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>;
  using TokR = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using TVec = Kokkos::View<float*, ExecSpace>;
  const bool need_cfg = do_cfg && !same;
  const size_t Oc = (size_t)cfg.out_channels;
  LatR x_d("sana_euler_x", 1, C, H, W);
  // Batched-CFG buffers (guidance path): one B=2 forward per step with
  // row 0 = cond, row 1 = uncond. y_b is filled once (captions are static);
  // x_b/t_b are refreshed per step (x evolves, t changes). v_b holds both
  // branch predictions. L is shared by construction (unc.size() ==
  // cond.size() is checked above).
  LatR x_b, v_b;
  TokR y_b;
  TVec t_b;
  // Single-branch buffers (guidance-disabled / cond == uncond fast path).
  LatR vc_d;
  TVec t_d;
  TokR y_cond_d;
  if (need_cfg) {
    x_b = LatR("sana_euler_xb", 2, C, H, W);
    v_b = LatR("sana_euler_vb", 2, Oc, H, W);
    y_b = TokR("sana_euler_yb", 2, L, Cc);
    t_b = TVec("sana_euler_tb", 2);
  } else {
    // Prediction buffer: a single conditional branch, matching FlowEuler's
    // guidance-disabled path.
    vc_d = LatR("sana_euler_vc", 1, Oc, H, W);
    t_d = TVec("sana_euler_t", 1);
    y_cond_d = TokR("sana_euler_y_cond", 1, L, Cc);
  }
  // Device residency: copy evolving state + captions once, keep x on device
  // across steps (only the scalar timestep crosses host->device per step).
  {
    auto xh = Kokkos::create_mirror_view(x_d);
    size_t k = 0;
    for (size_t c = 0; c < C; ++c)
      for (size_t h = 0; h < H; ++h)
        for (size_t w = 0; w < W; ++w) xh(0, c, h, w) = (Scalar)x_init[k++];
    Kokkos::deep_copy(x_d, xh);
  }
  auto fill_caption = [](TokR& dst, const std::vector<float>& y, size_t Lc,
                          size_t Cc_) {
    auto yh = Kokkos::create_mirror_view(dst);
    size_t k = 0;
    for (size_t l = 0; l < Lc; ++l)
      for (size_t c = 0; c < Cc_; ++c) yh(0, l, c) = (Scalar)y[k++];
    Kokkos::deep_copy(dst, yh);
  };
  if (need_cfg) {
    // Row 0 = cond, row 1 = uncond (must match the CFG combine below,
    // which reads vc = v_b[0] as cond and vu = v_b[1] as uncond).
    auto yh = Kokkos::create_mirror_view(y_b);
    size_t k = 0;
    for (size_t l = 0; l < L; ++l)
      for (size_t c = 0; c < Cc; ++c) yh(0, l, c) = (Scalar)cond[k++];
    k = 0;
    for (size_t l = 0; l < L; ++l)
      for (size_t c = 0; c < Cc; ++c) yh(1, l, c) = (Scalar)unc[k++];
    Kokkos::deep_copy(y_b, yh);
  } else {
    fill_caption(y_cond_d, cond, L, Cc);
  }

  const std::vector<float> sigmas = build_sigmas(s.steps, s.flow_shift);
  const std::vector<float> tvals = build_model_timesteps(sigmas);
  const float g = s.guidance_scale;
  if (need_cfg) {
    // CFG-active path: a single batched B=2 forward per step (cond row 0,
    // uncond row 1), halving launch/host overhead and doubling GEMM rows
    // versus two sequential B=1 forwards.
    auto tb_h = Kokkos::create_mirror_view(t_b);
    for (int i = 0; i < s.steps; ++i) {
      if (on_step) on_step(i, tvals[(size_t)i], sigmas[(size_t)i]);
      // Scalar timestep only: two floats host->device per step.
      tb_h(0) = tvals[(size_t)i];
      tb_h(1) = tvals[(size_t)i];
      Kokkos::deep_copy(t_b, tb_h);
      const float dt = sigmas[(size_t)i + 1] - sigmas[(size_t)i];
      // Refresh the B=2 input from the evolving device state (on-device;
      // x changes every step so this cannot be hoisted).
      Kokkos::parallel_for(
          "sana_euler_dup_x",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)2, C, H, W}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            x_b(b, c, h, w) = x_d(0, c, h, w);
          });
      Kokkos::fence();
      model.forward(x_b, t_b, y_b, v_b);
      Kokkos::fence();
      // Fused CFG combine + Euler update, matching host order/precision:
      //   v = uncond + g*(cond - uncond); x += dt*v   (all FP32),
      // with vc = v_b[0] (cond), vu = v_b[1] (uncond).
      Kokkos::parallel_for(
          "sana_euler_cfg_step",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)1, C, H, W}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            const float xc = static_cast<float>(x_d(b, c, h, w));
            const float vcond = static_cast<float>(v_b(0, c, h, w));
            const float vun = static_cast<float>(v_b(1, c, h, w));
            const float v = vun + g * (vcond - vun);
            x_d(b, c, h, w) = static_cast<Scalar>(xc + dt * v);
          });
      Kokkos::fence();
    }
  } else {
    auto t_h = Kokkos::create_mirror_view(t_d);
    for (int i = 0; i < s.steps; ++i) {
      if (on_step) on_step(i, tvals[(size_t)i], sigmas[(size_t)i]);
      // Scalar timestep only: single float host->device per step.
      t_h(0) = tvals[(size_t)i];
      Kokkos::deep_copy(t_d, t_h);
      const float dt = sigmas[(size_t)i + 1] - sigmas[(size_t)i];
      // No guidance (or cond == uncond): a single conditional branch,
      // matching FlowEuler's guidance-disabled path.
      model.forward(x_d, t_d, y_cond_d, vc_d);
      Kokkos::fence();
      Kokkos::parallel_for(
          "sana_euler_step",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)1, C, H, W}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            const float xc = static_cast<float>(x_d(b, c, h, w));
            const float v = static_cast<float>(vc_d(b, c, h, w));
            x_d(b, c, h, w) = static_cast<Scalar>(xc + dt * v);
          });
      Kokkos::fence();
    }
  }
  Kokkos::fence();
  // Single device->host copy of the final latents.
  auto xh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_d);
  std::vector<float> x(C * T);
  size_t k = 0;
  for (size_t c = 0; c < C; ++c)
    for (size_t h = 0; h < H; ++h)
      for (size_t w = 0; w < W; ++w) x[k++] = (float)xh(0, c, h, w);
  return x;
}

}  // namespace Sana
}  // namespace KokkosDNN

#endif
