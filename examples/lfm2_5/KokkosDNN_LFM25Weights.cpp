#include "KokkosDNN_LFM25Weights.hpp"

#include <cstring>
#include <random>
#include <stdexcept>

namespace KokkosDNN {
namespace LFM25 {
namespace {

// GGUF leaf names follow llama.cpp's `tn()` convention: per-layer tensors
// live under "blk.{l}." with a ".weight" (matrices/vectors) or ".bias"
// (expert bias) suffix. The resolver below also accepts the bare stems so
// files written without suffixes keep working.
std::string blk(int l, const std::string& leaf) {
  return "blk." + std::to_string(l) + "." + leaf;
}

std::string shape_str_2d(size_t rows, size_t cols) {
  return "[" + std::to_string(rows) + "," + std::to_string(cols) + "]";
}

std::string info_shape_str(const Gguf::GgufTensorInfo& info) {
  std::string o = "[";
  if (info.n_dims == 1) {
    o += std::to_string((size_t)info.numel);
  } else if (info.n_dims == 2) {
    o += std::to_string(info.rows()) + "," + std::to_string(info.cols());
  } else {
    for (uint32_t d = 0; d < info.n_dims; ++d) {
      if (d) o += ",";
      o += std::to_string(info.dims[d]);
    }
  }
  return o + "]:" + Gguf::ggml_type_name(info.dtype);
}

// Resolve stem+suffix, falling back to the bare stem. Returns nullptr when
// neither is present (caller reports the missing tensor).
const Gguf::GgufTensorInfo* resolve(const Gguf::GgufFile& file,
                                    const std::string& stem,
                                    const char* suffix) {
  const std::string suffixed = stem + suffix;
  if (file.contains(suffixed)) return &file.find(suffixed);
  if (file.contains(stem)) return &file.find(stem);
  return nullptr;
}

const Gguf::GgufTensorInfo& require(const Gguf::GgufFile& file,
                                     const std::string& stem,
                                     const char* suffix) {
  if (const auto* info = resolve(file, stem, suffix)) return *info;
  throw std::runtime_error("tensor not found in checkpoint: '" + stem +
                           suffix + "' (also tried '" + stem + "')");
}

// Prefer the llama.cpp `lfm2moe` leaf name; fall back to the legacy alias
// (the loader's require() below reports the missing tensor either way).
// Real LFM2.5-24B-A2B GGUFs name the final norm `token_embd_norm` (not
// `output_norm`) and the MoE routing bias `blk.{l}.exp_probs_b` (not
// `blk.{l}.ffn_exp_probs_b`).
std::string pick_stem(const Gguf::GgufFile& file, const std::string& primary,
                      const std::string& fallback, const char* suffix) {
  if (resolve(file, primary, suffix)) return primary;
  return fallback;
}

// Load one 2-D dense tensor: validate [rows, cols], dequantize on the host,
// convert into the device view.
template <class ExecSpace, class Scalar>
void load_dense_mat(
    const Gguf::GgufFile& file, const std::string& stem, size_t rows,
    size_t cols,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst) {
  const auto& info = require(file, stem, ".weight");
  if (info.n_dims != 2 || info.rows() != rows || info.cols() != cols)
    throw std::runtime_error("shape mismatch for '" + info.name +
                             "': checkpoint has " + info_shape_str(info) +
                             ", expected " + shape_str_2d(rows, cols));
  std::vector<float> tmp;
  try {
    Gguf::dequant_tensor_to_float(file, info, tmp);
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot decode '" + info.name + "': " + e.what());
  }
  copy_host_to_device<ExecSpace, Scalar>(tmp, rows, cols, dst);
}

// Load one 1-D dense tensor (norms, biases).
template <class ExecSpace, class Scalar>
void load_dense_vec(const Gguf::GgufFile& file, const std::string& stem,
                    const char* suffix,
                    const Kokkos::View<Scalar*, ExecSpace>& dst) {
  const auto& info = require(file, stem, suffix);
  if (info.numel != dst.extent(0))
    throw std::runtime_error("shape mismatch for '" + info.name +
                             "': checkpoint has " + info_shape_str(info) +
                             ", expected [" +
                             std::to_string(dst.extent(0)) + "]");
  std::vector<float> tmp;
  try {
    Gguf::dequant_tensor_to_float(file, info, tmp);
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot decode '" + info.name + "': " + e.what());
  }
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < tmp.size(); ++i)
    h(i) = static_cast<Scalar>(tmp[i]);
  Kokkos::deep_copy(dst, h);
}

// Slice expert `e` out of a stacked 3-D MoE tensor (ggml dim order
// {cols, rows, n_experts}) into a packed device QuantWeight.
template <class ExecSpace>
Gguf::QuantWeight<ExecSpace> load_expert_slice(
    const Gguf::GgufFile& file, const Gguf::GgufTensorInfo& info, size_t expert,
    size_t rows, size_t cols, size_t n_experts) {
  if (!Gguf::gguf_can_compute(info.dtype))
    throw std::runtime_error(
        "unsupported dtype " + Gguf::ggml_type_name(info.dtype) + " for '" +
        info.name + "' (supported: F32/F16/BF16/Q8_0/Q4_K/Q6_K)");
  if (info.n_dims != 3 || info.dims[2] != n_experts || info.dims[1] != rows ||
      info.dims[0] != cols)
    throw std::runtime_error(
        "shape mismatch for '" + info.name + "': checkpoint has " +
        info_shape_str(info) + ", expected {" + std::to_string(cols) + "," +
        std::to_string(rows) + "," + std::to_string(n_experts) +
        "} (ggml order {cols,rows,experts})");
  size_t rb = 0;
  try {
    rb = Gguf::ggml_row_bytes(info.dtype, cols);
  } catch (const std::exception& e) {
    throw std::runtime_error("tensor '" + info.name + "': " + e.what());
  }
  const size_t ebytes = rows * rb;
  if (info.nbytes != n_experts * ebytes)
    throw std::runtime_error("payload size mismatch for '" + info.name +
                             "': got " + std::to_string(info.nbytes) +
                             ", expected " + std::to_string(n_experts * ebytes));
  Gguf::QuantWeight<ExecSpace> qw;
  qw.dtype = info.dtype;
  qw.rows  = rows;
  qw.cols  = cols;
  qw.data  = Kokkos::View<uint8_t*, ExecSpace>("LFM25::expert", ebytes);
  Kokkos::View<uint8_t*, Kokkos::HostSpace> h("LFM25::expert_host", ebytes);
  std::memcpy(h.data(), file.payload(info) + expert * ebytes, ebytes);
  Kokkos::deep_copy(qw.data, h);
  return qw;
}

}  // namespace

template <class ExecSpace, class Scalar>
void copy_host_to_device(
    const std::vector<float>& host, size_t rows, size_t cols,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst) {
  if (dst.extent(0) != rows || dst.extent(1) != cols)
    throw std::runtime_error("copy_host_to_device: shape mismatch");
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      h(i, j) = static_cast<Scalar>(host[i * cols + j]);
  Kokkos::deep_copy(dst, h);
}

template <class ExecSpace, class Scalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed) {
  w.allocate(cfg);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  auto fill_mat = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      for (size_t j = 0; j < h.extent(1); ++j)
        h(i, j) = static_cast<Scalar>(dist(rng));
    Kokkos::deep_copy(v, h);
  };
  auto fill_vec_ones = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      h(i) = static_cast<Scalar>(1.0f);
    Kokkos::deep_copy(v, h);
  };
  auto fill_vec_zeros = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      h(i) = static_cast<Scalar>(0.0f);
    Kokkos::deep_copy(v, h);
  };
  // Packed F32 expert with small random values (raw-float packing is exact,
  // so quant_gemm() on it matches a dense reference bit-for-bit).
  auto fill_expert = [&](Gguf::QuantWeight<ExecSpace>& qw, size_t rows,
                         size_t cols) {
    qw.dtype = Gguf::GgmlType::F32;
    qw.rows  = rows;
    qw.cols  = cols;
    const size_t nbytes = qw.bytes();
    qw.data = Kokkos::View<uint8_t*, ExecSpace>("expert", nbytes);
    Kokkos::View<uint8_t*, Kokkos::HostSpace> h("expert_host", nbytes);
    float* f = reinterpret_cast<float*>(h.data());
    for (size_t i = 0; i < rows * cols; ++i) f[i] = dist(rng);
    Kokkos::deep_copy(qw.data, h);
  };
  fill_mat(w.embed);
  fill_mat(w.lm_head);
  fill_vec_ones(w.final_norm);
  for (auto& b : w.blocks) {
    if (b.is_attention) {
      fill_mat(b.wq);
      fill_mat(b.wk);
      fill_mat(b.wv);
      fill_mat(b.wo);
      fill_vec_ones(b.q_norm);
      fill_vec_ones(b.k_norm);
    } else {
      fill_mat(b.in_proj);
      fill_mat(b.conv_dw);
      fill_mat(b.out_proj);
    }
    fill_vec_ones(b.attn_norm);
    fill_vec_ones(b.ffn_norm);
    if (!b.is_moe) {
      fill_mat(b.wgate);
      fill_mat(b.wup);
      fill_mat(b.wdown);
    } else {
      fill_mat(b.gate_inp);
      if (b.exp_bias.extent(0) != 0) fill_vec_zeros(b.exp_bias);
      for (size_t e = 0; e < b.gate_exps.size(); ++e) {
        fill_expert(b.gate_exps[e], (size_t)cfg.moe_intermediate,
                    (size_t)cfg.dim);
        fill_expert(b.up_exps[e], (size_t)cfg.moe_intermediate,
                    (size_t)cfg.dim);
        fill_expert(b.down_exps[e], (size_t)cfg.dim,
                    (size_t)cfg.moe_intermediate);
      }
    }
  }
}

template <class ExecSpace, class Scalar>
void load_gguf(const Config& cfg, const std::string& path,
               Weights<ExecSpace, Scalar>& w) {
  w.allocate(cfg);
  Gguf::GgufFile file(path);  // throws on missing/corrupt file
  const size_t D = (size_t)cfg.dim;
  const size_t E = (size_t)cfg.n_experts;

  load_dense_mat<ExecSpace, Scalar>(file, "token_embd", (size_t)cfg.vocab_size,
                                   D, w.embed);
  load_dense_vec<ExecSpace, Scalar>(
      file, pick_stem(file, "token_embd_norm", "output_norm", ".weight"),
      ".weight", w.final_norm);
  // Tied embeddings fallback (LFM2.5 ships tie_word_embeddings=true).
  if (resolve(file, "output", ".weight") != nullptr) {
    load_dense_mat<ExecSpace, Scalar>(file, "output", (size_t)cfg.vocab_size,
                                     D, w.lm_head);
  } else if (cfg.tie_word_embeddings) {
    w.lm_head = w.embed;  // tied: alias the embedding table
  } else {
    require(file, "output", ".weight");  // rethrow the missing-tensor error
  }

  const size_t hq  = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
  const size_t hkv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
  const size_t HD  = (size_t)cfg.head_dim;
  const size_t K   = (size_t)cfg.conv_L_cache;
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b = w.blocks[l];
    load_dense_vec<ExecSpace, Scalar>(file, blk(l, "attn_norm"), ".weight",
                                     b.attn_norm);
    load_dense_vec<ExecSpace, Scalar>(file, blk(l, "ffn_norm"), ".weight",
                                     b.ffn_norm);
    if (b.is_attention) {
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "attn_q"), hq, D, b.wq);
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "attn_k"), hkv, D, b.wk);
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "attn_v"), hkv, D, b.wv);
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "attn_output"), D, hq,
                                       b.wo);
      load_dense_vec<ExecSpace, Scalar>(file, blk(l, "attn_q_norm"), ".weight",
                                       b.q_norm);
      load_dense_vec<ExecSpace, Scalar>(file, blk(l, "attn_k_norm"), ".weight",
                                       b.k_norm);
    } else {
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "shortconv.in_proj"),
                                       3 * D, D, b.in_proj);
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "shortconv.conv"), D, K,
                                       b.conv_dw);
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "shortconv.out_proj"), D,
                                       D, b.out_proj);
    }
    if (!b.is_moe) {
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "ffn_gate"),
                                       (size_t)cfg.ffn_dim, D, b.wgate);
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "ffn_up"),
                                       (size_t)cfg.ffn_dim, D, b.wup);
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "ffn_down"), D,
                                       (size_t)cfg.ffn_dim, b.wdown);
    } else {
      load_dense_mat<ExecSpace, Scalar>(file, blk(l, "ffn_gate_inp"), E, D,
                                       b.gate_inp);
      const auto& gi = require(file, blk(l, "ffn_gate_exps"), ".weight");
      const auto& ui = require(file, blk(l, "ffn_up_exps"), ".weight");
      const auto& di = require(file, blk(l, "ffn_down_exps"), ".weight");
      const size_t MI = (size_t)cfg.moe_intermediate;
      for (size_t e = 0; e < E; ++e) {
        b.gate_exps[e] = load_expert_slice<ExecSpace>(file, gi, e, MI, D, E);
        b.up_exps[e]   = load_expert_slice<ExecSpace>(file, ui, e, MI, D, E);
        b.down_exps[e] = load_expert_slice<ExecSpace>(file, di, e, D, MI, E);
      }
      if (cfg.use_expert_bias) {
        load_dense_vec<ExecSpace, Scalar>(
            file,
            pick_stem(file, blk(l, "exp_probs_b"), blk(l, "ffn_exp_probs_b"),
                      ".bias"),
            ".bias", b.exp_bias);
      }
    }
  }
}

// Explicit instantiations for the default execution space (defined here so
// the .cpp compiles them). All callers use DefaultExecutionSpace, so no
// backend needs enumerating here: whichever space Kokkos was configured with
// (Serial, OpenMP, Cuda, HIP, ...) is covered. DefaultHostExecutionSpace is
// deliberately not instantiated: nothing calls these functions on a host
// space, and on CPU-only builds it aliases DefaultExecutionSpace, where a
// second instantiation would be a duplicate-definition error.
// On builds without a native bhalf type Kokkos aliases bhalf_t to float
// (KOKKOS_BHALF_T_IS_FLOAT), so the bhalf_t lines are skipped there to
// avoid duplicate explicit instantiations.
template void random_init<Kokkos::DefaultExecutionSpace, float>(const Config&, Weights<Kokkos::DefaultExecutionSpace, float>&, uint64_t);
template void load_gguf<Kokkos::DefaultExecutionSpace, float>(const Config&, const std::string&, Weights<Kokkos::DefaultExecutionSpace, float>&);
template void copy_host_to_device<Kokkos::DefaultExecutionSpace, float>(const std::vector<float>&, size_t, size_t,
    const Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>&);
#if !KOKKOS_BHALF_T_IS_FLOAT
template void random_init<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const Config&, Weights<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>&, uint64_t);
template void load_gguf<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const Config&, const std::string&, Weights<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>&);
template void copy_host_to_device<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const std::vector<float>&, size_t, size_t,
    const Kokkos::View<Kokkos::Experimental::bhalf_t**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>&);
#endif

}  // namespace LFM25
}  // namespace KokkosDNN
