#include "KokkosDNN_MossWeights.hpp"
#include "KokkosDNN_MossZip.hpp"

#include <Kokkos_Half.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace KokkosDNN {
namespace Moss {
namespace {

// ZIP reader (offsets, FileReader) lives in KokkosDNN_MossZip.hpp so the
// parser is unit-testable without a real checkpoint.
using Zip::FileReader;
using Zip::ZipArchive;
using Zip::ZipEntry;
using Zip::ld16;
using Zip::ld32;
using Zip::ld64;

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Host-side staging: file bytes land in a HostSpace View chunk and move to
// the device with a single deep_copy per chunk (mirrors the Sana loaders).
// At most kStreamChunkElems (~1 MiB) of extra host RAM is live at once: the
// checkpoint is never materialized as a full-size std::vector<float> plus a
// full-size mirror (no 2x residency on either side). FP32 accumulation is
// unchanged: checkpoints store float32 LE and values convert Scalar(f32)
// elementwise (exact for float, round-to-nearest-even for bhalf_t).
inline constexpr size_t kStreamChunkElems = size_t{1} << 18;

template <class ExecSpace, class Scalar>
void stream_mat(const FileReader& fr, uint64_t off, size_t rows, size_t cols,
                const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>&
                    dst) {
  if (dst.extent(0) != rows || dst.extent(1) != cols)
    throw std::runtime_error("moss: device view shape mismatch");
  if (rows == 0 || cols == 0) return;
  const size_t rpc =
      std::max<size_t>(1, kStreamChunkElems / std::max<size_t>(cols, 1));
  std::vector<char> raw;
  for (size_t r0 = 0; r0 < rows; r0 += rpc) {
    const size_t r1 = std::min(rows, r0 + rpc);
    const size_t n = (r1 - r0) * cols;
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace> stage(
        "moss_chunk", r1 - r0, cols);
    if constexpr (std::is_same_v<Scalar, float>) {
      fr.read_bytes(off + r0 * cols * 4,
                    reinterpret_cast<char*>(stage.data()), n * 4);
    } else {
      raw.resize(n * 4);
      fr.read_bytes(off + r0 * cols * 4, raw.data(), n * 4);
      const float* f = reinterpret_cast<const float*>(raw.data());
      for (size_t i = 0; i < n; ++i)
        stage.data()[i] = static_cast<Scalar>(f[i]);
    }
    auto sub = Kokkos::subview(dst, std::make_pair(r0, r1), Kokkos::ALL());
    Kokkos::deep_copy(sub, stage);
  }
}

template <class ExecSpace, class Scalar>
void stream_vec(const FileReader& fr, uint64_t off,
                const Kokkos::View<Scalar*, ExecSpace>& dst) {
  const size_t n = dst.extent(0);
  if (n == 0) return;
  Kokkos::View<Scalar*, Kokkos::HostSpace> stage("moss_chunk", n);
  if constexpr (std::is_same_v<Scalar, float>) {
    fr.read_bytes(off, reinterpret_cast<char*>(stage.data()), n * 4);
  } else {
    std::vector<char> raw(n * 4);
    fr.read_bytes(off, raw.data(), n * 4);
    const float* f = reinterpret_cast<const float*>(raw.data());
    for (size_t i = 0; i < n; ++i) stage(i) = static_cast<Scalar>(f[i]);
  }
  Kokkos::deep_copy(dst, stage);
}

// ---- safetensors header parsing ----
struct SafeEntry {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;  // file offsets of the data slice
};

std::string parse_json_string(const std::string& s, size_t& pos) {
  // s[pos] == '"'; returns unescaped string, pos after closing quote.
  ++pos;
  std::string out;
  while (pos < s.size()) {
    char c = s[pos++];
    if (c == '"') return out;
    if (c == '\\' && pos < s.size()) {
      out += s[pos++];
    } else {
      out += c;
    }
  }
  throw std::runtime_error("moss: malformed safetensors header (string)");
}

// Parse the safetensors header JSON into name -> entry. Only understands
// the subset torch saves: {"__metadata__":{...},"name":{"dtype":...,
// "shape":[...],"data_offsets":[a,b]},...}.
std::map<std::string, SafeEntry> parse_safetensors_header(
    const std::string& js, uint64_t data_base) {
  std::map<std::string, SafeEntry> out;
  size_t pos = 0;
  auto skip_ws = [&]() {
    while (pos < js.size() &&
           (js[pos] == ' ' || js[pos] == '\t' || js[pos] == '\n' ||
            js[pos] == '\r'))
      ++pos;
  };
  skip_ws();
  if (pos >= js.size() || js[pos] != '{')
    throw std::runtime_error("moss: malformed safetensors header");
  ++pos;
  while (true) {
    skip_ws();
    if (pos >= js.size()) throw std::runtime_error("moss: truncated header");
    if (js[pos] == '}') break;
    if (js[pos] != '"') throw std::runtime_error("moss: bad header key");
    std::string name = parse_json_string(js, pos);
    skip_ws();
    if (pos >= js.size() || js[pos] != ':')
      throw std::runtime_error("moss: bad header separator");
    ++pos;
    skip_ws();
    if (name == "__metadata__") {
      // Skip a nested object (one level).
      if (pos >= js.size() || js[pos] != '{')
        throw std::runtime_error("moss: bad __metadata__");
      int depth = 0;
      bool in_str = false;
      while (pos < js.size()) {
        char c = js[pos++];
        if (in_str) {
          if (c == '\\') {
            if (pos < js.size()) ++pos;
          } else if (c == '"') {
            in_str = false;
          }
        } else {
          if (c == '"') in_str = true;
          if (c == '{') ++depth;
          if (c == '}') {
            if (--depth == 0) break;
          }
        }
      }
    } else {
      // Tensor entry object.
      if (pos >= js.size() || js[pos] != '{')
        throw std::runtime_error("moss: bad tensor entry for " + name);
      ++pos;
      SafeEntry e;
      bool have_dtype = false, have_shape = false, have_off = false;
      while (true) {
        skip_ws();
        if (pos >= js.size())
          throw std::runtime_error("moss: truncated entry " + name);
        if (js[pos] == '}') {
          ++pos;
          break;
        }
        if (js[pos] != '"')
          throw std::runtime_error("moss: bad entry key for " + name);
        std::string k = parse_json_string(js, pos);
        skip_ws();
        if (pos >= js.size() || js[pos] != ':')
          throw std::runtime_error("moss: bad entry separator " + name);
        ++pos;
        skip_ws();
        if (k == "dtype") {
          if (pos >= js.size() || js[pos] != '"')
            throw std::runtime_error("moss: bad dtype " + name);
          e.dtype = parse_json_string(js, pos);
          have_dtype = true;
        } else if (k == "shape") {
          if (pos >= js.size() || js[pos] != '[')
            throw std::runtime_error("moss: bad shape " + name);
          ++pos;
          e.shape.clear();
          while (true) {
            skip_ws();
            if (pos < js.size() && js[pos] == ']') {
              ++pos;
              break;
            }
            size_t st = pos;
            while (pos < js.size() &&
                   (isdigit((unsigned char)js[pos]) || js[pos] == '-'))
              ++pos;
            if (st == pos)
              throw std::runtime_error("moss: bad shape num " + name);
            e.shape.push_back(std::stoll(js.substr(st, pos - st)));
            skip_ws();
            if (pos < js.size() && js[pos] == ',') ++pos;
          }
          have_shape = true;
        } else if (k == "data_offsets") {
          if (pos >= js.size() || js[pos] != '[')
            throw std::runtime_error("moss: bad offsets " + name);
          ++pos;
          uint64_t a = 0, b = 0;
          skip_ws();
          {
            size_t st = pos;
            while (pos < js.size() && isdigit((unsigned char)js[pos])) ++pos;
            a = std::stoull(js.substr(st, pos - st));
          }
          skip_ws();
          if (pos >= js.size() || js[pos] != ',')
            throw std::runtime_error("moss: bad offsets " + name);
          ++pos;
          skip_ws();
          {
            size_t st = pos;
            while (pos < js.size() && isdigit((unsigned char)js[pos])) ++pos;
            b = std::stoull(js.substr(st, pos - st));
          }
          skip_ws();
          if (pos >= js.size() || js[pos] != ']')
            throw std::runtime_error("moss: bad offsets " + name);
          ++pos;
          e.begin = data_base + a;
          e.end = data_base + b;
          have_off = true;
        } else {
          // Unknown field: skip value (string/number/array/object 1-level).
          if (pos < js.size() && js[pos] == '"') {
            (void)parse_json_string(js, pos);
          } else if (pos < js.size() && js[pos] == '[') {
            int d = 0;
            bool q = false;
            while (pos < js.size()) {
              char c = js[pos++];
              if (q) {
                if (c == '\\') {
                  if (pos < js.size()) ++pos;
                } else if (c == '"')
                  q = false;
              } else {
                if (c == '"') q = true;
                if (c == '[') ++d;
                if (c == ']') {
                  if (--d == 0) break;
                }
              }
            }
          } else {
            while (pos < js.size() && js[pos] != ',' && js[pos] != '}') ++pos;
          }
        }
        skip_ws();
        if (pos < js.size() && js[pos] == ',') ++pos;
      }
      if (!have_dtype || !have_shape || !have_off)
        throw std::runtime_error("moss: incomplete entry " + name);
      out[name] = e;
    }
    skip_ws();
    if (pos < js.size() && js[pos] == ',') ++pos;
  }
  return out;
}

// Find an entry by suffix (prefix-agnostic, e.g. "model." prefixes).
const SafeEntry& find_entry(const std::map<std::string, SafeEntry>& m,
                            const std::string& suffix) {
  const SafeEntry* best = nullptr;
  size_t best_len = 0;
  for (const auto& kv : m) {
    if (ends_with(kv.first, suffix)) {
      if (!best || kv.first.size() < best_len) {
        best = &kv.second;
        best_len = kv.first.size();
      }
    }
  }
  if (!best) throw std::runtime_error("moss: missing tensor '" + suffix +
                                      "' in safetensors checkpoint");
  return *best;
}

// Canonical expected key list (positional order for the .pth path).
std::vector<std::string> expected_keys(const Config& cfg) {
  std::vector<std::string> ks;
  ks.push_back("patch_embed.proj.weight");
  ks.push_back("patch_embed.proj.bias");
  ks.push_back("time_embed.mlp.0.weight");
  ks.push_back("time_embed.mlp.0.bias");
  ks.push_back("time_embed.mlp.2.weight");
  ks.push_back("time_embed.mlp.2.bias");
  ks.push_back("time_embed.block.weight");
  ks.push_back("time_embed.block.bias");
  ks.push_back("cap_embed.proj.weight");
  ks.push_back("cap_embed.proj.bias");
  ks.push_back("cap_embed.norm.weight");
  for (int i = 0; i < cfg.depth; ++i) {
    const std::string p = "blocks." + std::to_string(i) + ".";
    ks.push_back(p + "scale_shift_table");
    ks.push_back(p + "attn.qkv.weight");
    ks.push_back(p + "attn.cap_kv.weight");
    ks.push_back(p + "attn.proj.weight");
    ks.push_back(p + "attn.proj.bias");
    ks.push_back(p + "mlp.fc1.weight");
    ks.push_back(p + "mlp.fc1.bias");
    ks.push_back(p + "mlp.fc2.weight");
    ks.push_back(p + "mlp.fc2.bias");
  }
  ks.push_back("final.scale_shift_table");
  ks.push_back("final.linear.weight");
  ks.push_back("final.linear.bias");
  return ks;
}

[[maybe_unused]] size_t expected_numel(const Config& cfg, const std::string& key) {
  (void)cfg;  // geometry cross-check helper (kept for diagnostics/tests)
  const size_t D = (size_t)cfg.hidden_size;
  const size_t Hh = (size_t)cfg.mlp_hidden();
  const size_t P = (size_t)cfg.patch_size;
  if (ends_with(key, "patch_embed.proj.weight"))
    return D * (size_t)cfg.in_channels * P * P;
  if (ends_with(key, "patch_embed.proj.bias")) return D;
  if (ends_with(key, "time_embed.mlp.0.weight"))
    return D * (size_t)cfg.t_freq_dim;
  if (ends_with(key, "time_embed.mlp.0.bias")) return D;
  if (ends_with(key, "time_embed.mlp.2.weight")) return D * D;
  if (ends_with(key, "time_embed.mlp.2.bias")) return D;
  if (ends_with(key, "time_embed.block.weight")) return 6 * D * D;
  if (ends_with(key, "time_embed.block.bias")) return 6 * D;
  if (ends_with(key, "cap_embed.proj.weight"))
    return D * (size_t)cfg.caption_channels;
  if (ends_with(key, "cap_embed.proj.bias")) return D;
  if (ends_with(key, "cap_embed.norm.weight")) return D;
  if (ends_with(key, "scale_shift_table")) {
    if (key.rfind("blocks.", 0) == 0) return 6 * D;
    return 2 * D;
  }
  if (ends_with(key, "attn.qkv.weight")) return 3 * D * D;
  if (ends_with(key, "attn.cap_kv.weight")) return 2 * D * D;
  if (ends_with(key, "attn.proj.weight")) return D * D;
  if (ends_with(key, "attn.proj.bias")) return D;
  if (ends_with(key, "mlp.fc1.weight")) return Hh * D;
  if (ends_with(key, "mlp.fc1.bias")) return Hh;
  if (ends_with(key, "mlp.fc2.weight")) return D * Hh;
  if (ends_with(key, "mlp.fc2.bias")) return D;
  if (ends_with(key, "final.linear.weight"))
    return (size_t)cfg.out_channels * (size_t)cfg.patch_size *
           (size_t)cfg.patch_size * D;
  if (ends_with(key, "final.linear.bias"))
    return (size_t)cfg.out_channels * (size_t)cfg.patch_size *
           (size_t)cfg.patch_size;
  throw std::runtime_error("moss: unknown key " + key);
}

// Extract printable ASCII substrings (len>=4) from the pickle blob:
// these include the state_dict key strings in order.
std::vector<std::string> pickle_strings(const std::string& blob) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : blob) {
    if (c >= 32 && c < 127) {
      cur += c;
    } else {
      if (cur.size() >= 4) out.push_back(cur);
      cur.clear();
    }
  }
  if (cur.size() >= 4) out.push_back(cur);
  return out;
}

}  // namespace

template <class ExecSpace, class Scalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed) {
  cfg.validate();
  w.allocate(cfg);
  std::mt19937_64 eng(seed);
  std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
  auto rnd_mat = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(Kokkos::HostSpace(), v);
    for (size_t i = 0; i < v.extent(0); ++i)
      for (size_t j = 0; j < v.extent(1); ++j)
        h(i, j) = static_cast<Scalar>(dist(eng));
    Kokkos::deep_copy(v, h);
  };
  auto rnd_vec = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(Kokkos::HostSpace(), v);
    for (size_t i = 0; i < v.extent(0); ++i)
      h(i) = static_cast<Scalar>(dist(eng));
    Kokkos::deep_copy(v, h);
  };
  auto ones_vec = [&](auto& v) {
    Kokkos::deep_copy(v, static_cast<Scalar>(1.0f));
  };
  auto zero_mat = [&](auto& v) {
    Kokkos::deep_copy(v, static_cast<Scalar>(0.0f));
  };
  auto zero_vec = [&](auto& v) {
    Kokkos::deep_copy(v, static_cast<Scalar>(0.0f));
  };
  rnd_mat(w.x_proj_w);
  rnd_vec(w.x_proj_b);
  rnd_mat(w.t_mlp0_w);
  rnd_vec(w.t_mlp0_b);
  rnd_mat(w.t_mlp2_w);
  rnd_vec(w.t_mlp2_b);
  rnd_mat(w.t_block_w);
  rnd_vec(w.t_block_b);
  rnd_mat(w.cap_proj_w);
  rnd_vec(w.cap_proj_b);
  ones_vec(w.cap_norm_w);
  for (auto& b : w.blocks) {
    zero_mat(b.scale_shift_table);  // adaLN-zero init: identity blocks
    rnd_mat(b.attn_qkv_w);
    rnd_mat(b.attn_cap_kv_w);
    rnd_mat(b.attn_proj_w);
    rnd_vec(b.attn_proj_b);
    rnd_mat(b.mlp_fc1_w);
    rnd_vec(b.mlp_fc1_b);
    rnd_mat(b.mlp_fc2_w);
    zero_vec(b.mlp_fc2_b);
  }
  zero_mat(w.final_table);
  rnd_mat(w.final_w);
  rnd_vec(w.final_b);
}

template <class ExecSpace, class Scalar>
void load_safetensors(const Config& cfg, const std::string& st_path,
                      Weights<ExecSpace, Scalar>& w) {
  cfg.validate();
  w.allocate(cfg);
  FileReader fr(st_path);
  if (fr.size < 8) throw std::runtime_error("moss: too small: " + st_path);
  std::string hlen = fr.read_str(0, 8);
  uint64_t hn = ld64(hlen.data());
  if (8 + hn > fr.size)
    throw std::runtime_error("moss: bad safetensors header in " + st_path);
  std::string js = fr.read_str(8, (size_t)hn);
  auto entries = parse_safetensors_header(js, 8 + hn);

  // Validate a safetensors entry (F32 dtype, exact numel) and stream it
  // straight from the file into the device view via HostSpace chunks.
  auto put_mat = [&](const std::string& suffix, const auto& dst) {
    const SafeEntry& e = find_entry(entries, suffix);
    if (e.dtype != "F32")
      throw std::runtime_error("moss: tensor '" + suffix + "' has dtype " +
                               e.dtype + " (only F32 supported)");
    int64_t numel = 1;
    for (int64_t d : e.shape) numel *= d;
    const size_t want = dst.extent(0) * dst.extent(1);
    if (numel < 0 || (size_t)numel != want)
      throw std::runtime_error("moss: tensor '" + suffix + "' has numel " +
                               std::to_string(numel) + ", expected " +
                               std::to_string(want));
    if (e.end - e.begin != want * 4)
      throw std::runtime_error("moss: tensor '" + suffix +
                               "' slice size mismatch");
    stream_mat<ExecSpace, Scalar>(fr, e.begin, dst.extent(0), dst.extent(1),
                                  dst);
  };
  auto put_vec = [&](const std::string& suffix, const auto& dst) {
    const SafeEntry& e = find_entry(entries, suffix);
    if (e.dtype != "F32")
      throw std::runtime_error("moss: tensor '" + suffix + "' has dtype " +
                               e.dtype + " (only F32 supported)");
    int64_t numel = 1;
    for (int64_t d : e.shape) numel *= d;
    const size_t want = dst.extent(0);
    if (numel < 0 || (size_t)numel != want)
      throw std::runtime_error("moss: tensor '" + suffix + "' has numel " +
                               std::to_string(numel) + ", expected " +
                               std::to_string(want));
    if (e.end - e.begin != want * 4)
      throw std::runtime_error("moss: tensor '" + suffix +
                               "' slice size mismatch");
    stream_vec<ExecSpace, Scalar>(fr, e.begin, dst);
  };

  put_mat("patch_embed.proj.weight", w.x_proj_w);
  put_vec("patch_embed.proj.bias", w.x_proj_b);
  put_mat("time_embed.mlp.0.weight", w.t_mlp0_w);
  put_vec("time_embed.mlp.0.bias", w.t_mlp0_b);
  put_mat("time_embed.mlp.2.weight", w.t_mlp2_w);
  put_vec("time_embed.mlp.2.bias", w.t_mlp2_b);
  put_mat("time_embed.block.weight", w.t_block_w);
  put_vec("time_embed.block.bias", w.t_block_b);
  put_mat("cap_embed.proj.weight", w.cap_proj_w);
  put_vec("cap_embed.proj.bias", w.cap_proj_b);
  put_vec("cap_embed.norm.weight", w.cap_norm_w);
  for (int i = 0; i < cfg.depth; ++i) {
    const std::string p = "blocks." + std::to_string(i) + ".";
    auto& b = w.blocks[(size_t)i];
    put_mat(p + "scale_shift_table", b.scale_shift_table);
    put_mat(p + "attn.qkv.weight", b.attn_qkv_w);
    put_mat(p + "attn.cap_kv.weight", b.attn_cap_kv_w);
    put_mat(p + "attn.proj.weight", b.attn_proj_w);
    put_vec(p + "attn.proj.bias", b.attn_proj_b);
    put_mat(p + "mlp.fc1.weight", b.mlp_fc1_w);
    put_vec(p + "mlp.fc1.bias", b.mlp_fc1_b);
    put_mat(p + "mlp.fc2.weight", b.mlp_fc2_w);
    put_vec(p + "mlp.fc2.bias", b.mlp_fc2_b);
  }
  put_mat("final.scale_shift_table", w.final_table);
  put_mat("final.linear.weight", w.final_w);
  put_vec("final.linear.bias", w.final_b);
}

template <class ExecSpace, class Scalar>
void load_torch_pth(const Config& cfg, const std::string& pth_path,
                     Weights<ExecSpace, Scalar>& w) {
  cfg.validate();
  w.allocate(cfg);
  FileReader fr(pth_path);
  ZipArchive za(fr);
  // Locate the pickle blob (data.pkl or *.pkl).
  std::string pkl_name;
  for (const auto& kv : za.entries) {
    if (ends_with(kv.first, ".pkl") || ends_with(kv.first, ".pickle")) {
      pkl_name = kv.first;
      break;
    }
  }
  if (pkl_name.empty())
    throw std::runtime_error("moss: no .pkl entry in " + pth_path +
                             " (need torch.save or use --tiny)");
  std::string blob = za.read_stored(pkl_name);
  std::vector<std::string> strs = pickle_strings(blob);
  // Collect storage entries sorted by name (data/0, data/1, ...).
  std::vector<std::string> storages;
  for (const auto& kv : za.entries) {
    if (kv.first.rfind("data/", 0) == 0) storages.push_back(kv.first);
  }
  std::sort(storages.begin(), storages.end(), [](const std::string& a,
                                                 const std::string& b) {
    auto num = [](const std::string& s) {
      size_t p = s.find_last_of('/');
      try {
        return std::stoi(s.substr(p + 1));
      } catch (...) {
        return -1;
      }
    };
    return num(a) < num(b);
  });
  const std::vector<std::string> keys = expected_keys(cfg);
  // Positional mapping with suffix validation: the key strings appear in
  // the pickle in storage order; verify each expected suffix occurs.
  // Validation runs first so a layout mismatch throws before any device
  // allocation is touched.
  {
    size_t cursor = 0;
    for (size_t n = 0; n < keys.size(); ++n) {
      size_t found = std::string::npos;
      for (size_t k = cursor; k < strs.size(); ++k) {
        if (ends_with(strs[k], keys[n])) {
          found = k;
          break;
        }
      }
      if (found == std::string::npos)
        throw std::runtime_error(
            "moss: .pth key '" + keys[n] +
            "' not found in pickle order (checkpoint layout differs from " +
            "the documented MOSS key list; see README). Loaded " +
            std::to_string(n) + "/" + std::to_string(keys.size()));
      cursor = found + 1;
    }
    if (storages.size() < keys.size())
      throw std::runtime_error("moss: .pth has fewer storages (" +
                               std::to_string(storages.size()) +
                               ") than expected keys (" +
                               std::to_string(keys.size()) + ")");
  }
  // Stream each storage straight from the zip into its device view via
  // HostSpace chunks (no full-tensor std::vector<float> staging).
  size_t n = 0;
  auto take_mat = [&](auto& v) {
    const ZipEntry& ze = za.entries.at(storages[n]);
    const size_t expect = v.extent(0) * v.extent(1);
    if (ze.uncomp_size != expect * 4)
      throw std::runtime_error("moss: .pth storage for '" + keys[n] +
                               "' has " + std::to_string(ze.uncomp_size) +
                               " bytes, expected " +
                               std::to_string(expect * 4));
    if (ze.method != 0)
      throw std::runtime_error("moss: compressed .pth storage unsupported: " +
                               storages[n]);
    stream_mat<ExecSpace, Scalar>(fr, za.data_off(ze), v.extent(0),
                                  v.extent(1), v);
    ++n;
  };
  auto take_vec = [&](auto& v) {
    const ZipEntry& ze = za.entries.at(storages[n]);
    const size_t expect = v.extent(0);
    if (ze.uncomp_size != expect * 4)
      throw std::runtime_error("moss: .pth storage for '" + keys[n] +
                               "' has " + std::to_string(ze.uncomp_size) +
                               " bytes, expected " +
                               std::to_string(expect * 4));
    if (ze.method != 0)
      throw std::runtime_error("moss: compressed .pth storage unsupported: " +
                               storages[n]);
    stream_vec<ExecSpace, Scalar>(fr, za.data_off(ze), v);
    ++n;
  };
  take_mat(w.x_proj_w);
  take_vec(w.x_proj_b);
  take_mat(w.t_mlp0_w);
  take_vec(w.t_mlp0_b);
  take_mat(w.t_mlp2_w);
  take_vec(w.t_mlp2_b);
  take_mat(w.t_block_w);
  take_vec(w.t_block_b);
  take_mat(w.cap_proj_w);
  take_vec(w.cap_proj_b);
  take_vec(w.cap_norm_w);
  for (int i = 0; i < cfg.depth; ++i) {
    auto& b = w.blocks[(size_t)i];
    take_mat(b.scale_shift_table);
    take_mat(b.attn_qkv_w);
    take_mat(b.attn_cap_kv_w);
    take_mat(b.attn_proj_w);
    take_vec(b.attn_proj_b);
    take_mat(b.mlp_fc1_w);
    take_vec(b.mlp_fc1_b);
    take_mat(b.mlp_fc2_w);
    take_vec(b.mlp_fc2_b);
  }
  take_mat(w.final_table);
  take_mat(w.final_w);
  take_vec(w.final_b);
}

std::vector<float> load_caption_raw(const std::string& path, int expect_L,
                                    int expect_Cc) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("moss: cannot open '" + path + "'");
  const long long n = (long long)f.tellg();
  if (n <= 0 || n % 4 != 0)
    throw std::runtime_error("moss: bad float32 file '" + path + "'");
  f.seekg(0, std::ios::beg);
  std::vector<float> out((size_t)n / 4);
  f.read(reinterpret_cast<char*>(out.data()), n);
  if (!f) throw std::runtime_error("moss: short read of '" + path + "'");
  const size_t want = (size_t)expect_L * (size_t)expect_Cc;
  // Accept [L,Cc], [1,L,Cc], [1,1,L,Cc] flattened (trailing reshape only).
  if (out.size() == want) return out;
  if (out.size() > want && out.size() % want == 0) {
    // Take the last [L,Cc] block (batch-first layouts).
    return std::vector<float>(out.end() - (long long)want, out.end());
  }
  throw std::runtime_error("moss: caption file '" + path + "' holds " +
                           std::to_string(out.size()) + " floats, expected " +
                           std::to_string(want) + " ([L,Cc])");
}

// Explicit instantiations for the default execution space (float + bhalf_t).
// Both stage identically: float32 checkpoint bytes stream through HostSpace
// chunks with a single deep_copy per tensor; accumulation stays FP32.
template void random_init<Kokkos::DefaultExecutionSpace, float>(
    const Config&, Weights<Kokkos::DefaultExecutionSpace, float>&, uint64_t);
template void load_safetensors<Kokkos::DefaultExecutionSpace, float>(
    const Config&, const std::string&,
    Weights<Kokkos::DefaultExecutionSpace, float>&);
template void load_torch_pth<Kokkos::DefaultExecutionSpace, float>(
    const Config&, const std::string&,
    Weights<Kokkos::DefaultExecutionSpace, float>&);
// On backends without native bfloat16, Kokkos aliases bhalf_t to float
// (KOKKOS_BHALF_T_IS_FLOAT), so the instantiations below would duplicate
// the float ones; skip them there.
#if !KOKKOS_BHALF_T_IS_FLOAT
template void random_init<Kokkos::DefaultExecutionSpace,
                          Kokkos::Experimental::bhalf_t>(
    const Config&, Weights<Kokkos::DefaultExecutionSpace,
                           Kokkos::Experimental::bhalf_t>&,
    uint64_t);
template void load_safetensors<Kokkos::DefaultExecutionSpace,
                               Kokkos::Experimental::bhalf_t>(
    const Config&, const std::string&,
    Weights<Kokkos::DefaultExecutionSpace,
            Kokkos::Experimental::bhalf_t>&);
template void load_torch_pth<Kokkos::DefaultExecutionSpace,
                             Kokkos::Experimental::bhalf_t>(
    const Config&, const std::string&,
    Weights<Kokkos::DefaultExecutionSpace,
            Kokkos::Experimental::bhalf_t>&);
#endif

}  // namespace Moss
}  // namespace KokkosDNN
