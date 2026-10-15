#include "KokkosDNN_SanaWeights.hpp"

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

// mmap-backed reads (POSIX). Fallback: bounded ifstream reads with the same
// chunked staging when mmap is unavailable (mirrors the safetensors loaders).
#if !defined(KOKKOSDNN_HAS_MMAP)
#if defined(__unix__) || defined(__unix) || defined(__APPLE__) || \
    defined(__linux__)
#define KOKKOSDNN_HAS_MMAP 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define KOKKOSDNN_HAS_MMAP 0
#endif
#endif

namespace KokkosDNN {
namespace Sana {
namespace {

// ---- little-endian loads ----
uint16_t ld16(const char* p) {
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  return (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
}
uint32_t ld32(const char* p) {
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  return b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
}

// ---- file reader: mmap window or bounded ifstream seeks ----
struct FileReader {
  std::string path;
  size_t size = 0;
#if KOKKOSDNN_HAS_MMAP
  const char* map = nullptr;
#endif
  explicit FileReader(const std::string& p) : path(p) {
#if KOKKOSDNN_HAS_MMAP
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("sana: cannot open " + p);
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("sana: cannot stat " + p);
    }
    if (st.st_size <= 0) {
      ::close(fd);
      throw std::runtime_error("sana: empty file: " + p);
    }
    size    = (size_t)st.st_size;
    void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("sana: mmap failed: " + p);
    map = static_cast<const char*>(m);
#else
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("sana: cannot open " + p);
    std::streamoff e = f.tellg();
    if (e <= 0) throw std::runtime_error("sana: empty file: " + p);
    size = (size_t)e;
#endif
  }
  ~FileReader() {
#if KOKKOSDNN_HAS_MMAP
    if (map) ::munmap(const_cast<char*>(map), size);
#endif
  }
  FileReader(const FileReader&)            = delete;
  FileReader& operator=(const FileReader&) = delete;

  void read_bytes(uint64_t off, char* dst, size_t n) const {
    if (off + n > size)
      throw std::runtime_error("sana: read out of range");
#if KOKKOSDNN_HAS_MMAP
    std::memcpy(dst, map + off, n);
#else
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("sana: cannot open " + path);
    f.seekg((std::streamoff)off);
    f.read(dst, (std::streamsize)n);
    if (!f) throw std::runtime_error("sana: truncated read");
#endif
  }
  std::string read_str(uint64_t off, size_t n) const {
    std::string s(n, '\0');
    if (n) read_bytes(off, s.data(), n);
    return s;
  }
};

// ---- minimal ZIP central-directory reader (stored entries) ----
struct ZipEntry {
  uint16_t method = 0;
  uint64_t comp_size = 0, uncomp_size = 0, local_off = 0;
};

struct ZipArchive {
  const FileReader& fr;
  std::map<std::string, ZipEntry> entries;
  explicit ZipArchive(const FileReader& r) : fr(r) {
    if (fr.size < 22) throw std::runtime_error("sana: not a zip file");
    // Locate EOCD (max 64K comment).
    const size_t tail =
        std::min<size_t>(fr.size, (size_t)65535 + 22);
    std::string buf = fr.read_str(fr.size - tail, tail);
    size_t eocd = std::string::npos;
    for (size_t i = buf.size() - 22;; --i) {
      if (ld32(buf.data() + i) == 0x06054b50) {
        eocd = i;
        break;
      }
      if (i == 0) break;
    }
    if (eocd == std::string::npos)
      throw std::runtime_error("sana: zip EOCD not found");
    const char* e      = buf.data() + eocd;
    uint64_t cd_count  = ld16(e + 10);
    uint64_t cd_size   = ld32(e + 12);
    uint64_t cd_offset = ld32(e + 16);
    (void)cd_size;
    if (cd_offset >= fr.size)
      throw std::runtime_error("sana: bad zip central directory offset");
    // Walk central directory.
    uint64_t p = cd_offset;
    for (uint64_t k = 0; k < cd_count; ++k) {
      char hdr[46];
      fr.read_bytes(p, hdr, 46);
      if (ld32(hdr) != 0x02014b50)
        throw std::runtime_error("sana: bad zip central header");
      ZipEntry ze;
      ze.method      = ld16(hdr + 10);
      ze.comp_size   = ld32(hdr + 20);
      ze.uncomp_size = ld32(hdr + 24);
      uint16_t fn_len = ld16(hdr + 28);
      uint16_t ex_len = ld16(hdr + 30);
      uint16_t fc_len = ld16(hdr + 32);
      ze.local_off   = ld32(hdr + 42);
      std::string name = fr.read_str(p + 46, fn_len);
      entries[name]    = ze;
      p += 46 + fn_len + ex_len + fc_len;
    }
  }

  // Absolute file offset of an entry's data (via its local header).
  uint64_t data_offset(const std::string& name) const {
    auto it = entries.find(name);
    if (it == entries.end())
      throw std::runtime_error("sana: zip entry not found: " + name);
    char lh[30];
    fr.read_bytes(it->second.local_off, lh, 30);
    if (ld32(lh) != 0x04034b50)
      throw std::runtime_error("sana: bad zip local header for " + name);
    uint16_t fn_len = ld16(lh + 26);
    uint16_t ex_len = ld16(lh + 28);
    return it->second.local_off + 30 + fn_len + ex_len;
  }

  std::string read_entry(const std::string& name) const {
    auto it = entries.find(name);
    if (it == entries.end())
      throw std::runtime_error("sana: zip entry not found: " + name);
    if (it->second.method != 0)
      throw std::runtime_error("sana: unsupported zip compression for " +
                               name + " (only stored entries are supported)");
    return fr.read_str(data_offset(name), (size_t)it->second.uncomp_size);
  }
};

// Streaming chunk size (elements per host staging buffer): bounds extra
// host RAM to ~1 MiB regardless of tensor size.
inline constexpr size_t kStreamChunkElems = size_t{1} << 18;

template <class ExecSpace, class Scalar>
void stream_mat(const FileReader& fr, uint64_t off, size_t rows, size_t cols,
                const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>&
                    dst) {
  if (dst.extent(0) != rows || dst.extent(1) != cols)
    throw std::runtime_error("sana: device view shape mismatch");
  if (rows == 0 || cols == 0) return;
  const size_t rpc =
      std::max<size_t>(1, kStreamChunkElems / std::max<size_t>(cols, 1));
  std::vector<char> raw;
  for (size_t r0 = 0; r0 < rows; r0 += rpc) {
    const size_t r1 = std::min(rows, r0 + rpc);
    const size_t n  = (r1 - r0) * cols;
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace> stage(
        "sana_chunk", r1 - r0, cols);
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
    auto sub =
        Kokkos::subview(dst, std::make_pair(r0, r1), Kokkos::ALL());
    Kokkos::deep_copy(sub, stage);
  }
}

template <class ExecSpace, class Scalar>
void stream_vec(const FileReader& fr, uint64_t off,
                const Kokkos::View<Scalar*, ExecSpace>& dst) {
  const size_t n = dst.extent(0);
  if (n == 0) return;
  Kokkos::View<Scalar*, Kokkos::HostSpace> stage("sana_chunk", n);
  if constexpr (std::is_same_v<Scalar, float>) {
    fr.read_bytes(off, reinterpret_cast<char*>(stage.data()), n * 4);
  } else {
    std::vector<char> raw(n * 4);
    fr.read_bytes(off, raw.data(), n * 4);
    const float* f = reinterpret_cast<const float*>(raw.data());
    for (size_t i = 0; i < n; ++i)
      stage(i) = static_cast<Scalar>(f[i]);
  }
  Kokkos::deep_copy(dst, stage);
}

// Expected state_dict key order for a config (pos_embed + 14 global +
// 15 tensors per block + 3 final = 18 + 15*depth keys).
std::vector<std::string> expected_keys(const Config& cfg) {
  std::vector<std::string> k;
  k.reserve(18 + (size_t)15 * (size_t)cfg.depth);
  k.push_back("pos_embed");
  k.push_back("x_embedder.proj.weight");
  k.push_back("x_embedder.proj.bias");
  k.push_back("t_embedder.mlp.0.weight");
  k.push_back("t_embedder.mlp.0.bias");
  k.push_back("t_embedder.mlp.2.weight");
  k.push_back("t_embedder.mlp.2.bias");
  k.push_back("t_block.1.weight");
  k.push_back("t_block.1.bias");
  k.push_back("y_embedder.y_embedding");
  k.push_back("y_embedder.y_proj.fc1.weight");
  k.push_back("y_embedder.y_proj.fc1.bias");
  k.push_back("y_embedder.y_proj.fc2.weight");
  k.push_back("y_embedder.y_proj.fc2.bias");
  k.push_back("attention_y_norm.weight");
  for (int b = 0; b < cfg.depth; ++b) {
    const std::string p = "blocks." + std::to_string(b) + ".";
    k.push_back(p + "scale_shift_table");
    k.push_back(p + "attn.qkv.weight");
    k.push_back(p + "attn.proj.weight");
    k.push_back(p + "attn.proj.bias");
    k.push_back(p + "cross_attn.q_linear.weight");
    k.push_back(p + "cross_attn.q_linear.bias");
    k.push_back(p + "cross_attn.kv_linear.weight");
    k.push_back(p + "cross_attn.kv_linear.bias");
    k.push_back(p + "cross_attn.proj.weight");
    k.push_back(p + "cross_attn.proj.bias");
    k.push_back(p + "mlp.inverted_conv.conv.weight");
    k.push_back(p + "mlp.inverted_conv.conv.bias");
    k.push_back(p + "mlp.depth_conv.conv.weight");
    k.push_back(p + "mlp.depth_conv.conv.bias");
    k.push_back(p + "mlp.point_conv.conv.weight");
  }
  k.push_back("final_layer.scale_shift_table");
  k.push_back("final_layer.linear.weight");
  k.push_back("final_layer.linear.bias");
  return k;
}

}  // namespace

template <class ExecSpace, class Scalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed) {
  cfg.validate();
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
  auto fill_mat3 = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      for (size_t j = 0; j < h.extent(1); ++j)
        for (size_t k = 0; k < h.extent(2); ++k)
          h(i, j, k) = static_cast<Scalar>(dist(rng));
    Kokkos::deep_copy(v, h);
  };
  auto fill_vec_rand = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      h(i) = static_cast<Scalar>(dist(rng));
    Kokkos::deep_copy(v, h);
  };
  auto fill_vec_ones = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      h(i) = static_cast<Scalar>(1.0f);
    Kokkos::deep_copy(v, h);
  };
  auto fill_vec_zero = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      h(i) = static_cast<Scalar>(0.0f);
    Kokkos::deep_copy(v, h);
  };
  auto fill_mat_zero = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      for (size_t j = 0; j < h.extent(1); ++j)
        h(i, j) = static_cast<Scalar>(0.0f);
    Kokkos::deep_copy(v, h);
  };
  fill_mat(w.x_proj_w);
  fill_vec_rand(w.x_proj_b);
  fill_mat(w.t_mlp0_w);
  fill_vec_rand(w.t_mlp0_b);
  fill_mat(w.t_mlp2_w);
  fill_vec_rand(w.t_mlp2_b);
  fill_mat(w.t_block_w);
  fill_vec_zero(w.t_block_b);  // adaLN-zero init: conditioning starts at 0
  fill_mat(w.y_embedding);
  fill_mat(w.y_fc1_w);
  fill_vec_rand(w.y_fc1_b);
  fill_mat(w.y_fc2_w);
  fill_vec_rand(w.y_fc2_b);
  fill_vec_ones(w.y_norm_w);
  for (auto& b : w.blocks) {
    fill_mat_zero(b.scale_shift_table);
    fill_mat(b.attn_qkv_w);
    fill_mat(b.attn_proj_w);
    fill_vec_rand(b.attn_proj_b);
    fill_mat(b.cross_q_w);
    fill_vec_rand(b.cross_q_b);
    fill_mat(b.cross_kv_w);
    fill_vec_rand(b.cross_kv_b);
    fill_mat(b.cross_proj_w);
    fill_vec_rand(b.cross_proj_b);
    fill_mat(b.mlp_inv_w);
    fill_vec_rand(b.mlp_inv_b);
    fill_mat3(b.mlp_dw_w);
    fill_vec_rand(b.mlp_dw_b);
    fill_mat(b.mlp_pt_w);
  }
  fill_mat_zero(w.final_table);
  fill_mat(w.final_w);
  fill_vec_rand(w.final_b);
}

template <class ExecSpace, class Scalar>
void load_torch_pth(const Config& cfg, const std::string& pth_path,
                     Weights<ExecSpace, Scalar>& w) {
  cfg.validate();
  FileReader fr(pth_path);
  ZipArchive zip(fr);

  // Archive prefix: torch.save writes `<stem>/data.pkl`, `<stem>/data/N`.
  std::string pkl_name;
  for (const auto& [name, ze] : zip.entries) {
    (void)ze;
    if (name.size() >= 9 &&
        name.compare(name.size() - 9, 9, "/data.pkl") == 0) {
      pkl_name = name;
      break;
    }
  }
  if (pkl_name.empty())
    throw std::runtime_error("sana: data.pkl not found in " + pth_path);
  const std::string prefix = pkl_name.substr(0, pkl_name.size() - 9);

  // Endianness guard: storage files are raw little-endian float32.
  const std::string byteorder = zip.read_entry(prefix + "/byteorder");
  if (byteorder != "little")
    throw std::runtime_error("sana: unsupported byteorder '" + byteorder +
                             "' (only little-endian float32 is supported)");

  const std::string pkl = zip.read_entry(pkl_name);

  // Dtype guard: only float32 storages are supported. storages appear as
  // `torch <Type>Storage` GLOBAL references in data.pkl.
  {
    std::map<std::string, int> seen;
    const std::string tag = "Storage";
    for (size_t p = 0;;) {
      p = pkl.find(tag, p);
      if (p == std::string::npos) break;
      size_t a = p;
      while (a > 0 && (std::isalnum((unsigned char)pkl[a - 1]) ||
                       pkl[a - 1] == '_'))
        --a;
      seen[pkl.substr(a, p + tag.size() - a)]++;
      p += tag.size();
    }
    if (seen.empty())
      throw std::runtime_error("sana: no torch Storage found in data.pkl");
    for (const auto& [t, n] : seen) {
      (void)n;
      if (t != "FloatStorage")
        throw std::runtime_error("sana: unsupported storage dtype '" + t +
                                 "' (only float32 is supported)");
    }
  }

  // Key scanner: the state_dict keys appear in data.pkl as contiguous
  // BINUNICODE/SHORT_BINUNICODE strings in insertion order, which matches
  // the storage id order (Nth key <-> data/N). Verify each expected key in
  // strictly increasing offset order.
  const std::vector<std::string> want = expected_keys(cfg);
  {
    size_t pos = 0;
    for (const auto& key : want) {
      pos = pkl.find(key, pos);
      if (pos == std::string::npos)
        throw std::runtime_error("sana: key not found in data.pkl: " + key);
      pos += key.size();
    }
  }

  // Storage files data/N must exist 1:1 with the keys.
  for (size_t i = 0; i < want.size(); ++i) {
    const std::string nm = prefix + "/data/" + std::to_string(i);
    if (!zip.entries.count(nm))
      throw std::runtime_error("sana: missing storage file " + nm +
                               " for key '" + want[i] + "'");
  }
  for (const auto& [name, ze] : zip.entries) {
    (void)ze;
    const std::string dp = prefix + "/data/";
    if (name.compare(0, dp.size(), dp) == 0) {
      const std::string tail = name.substr(dp.size());
      bool digits = !tail.empty();
      for (char c : tail) digits = digits && std::isdigit((unsigned char)c);
      if (digits && (size_t)std::stoull(tail) >= want.size())
        throw std::runtime_error("sana: unexpected extra storage file " +
                                 name);
    }
  }

  w.allocate(cfg);
  const size_t D  = (size_t)cfg.hidden_size;
  const size_t Hh = (size_t)cfg.mlp_hidden();
  const size_t F  = (size_t)cfg.t_freq_dim;
  const size_t Cc = (size_t)cfg.caption_channels;
  const size_t Ml = (size_t)cfg.model_max_length;
  const size_t Ic = (size_t)cfg.in_channels;
  const size_t Oc = (size_t)cfg.out_channels;

  size_t ik          = 0;  // key/storage index (Nth key <-> data/N)
  uint64_t num_total = 0;  // bytes consumed (sanity only)
  auto entry_off_sz = [&](const std::string& key, uint64_t numel) {
    const std::string nm = prefix + "/data/" + std::to_string(ik);
    auto it              = zip.entries.find(nm);
    if (it == zip.entries.end())
      throw std::runtime_error("sana: missing storage for " + key);
    if (it->second.method != 0)
      throw std::runtime_error("sana: unsupported compression for " + nm);
    if (it->second.uncomp_size != numel * 4)
      throw std::runtime_error("sana: shape mismatch for '" + key +
                               "': storage has " +
                               std::to_string(it->second.uncomp_size) +
                               " bytes, expected " + std::to_string(numel) +
                               " float32 elements");
    uint64_t off = zip.data_offset(nm);
    ++ik;
    num_total += numel * 4;
    return off;
  };
  auto load_mat = [&](const std::string& key, size_t rows, size_t cols,
                      const auto& dst) {
    const uint64_t numel = rows * cols;
    const uint64_t off   = entry_off_sz(key, numel);
    stream_mat<ExecSpace, Scalar>(fr, off, rows, cols, dst);
  };
  auto load_vec = [&](const std::string& key, size_t n, const auto& dst) {
    const uint64_t off = entry_off_sz(key, n);
    if (dst.extent(0) != n)
      throw std::runtime_error("sana: device view shape mismatch");
    stream_vec<ExecSpace, Scalar>(fr, off, dst);
  };

  // pos_embed is frozen and unused (use_pe=false): size-check only. Its
  // token count is not part of Config; require whole tokens of dim D.
  {
    const std::string nm = prefix + "/data/" + std::to_string(ik);
    auto it              = zip.entries.find(nm);
    if (it == zip.entries.end() || want[ik] != "pos_embed")
      throw std::runtime_error("sana: expected 'pos_embed' first");
    if (it->second.uncomp_size % (D * 4) != 0)
      throw std::runtime_error("sana: pos_embed numel is not a multiple "
                               "of hidden_size");
    ++ik;
    num_total += it->second.uncomp_size;
  }
  load_mat("x_embedder.proj.weight", D, Ic, w.x_proj_w);  // [D,Ic,1,1]
  load_vec("x_embedder.proj.bias", D, w.x_proj_b);
  load_mat("t_embedder.mlp.0.weight", D, F, w.t_mlp0_w);
  load_vec("t_embedder.mlp.0.bias", D, w.t_mlp0_b);
  load_mat("t_embedder.mlp.2.weight", D, D, w.t_mlp2_w);
  load_vec("t_embedder.mlp.2.bias", D, w.t_mlp2_b);
  load_mat("t_block.1.weight", 6 * D, D, w.t_block_w);
  load_vec("t_block.1.bias", 6 * D, w.t_block_b);
  // y_embedding must be [model_max_length, caption_channels] ([300,2304]).
  load_mat("y_embedder.y_embedding", Ml, Cc, w.y_embedding);
  load_mat("y_embedder.y_proj.fc1.weight", D, Cc, w.y_fc1_w);
  load_vec("y_embedder.y_proj.fc1.bias", D, w.y_fc1_b);
  load_mat("y_embedder.y_proj.fc2.weight", D, D, w.y_fc2_w);
  load_vec("y_embedder.y_proj.fc2.bias", D, w.y_fc2_b);
  load_vec("attention_y_norm.weight", D, w.y_norm_w);
  for (int b = 0; b < cfg.depth; ++b) {
    const std::string p = "blocks." + std::to_string(b) + ".";
    auto& bw            = w.blocks[(size_t)b];
    load_mat(p + "scale_shift_table", 6, D, bw.scale_shift_table);
    load_mat(p + "attn.qkv.weight", 3 * D, D, bw.attn_qkv_w);
    load_mat(p + "attn.proj.weight", D, D, bw.attn_proj_w);
    load_vec(p + "attn.proj.bias", D, bw.attn_proj_b);
    load_mat(p + "cross_attn.q_linear.weight", D, D, bw.cross_q_w);
    load_vec(p + "cross_attn.q_linear.bias", D, bw.cross_q_b);
    load_mat(p + "cross_attn.kv_linear.weight", 2 * D, D, bw.cross_kv_w);
    load_vec(p + "cross_attn.kv_linear.bias", 2 * D, bw.cross_kv_b);
    load_mat(p + "cross_attn.proj.weight", D, D, bw.cross_proj_w);
    load_vec(p + "cross_attn.proj.bias", D, bw.cross_proj_b);
    // Conv weights squeeze singleton dims without reordering:
    // [2H,D,1,1] -> [2H,D], [2H,1,3,3] -> [2H,3,3], [D,H,1,1] -> [D,H].
    load_mat(p + "mlp.inverted_conv.conv.weight", 2 * Hh, D, bw.mlp_inv_w);
    load_vec(p + "mlp.inverted_conv.conv.bias", 2 * Hh, bw.mlp_inv_b);
    {
      const uint64_t numel = (2 * Hh) * 9;
      const uint64_t off   = entry_off_sz(p + "mlp.depth_conv.conv.weight",
                                          numel);
      Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace> flat(
          "sana_dw_flat", 2 * Hh, 9);
      if constexpr (std::is_same_v<Scalar, float>) {
        fr.read_bytes(off, reinterpret_cast<char*>(flat.data()), numel * 4);
      } else {
        std::vector<char> raw(numel * 4);
        fr.read_bytes(off, raw.data(), numel * 4);
        const float* f = reinterpret_cast<const float*>(raw.data());
        for (size_t i = 0; i < numel; ++i)
          flat.data()[i] = static_cast<Scalar>(f[i]);
      }
      auto h = Kokkos::create_mirror_view(bw.mlp_dw_w);
      for (size_t i = 0; i < 2 * Hh; ++i)
        for (size_t k = 0; k < 9; ++k)
          h(i, k / 3, k % 3) = flat(i, k);
      Kokkos::deep_copy(bw.mlp_dw_w, h);
    }
    load_vec(p + "mlp.depth_conv.conv.bias", 2 * Hh, bw.mlp_dw_b);
    load_mat(p + "mlp.point_conv.conv.weight", D, Hh, bw.mlp_pt_w);
  }
  load_mat("final_layer.scale_shift_table", 2, D, w.final_table);
  load_mat("final_layer.linear.weight", Oc, D, w.final_w);
  load_vec("final_layer.linear.bias", Oc, w.final_b);
  if (ik != want.size())
    throw std::runtime_error("sana: internal loader error (key count)");
  (void)num_total;
}

// Explicit instantiations for float (the checkpoint dtype) on
// DefaultExecutionSpace (other ExecSpaces/Scalars instantiate implicitly
// from the header). DefaultHostExecutionSpace is deliberately not
// instantiated: all callers use DefaultExecutionSpace, and on CPU-only
// builds it aliases DefaultExecutionSpace, where a second instantiation
// would be a duplicate.
template void random_init<Kokkos::DefaultExecutionSpace, float>(const Config&,
                                                 Weights<Kokkos::DefaultExecutionSpace, float>&,
                                                 uint64_t);
template void load_torch_pth<Kokkos::DefaultExecutionSpace, float>(
    const Config&, const std::string&, Weights<Kokkos::DefaultExecutionSpace, float>&);

}  // namespace Sana
}  // namespace KokkosDNN
