// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// torch.save (.pth) reader for the Sana example: stored-entry ZIP parsing,
// protocol-2 pickle VM (closed opcode set), and fp32 storage streaming.
// See KokkosDNN_SanaPth.hpp for the format contract.

#include "KokkosDNN_SanaPth.hpp"

#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace KokkosDNN {
namespace Sana {
namespace Impl {
namespace {

// ---- Little-endian readers -------------------------------------------------
uint16_t rd_u16(const std::vector<char>& b, size_t p) {
  uint16_t v = 0;
  std::memcpy(&v, b.data() + p, 2);
  return v;
}

uint32_t rd_u32(const std::vector<char>& b, size_t p) {
  uint32_t v = 0;
  std::memcpy(&v, b.data() + p, 4);
  return v;
}

// ---- Stored-entry ZIP index ------------------------------------------------
// Central-directory entry we care about.
struct ZipEntry {
  std::string name;
  uint16_t method = 0;
  uint32_t size = 0;
  uint32_t local_offset = 0;
};

constexpr uint32_t kEocdSig = 0x06054b50;
constexpr uint32_t kCdSig = 0x02014b50;
constexpr uint32_t kLocalSig = 0x04034b50;

// Read the whole file (data.pkl is ~91 KiB; central directory is tiny).
// Storage blobs are never bulk-loaded: readers seek to record offsets.
std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("open_pth: cannot open '" + path + "'");
  f.seekg(0, std::ios::end);
  const long long n = (long long)f.tellg();
  if (n < 0) throw std::runtime_error("open_pth: cannot stat '" + path + "'");
  f.seekg(0, std::ios::beg);
  std::vector<char> buf((size_t)n);
  if (n > 0 && !f.read(buf.data(), n))
    throw std::runtime_error("open_pth: short read of '" + path + "'");
  return buf;
}

std::vector<ZipEntry> read_central_directory(const std::vector<char>& buf,
                                             const std::string& path) {
  // EOCD is within the last 22 + 64 KiB (max comment length).
  const size_t n = buf.size();
  if (n < 22) throw std::runtime_error("open_pth: '" + path + "' too small");
  const size_t lo = (n > (22 + 65535)) ? n - (22 + 65535) : 0;
  size_t eocd = n;  // not found
  for (size_t p = n - 22;; --p) {
    if (rd_u32(buf, p) == kEocdSig) {
      eocd = p;
      break;
    }
    if (p == lo) break;
  }
  if (eocd == n)
    throw std::runtime_error("open_pth: '" + path + "': EOCD not found");
  const uint32_t count = rd_u16(buf, eocd + 10);
  const uint32_t cd_size = rd_u32(buf, eocd + 12);
  const uint32_t cd_off = rd_u32(buf, eocd + 16);
  if ((uint64_t)cd_off + cd_size > n)
    throw std::runtime_error("open_pth: '" + path + "': bad central dir");
  std::vector<ZipEntry> out;
  size_t p = cd_off;
  for (uint32_t i = 0; i < count; ++i) {
    if (p + 46 > n || rd_u32(buf, p) != kCdSig)
      throw std::runtime_error("open_pth: '" + path + "': bad entry");
    ZipEntry e;
    e.method = rd_u16(buf, p + 10);
    e.size = rd_u32(buf, p + 24);
    e.local_offset = rd_u32(buf, p + 42);
    const uint16_t nl = rd_u16(buf, p + 28), el = rd_u16(buf, p + 30),
                   cl = rd_u16(buf, p + 32);
    if (p + 46 + nl > n)
      throw std::runtime_error("open_pth: '" + path + "': bad name");
    e.name.assign(buf.data() + p + 46, nl);
    p += 46 + nl + el + cl;
    out.push_back(e);
  }
  return out;
}

// Byte range of a stored record's payload.
std::pair<uint64_t, uint64_t> record_range(
    const std::vector<char>& buf, const std::vector<ZipEntry>& entries,
    const std::string& name, const std::string& path) {
  for (const auto& e : entries) {
    if (e.name != name) continue;
    if (e.method != 0)
      throw std::runtime_error("open_pth: '" + path + "': '" + name +
                               "' uses deflate; only stored records supported");
    const size_t lh = (size_t)e.local_offset;
    if (lh + 30 > buf.size() || rd_u32(buf, lh) != kLocalSig)
      throw std::runtime_error("open_pth: '" + path + "': bad local header");
    const uint16_t nl = rd_u16(buf, lh + 26), el = rd_u16(buf, lh + 28);
    const uint64_t data = (uint64_t)lh + 30 + nl + el;
    if (data + e.size > buf.size())
      throw std::runtime_error("open_pth: '" + path + "': truncated record");
    return {data, e.size};
  }
  throw std::runtime_error("open_pth: '" + path + "': missing record '" +
                           name + "'");
}

// ---- Minimal pickle VM -----------------------------------------------------
// Closed opcode set torch.save emits for a cpu fp32 state_dict (protocol 2).
// Values beyond strings/ints/tuples/maps never occur here.

struct Value;
using DictEntries = std::vector<std::pair<std::string, std::shared_ptr<Value>>>;

struct Value {
  enum class Type { Nil, Bool, Int, Str, Tuple, Global, Storage, Tensor, Dict };
  Type type = Type::Nil;
  bool boolean = false;
  int64_t integer = 0;
  std::string str;  // Str text, Global "module name", Storage dtype
  std::vector<Value> items;  // Tuple elements
  // Tensor fields:
  int storage_id = -1;
  int64_t storage_offset = 0, numel = 0;
  std::vector<int64_t> shape, stride;
  // DictEntries fields:
  std::shared_ptr<DictEntries> dict = nullptr;
  // Storage fields: str=dtype, integer=numel, storage_id=record id.
};

class Unpickler {
 public:
  explicit Unpickler(const std::vector<char>& bytes) : b_(bytes) {}

  Value run() {
    for (;;) {
      const uint8_t op = get();
      switch (op) {
        case 0x80: {  // PROTO
          const uint8_t v = get();
          if (v != 2)
            fail("unsupported protocol " + std::to_string(v));
          break;
        }
        case 0x28:  // MARK '('
          stack_.push_back(mark());
          break;
        case 0x2e:  // STOP '.'
          if (stack_.size() != 1) fail("STOP with stack depth != 1");
          return stack_.back();
        case 0x29: {  // EMPTY_TUPLE ')'
          Value v;
          v.type = Value::Type::Tuple;
          stack_.push_back(v);
          break;
        }
        case 0x74: {  // TUPLE 't'
          Value v;
          v.type = Value::Type::Tuple;
          v.items = pop_mark();
          stack_.push_back(v);
          break;
        }
        case 0x85:  // TUPLE1
        case 0x86:  // TUPLE2
        case 0x87: {  // TUPLE3
          const int k = (int)op - 0x84;
          if (stack_.size() < (size_t)k) fail("TUPLE stack underflow");
          Value v;
          v.type = Value::Type::Tuple;
          v.items.assign(stack_.end() - k, stack_.end());
          stack_.erase(stack_.end() - k, stack_.end());
          stack_.push_back(v);
          break;
        }
        case 0x7d: {  // EMPTY_DICT '}'
          Value v;
          v.type = Value::Type::Dict;
          v.dict = std::make_shared<DictEntries>();
          stack_.push_back(v);
          break;
        }
        case 0x73: {  // SETITEM 's'
          Value val = pop(), key = pop();
          Value& obj = top();
          dict_set(obj, key, val);
          break;
        }
        case 0x75: {  // SETITEMS 'u'
          std::vector<Value> items = pop_mark();
          if (items.size() % 2 != 0) fail("SETITEMS odd items");
          Value& obj = top();
          for (size_t i = 0; i < items.size(); i += 2)
            dict_set(obj, items[i], items[i + 1]);
          break;
        }
        case 0x63: {  // GLOBAL 'c'  module\nname\n
          Value v;
          v.type = Value::Type::Global;
          const std::string mod = get_line(), name = get_line();
          v.str = mod + " " + name;
          stack_.push_back(v);
          break;
        }
        case 0x52: {  // REDUCE 'R'
          Value args = pop(), fn = pop();
          stack_.push_back(do_reduce(fn, args));
          break;
        }
        case 0x62: {  // BUILD 'b'
          // torch.save finishes a state_dict as: REDUCE(OrderedDict) ..
          // pairs .. SETITEMS, then BUILD with state {'_metadata': ...}.
          // The state carries only module version metadata (no tensor
          // data); capture it for the inventory and keep stack discipline.
          Value state = pop();
          Value& obj = top();
          if (state.type != Value::Type::Dict || !state.dict ||
              obj.type != Value::Type::Dict || !obj.dict)
            fail("BUILD on non-dict");
          if (state.dict->size() != 1 ||
              (*state.dict)[0].first != "_metadata" ||
              !(*state.dict)[0].second ||
              (*state.dict)[0].second->type != Value::Type::Dict)
            fail("BUILD with unexpected state");
          pending_metadata_ = (*state.dict)[0].second->dict;
          break;
        }
        case 0x51: {  // BINPERSID 'Q'
          Value pid = pop();
          if (pid.type != Value::Type::Tuple || pid.items.size() != 5)
            fail("BINPERSID bad pid");
          const Value& kind = pid.items[0];
          const Value& stype = pid.items[1];
          const Value& key = pid.items[2];
          const Value& numel = pid.items[4];
          if (kind.type != Value::Type::Str || kind.str != "storage")
            fail("BINPERSID non-storage pid");
          if (stype.type != Value::Type::Global ||
              key.type != Value::Type::Str ||
              numel.type != Value::Type::Int)
            fail("BINPERSID bad pid fields");
          Value v;
          v.type = Value::Type::Storage;
          v.str = stype.str;  // e.g. "torch FloatStorage"
          v.storage_id = std::stoi(key.str);
          v.integer = numel.integer;
          stack_.push_back(v);
          break;
        }
        case 0x58: {  // BINUNICODE 'X'
          const uint32_t n = rd_u32();
          Value v;
          v.type = Value::Type::Str;
          v.str.assign(&b_[pos_], n);
          pos_ += n;
          stack_.push_back(v);
          break;
        }
        case 0x4a: {  // BININT 'J'
          Value v;
          v.type = Value::Type::Int;
          v.integer = (int64_t)rd_i32();
          stack_.push_back(v);
          break;
        }
        case 0x4b: {  // BININT1 'K'
          Value v;
          v.type = Value::Type::Int;
          v.integer = (int64_t)get();
          stack_.push_back(v);
          break;
        }
        case 0x4d: {  // BININT2 'M'
          Value v;
          v.type = Value::Type::Int;
          v.integer = (int64_t)rd_u16();
          stack_.push_back(v);
          break;
        }
        case 0x89: {  // NEWFALSE
          Value v;
          v.type = Value::Type::Bool;
          v.boolean = false;
          stack_.push_back(v);
          break;
        }
        case 0x71: {  // BINPUT 'q'
          const uint8_t i = get();
          memo_put(i, top());
          break;
        }
        case 0x72: {  // LONG_BINPUT 'r'
          const uint32_t i = rd_u32();
          memo_put(i, top());
          break;
        }
        case 0x68: {  // BINGET 'h'
          const uint8_t i = get();
          stack_.push_back(memo_get(i));
          break;
        }
        case 0x6a: {  // LONG_BINGET 'j'
          const uint32_t i = rd_u32();
          stack_.push_back(memo_get(i));
          break;
        }
        default:
          fail("unsupported opcode 0x" + hex2(op));
      }
    }
  }

  // _metadata dict captured from the single BUILD (module prefix -> version).
  std::shared_ptr<DictEntries> pending_metadata_;

 private:
  const std::vector<char>& b_;
  size_t pos_ = 0;
  std::vector<Value> stack_;
  std::vector<Value> memo_;
  std::vector<char> memo_set_;

  [[noreturn]] void fail(const std::string& m) {
    throw std::runtime_error("open_pth: pickle error at byte " +
                             std::to_string(pos_) + ": " + m);
  }

  static std::string hex2(uint8_t v) {
    const char* h = "0123456789abcdef";
    return std::string{h[v >> 4], h[v & 15]};
  }

  uint8_t get() {
    if (pos_ >= b_.size()) fail("truncated stream");
    return (uint8_t)b_[pos_++];
  }

  uint16_t rd_u16() {
    if (pos_ + 2 > b_.size()) fail("truncated u16");
    uint16_t v;
    std::memcpy(&v, &b_[pos_], 2);
    pos_ += 2;
    return v;
  }

  uint32_t rd_u32() {
    if (pos_ + 4 > b_.size()) fail("truncated u32");
    uint32_t v;
    std::memcpy(&v, &b_[pos_], 4);
    pos_ += 4;
    return v;
  }

  int32_t rd_i32() {
    uint32_t u = rd_u32();
    int32_t v;
    std::memcpy(&v, &u, 4);
    return v;
  }

  std::string get_line() {
    size_t e = pos_;
    while (e < b_.size() && b_[e] != '\n') ++e;
    if (e >= b_.size()) fail("truncated GLOBAL line");
    std::string s(&b_[pos_], e - pos_);
    pos_ = e + 1;
    return s;
  }

  Value& top() {
    if (stack_.empty()) fail("stack underflow");
    return stack_.back();
  }

  Value pop() {
    if (stack_.empty()) fail("stack underflow");
    Value v = stack_.back();
    stack_.pop_back();
    return v;
  }

  static Value mark() {
    Value v;  // Nil marks the MARK slot (Nil never occurs otherwise).
    return v;
  }

  std::vector<Value> pop_mark() {
    std::vector<Value> items;
    while (!stack_.empty() && stack_.back().type != Value::Type::Nil) {
      items.push_back(stack_.back());
      stack_.pop_back();
    }
    if (stack_.empty()) fail("MARK not found");
    stack_.pop_back();  // drop the mark
    // Items were collected LIFO; restore file order.
    for (size_t i = 0, n = items.size(); i < n / 2; ++i) {
      Value t = items[i];
      items[i] = items[n - 1 - i];
      items[n - 1 - i] = t;
    }
    return items;
  }

  void memo_put(uint32_t i, const Value& v) {
    if (i >= memo_.size()) {
      memo_.resize(i + 1);
      memo_set_.resize(i + 1, 0);
    }
    memo_[i] = v;
    memo_set_[i] = 1;
  }

  Value memo_get(uint32_t i) {
    if (i >= memo_.size() || !memo_set_[i]) fail("bad memo index");
    return memo_[i];
  }

  static void dict_set(Value& obj, const Value& key, const Value& val) {
    if (obj.type != Value::Type::Dict || !obj.dict)
      throw std::runtime_error("open_pth: SETITEM on non-dict");
    if (key.type != Value::Type::Str)
      throw std::runtime_error("open_pth: non-string dict key");
    // Keys are unique in this archive; emplace preserves file order.
    obj.dict->emplace_back(key.str, std::make_shared<Value>(val));
  }

  static int64_t as_int(const Value& v, const char* ctx) {
    if (v.type != Value::Type::Int)
      throw std::runtime_error(
          std::string("open_pth: ") + ctx + " is not an int");
    return v.integer;
  }

  // REDUCE dispatch for the two globals torch.save emits here.
  static Value do_reduce(const Value& fn, const Value& args) {
    if (fn.type != Value::Type::Global)
      throw std::runtime_error("open_pth: REDUCE of non-global");
    if (args.type != Value::Type::Tuple)
      throw std::runtime_error("open_pth: REDUCE args not a tuple");
    if (fn.str == "collections OrderedDict") {
      if (!args.items.empty())
        throw std::runtime_error("open_pth: OrderedDict with args");
      Value v;
      v.type = Value::Type::Dict;
      v.dict = std::make_shared<DictEntries>();
      return v;
    }
    if (fn.str == "torch._utils _rebuild_tensor_v2") {
      if (args.items.size() != 6)
        throw std::runtime_error("open_pth: bad tensor rebuild args");
      const Value& st = args.items[0];
      const Value& off = args.items[1];
      const Value& size = args.items[2];
      const Value& stride = args.items[3];
      if (st.type != Value::Type::Storage)
        throw std::runtime_error("open_pth: tensor without storage ref");
      Value v;
      v.type = Value::Type::Tensor;
      v.storage_id = st.storage_id;
      v.str = st.str;
      v.integer = st.integer;  // storage numel
      v.storage_offset = as_int(off, "storage_offset");
      for (const auto& d : size.items) v.shape.push_back(as_int(d, "size"));
      for (const auto& d : stride.items)
        v.stride.push_back(as_int(d, "stride"));
      int64_t n = 1;
      for (int64_t d : v.shape) n *= d;
      v.numel = n;
      return v;
    }
    throw std::runtime_error("open_pth: REDUCE of unknown " + fn.str);
  }
};

}  // namespace

// ---- Expected Sana tensor inventory ----------------------------------------
std::vector<std::pair<std::string, std::vector<int64_t>>> sana_pth_expected(
    const Config& cfg, int latent_h, int latent_w) {
  cfg.validate();
  const int64_t D = cfg.hidden_size, C = cfg.in_channels,
                Co = cfg.out_channels;
  const int64_t T = cfg.num_tokens(latent_h, latent_w);
  if (T <= 0)
    throw std::runtime_error("sana_pth_expected: bad latent geometry");
  const int64_t L = cfg.model_max_length, Cc = cfg.caption_channels,
                H = cfg.mlp_hidden(), F = cfg.t_freq_dim;
  std::vector<std::pair<std::string, std::vector<int64_t>>> out;
  const auto add = [&](const std::string& n, std::vector<int64_t> s) {
    out.emplace_back(n, std::move(s));
  };
  add("pos_embed", {1, T, D});
  add("x_embedder.proj.weight", {D, C, 1, 1});
  add("x_embedder.proj.bias", {D});
  add("t_embedder.mlp.0.weight", {D, F});
  add("t_embedder.mlp.0.bias", {D});
  add("t_embedder.mlp.2.weight", {D, D});
  add("t_embedder.mlp.2.bias", {D});
  add("t_block.1.weight", {6 * D, D});
  add("t_block.1.bias", {6 * D});
  add("y_embedder.y_embedding", {L, Cc});
  add("y_embedder.y_proj.fc1.weight", {D, Cc});
  add("y_embedder.y_proj.fc1.bias", {D});
  add("y_embedder.y_proj.fc2.weight", {D, D});
  add("y_embedder.y_proj.fc2.bias", {D});
  add("attention_y_norm.weight", {D});
  for (int l = 0; l < cfg.depth; ++l) {
    const std::string p = "blocks." + std::to_string(l) + ".";
    add(p + "scale_shift_table", {6, D});
    add(p + "attn.qkv.weight", {3 * D, D});
    add(p + "attn.proj.weight", {D, D});
    add(p + "attn.proj.bias", {D});
    add(p + "cross_attn.q_linear.weight", {D, D});
    add(p + "cross_attn.q_linear.bias", {D});
    add(p + "cross_attn.kv_linear.weight", {2 * D, D});
    add(p + "cross_attn.kv_linear.bias", {2 * D});
    add(p + "cross_attn.proj.weight", {D, D});
    add(p + "cross_attn.proj.bias", {D});
    add(p + "mlp.inverted_conv.conv.weight", {2 * H, D, 1, 1});
    add(p + "mlp.inverted_conv.conv.bias", {2 * H});
    add(p + "mlp.depth_conv.conv.weight", {2 * H, 1, 3, 3});
    add(p + "mlp.depth_conv.conv.bias", {2 * H});
    add(p + "mlp.point_conv.conv.weight", {D, H, 1, 1});
  }
  add("final_layer.scale_shift_table", {2, D});
  add("final_layer.linear.weight", {Co, D});
  add("final_layer.linear.bias", {Co});
  return out;
}

}  // namespace Impl

PthArchive open_pth(const std::string& path) {
  const std::vector<char> file = Impl::read_file(path);
  const std::vector<Impl::ZipEntry> entries =
      Impl::read_central_directory(file, path);
  // Locate data.pkl and derive the record prefix.
  std::string pkl_name;
  for (const auto& e : entries) {
    if (e.name.size() >= 8 &&
        e.name.compare(e.name.size() - 8, 8, "data.pkl") == 0) {
      pkl_name = e.name;
      break;
    }
  }
  if (pkl_name.empty())
    throw std::runtime_error("open_pth: '" + path + "': no data.pkl record");
  const std::string prefix = pkl_name.substr(0, pkl_name.size() - 8);
  const auto pkl_range = Impl::record_range(file, entries, pkl_name, path);
  std::vector<char> pkl(file.begin() + (long long)pkl_range.first,
                        file.begin() + (long long)(pkl_range.first +
                                                  pkl_range.second));
  Impl::Unpickler up(pkl);
  Impl::Value top = up.run();
  if (top.type != Impl::Value::Type::Dict || !top.dict || top.dict->empty())
    throw std::runtime_error("open_pth: '" + path + "': top is not a dict");
  // Top-level must be exactly {'state_dict': OrderedDict}.
  if (top.dict->size() != 1 || (*top.dict)[0].first != "state_dict" ||
      !(*top.dict)[0].second ||
      (*top.dict)[0].second->type != Impl::Value::Type::Dict)
    throw std::runtime_error("open_pth: '" + path +
                             "': expected {'state_dict': ...}");
  const Impl::DictEntries& sd = *(*top.dict)[0].second->dict;

  PthArchive ar;
  ar.path = path;
  ar.prefix = prefix;
  for (const auto& kv : sd) {
    const Impl::Value& v = *kv.second;
    if (v.type != Impl::Value::Type::Tensor)
      throw std::runtime_error("open_pth: '" + path + "': entry '" +
                               kv.first + "' is not a tensor");
    if (v.str != "torch FloatStorage")
      throw std::runtime_error("open_pth: '" + path + "': entry '" +
                               kv.first + "' has " + v.str +
                               " (only float32 supported)");
    PthTensorMeta m;
    m.name = kv.first;
    m.shape = v.shape;
    m.stride = v.stride;
    m.storage_offset = v.storage_offset;
    m.storage_id = v.storage_id;
    m.numel = v.numel;
    m.dtype = v.str;
    ar.tensors.push_back(std::move(m));
    if (!ar.records.count(v.storage_id)) {
      const std::string rec =
          prefix + "data/" + std::to_string(v.storage_id);
      ar.records[v.storage_id] =
          Impl::record_range(file, entries, rec, path);
    }
  }
  if (up.pending_metadata_) {
    // _metadata maps module prefix -> {'version': N}; unwrap one level.
    for (const auto& kv : *up.pending_metadata_) {
      if (!kv.second || kv.second->type != Impl::Value::Type::Dict ||
          !kv.second->dict || kv.second->dict->size() != 1 ||
          (*kv.second->dict)[0].first != "version" ||
          !(*kv.second->dict)[0].second ||
          (*kv.second->dict)[0].second->type != Impl::Value::Type::Int)
        throw std::runtime_error("open_pth: '" + path +
                                 "': bad _metadata entry '" + kv.first + "'");
      ar.metadata[kv.first] = (*kv.second->dict)[0].second->integer;
    }
  }
  return ar;
}

int64_t pth_total_params(const PthArchive& archive) {
  int64_t n = 0;
  for (const auto& t : archive.tensors) n += t.numel;
  return n;
}

std::vector<float> read_pth_storage(const PthArchive& archive,
                                    int storage_id) {
  auto it = archive.records.find(storage_id);
  if (it == archive.records.end())
    throw std::runtime_error("read_pth_storage: unknown storage " +
                             std::to_string(storage_id));
  std::ifstream f(archive.path, std::ios::binary);
  if (!f)
    throw std::runtime_error("read_pth_storage: cannot open '" +
                             archive.path + "'");
  f.seekg((long long)it->second.first, std::ios::beg);
  if (!f) throw std::runtime_error("read_pth_storage: seek failed");
  std::vector<char> raw(it->second.second);
  if (!raw.empty() && !f.read(raw.data(), (long long)raw.size()))
    throw std::runtime_error("read_pth_storage: short read");
  if (raw.size() % 4 != 0)
    throw std::runtime_error("read_pth_storage: non-float32 record size");
  std::vector<float> out(raw.size() / 4);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

}  // namespace Sana
}  // namespace KokkosDNN
