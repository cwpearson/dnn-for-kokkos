// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Kokoro torch.save reader: ZIP index + protocol-2 pickle VM.
// See KokkosDNN_KokoroPth.hpp for the format contract.

#include "KokkosDNN_KokoroPth.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace KokkosDNN {
namespace Kokoro {
namespace {

// ---------------------------------------------------------------------------
// Little-endian readers.
// ---------------------------------------------------------------------------
uint16_t rd16(const std::vector<char>& b, size_t p) {
  return (uint16_t)((uint8_t)b[p] | ((uint8_t)b[p + 1] << 8));
}
uint32_t rd32(const std::vector<char>& b, size_t p) {
  return (uint32_t)((uint8_t)b[p] | ((uint8_t)b[p + 1] << 8) |
                    ((uint8_t)b[p + 2] << 16) | ((uint8_t)b[p + 3] << 24));
}

// ---------------------------------------------------------------------------
// ZIP index (stored entries only).
// ---------------------------------------------------------------------------
struct ZipEntry {
  std::string name;
  uint64_t local_offset = 0;  // local file header offset
  uint64_t comp_size = 0;
  uint64_t uncomp_size = 0;
  uint16_t method = 0;
};

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("open_pth: cannot open '" + path + "'");
  const long long n = (long long)f.tellg();
  if (n <= 0) throw std::runtime_error("open_pth: empty file '" + path + "'");
  f.seekg(0, std::ios::beg);
  std::vector<char> buf((size_t)n);
  f.read(buf.data(), n);
  if (!f) throw std::runtime_error("open_pth: short read '" + path + "'");
  return buf;
}

std::vector<ZipEntry> zip_index(const std::vector<char>& buf) {
  const size_t n = buf.size();
  if (n < 22) throw std::runtime_error("open_pth: too small for ZIP");
  // Find EOCD (PK\x05\x06) scanning the tail (comment <= 64KB).
  size_t eocd = n;
  const size_t lo = (n > (65536 + 22)) ? n - (65536 + 22) : 0;
  for (size_t p = n - 22; ; --p) {
    if (buf[p] == 'P' && buf[p + 1] == 'K' && buf[p + 2] == '\x05' &&
        buf[p + 3] == '\x06') {
      eocd = p;
      break;
    }
    if (p == lo) throw std::runtime_error("open_pth: EOCD not found");
  }
  const uint16_t nrec = rd16(buf, eocd + 8);
  const uint32_t cd_size = rd32(buf, eocd + 12);
  const uint32_t cd_off = rd32(buf, eocd + 16);
  (void)cd_size;
  std::vector<ZipEntry> out;
  size_t p = cd_off;
  for (uint16_t i = 0; i < nrec; ++i) {
    if (p + 46 > n || !(buf[p] == 'P' && buf[p + 1] == 'K' && buf[p + 2] == '\x01' &&
                        buf[p + 3] == '\x02'))
      throw std::runtime_error("open_pth: bad central directory entry");
    const uint16_t method = rd16(buf, p + 10);
    const uint32_t cs = rd32(buf, p + 20), us = rd32(buf, p + 24);
    const uint16_t nl = rd16(buf, p + 28), el = rd16(buf, p + 30),
                   cl = rd16(buf, p + 32);
    const uint32_t lh = rd32(buf, p + 42);
    if (p + 46 + nl > n)
      throw std::runtime_error("open_pth: central directory overrun");
    ZipEntry e;
    e.name.assign(buf.data() + p + 46, nl);
    e.method = method;
    e.comp_size = cs;
    e.uncomp_size = us;
    e.local_offset = lh;
    out.push_back(e);
    p += 46 + nl + el + cl;
  }
  return out;
}

// Data range of a stored entry: local header gives name/extra lengths.
std::pair<uint64_t, uint64_t> zip_data_range(const std::vector<char>& buf,
                                             const ZipEntry& e) {
  if (e.method != 0)
    throw std::runtime_error("open_pth: entry '" + e.name +
                             "' is compressed (only stored supported)");
  const size_t p = (size_t)e.local_offset;
  if (p + 30 > buf.size() || !(buf[p] == 'P' && buf[p + 1] == 'K' &&
                              buf[p + 2] == '\x03' && buf[p + 3] == '\x04'))
    throw std::runtime_error("open_pth: bad local header for '" + e.name +
                             "'");
  const uint16_t nl = rd16(buf, p + 26), el = rd16(buf, p + 28);
  const uint64_t start = (uint64_t)p + 30 + nl + el;
  if (start + e.comp_size > buf.size())
    throw std::runtime_error("open_pth: entry '" + e.name + "' overruns file");
  return {start, e.comp_size};
}

// ---------------------------------------------------------------------------
// Pickle VM.
// ---------------------------------------------------------------------------
struct Node;
using NodePtr = std::shared_ptr<Node>;
struct Node {
  enum class Kind { None, Bool, Int, Double, Unicode, Global, Tuple, Dict,
                    Storage, Tensor, Mark };
  Kind kind = Kind::None;
  bool flag = false;
  int64_t ival = 0;
  double dval = 0.0;
  std::string sval;                      // Unicode / Global "mod \x1f name"
  std::vector<NodePtr> items;            // Tuple
  std::vector<std::pair<std::string, NodePtr>> dict;  // Dict (ordered)
  // Storage: sval = storage key; ival = numel; dict-note = dtype in `extra`.
  // Tensor: items = {storage, offset, size-tuple, stride-tuple}; ival = unused.
  std::string extra;                     // Storage dtype / Global display
};

NodePtr mk(Node::Kind k) {
  NodePtr n = std::make_shared<Node>();
  n->kind = k;
  return n;
}

struct Cursor {
  const char* p;
  const char* end;
  void need(size_t n, const char* what) {
    if ((size_t)(end - p) < n)
      throw std::runtime_error(std::string("open_pth: truncated pickle (") +
                               what + ")");
  }
  uint8_t u8() {
    need(1, "u8");
    return (uint8_t)*p++;
  }
  uint16_t u16() {
    need(2, "u16");
    uint16_t v = (uint8_t)p[0] | ((uint8_t)p[1] << 8);
    p += 2;
    return v;
  }
  uint32_t u32() {
    need(4, "u32");
    uint32_t v = (uint8_t)p[0] | ((uint8_t)p[1] << 8) | ((uint8_t)p[2] << 16) |
                 ((uint8_t)p[3] << 24);
    p += 4;
    return v;
  }
  int32_t i32() { return (int32_t)u32(); }
  std::string bytes(size_t n) {
    need(n, "bytes");
    std::string s(p, n);
    p += n;
    return s;
  }
  std::string line() {  // up to and including '\n'
    const char* q = p;
    while (q != end && *q != '\n') ++q;
    if (q == end) throw std::runtime_error("open_pth: unterminated line");
    std::string s(p, (size_t)(q - p + 1));
    p = q + 1;
    return s;
  }
};

int64_t node_int(const NodePtr& n, const char* what) {
  if (!n || n->kind != Node::Kind::Int)
    throw std::runtime_error(std::string("open_pth: expected int for ") + what);
  return n->ival;
}

void parse_pickle(const std::vector<char>& data, NodePtr& root) {
  Cursor c{data.data(), data.data() + data.size()};
  std::vector<NodePtr> stack;
  std::vector<NodePtr> memo;
  auto pop = [&]() -> NodePtr {
    if (stack.empty()) throw std::runtime_error("open_pth: stack underflow");
    NodePtr n = stack.back();
    stack.pop_back();
    return n;
  };
  auto pop_mark = [&]() -> std::vector<NodePtr> {
    std::vector<NodePtr> items;
    while (!stack.empty() && stack.back()->kind != Node::Kind::Mark) {
      items.push_back(stack.back());
      stack.pop_back();
    }
    if (stack.empty()) throw std::runtime_error("open_pth: MARK not found");
    stack.pop_back();  // drop MARK
    // items were collected reversed; restore stack order (key,value,...).
    std::reverse(items.begin(), items.end());
    return items;
  };
  bool done = false;
  while (!done) {
    c.need(1, "opcode");
    const uint8_t op = c.u8();
    switch (op) {
      case 0x80: {  // PROTO
        const uint8_t ver = c.u8();
        if (ver > 4)
          throw std::runtime_error("open_pth: unsupported protocol");
        break;
      }
      case 'c': {  // GLOBAL: line module, line name
        std::string mod = c.line(), name = c.line();
        if (!mod.empty() && mod.back() == '\n') mod.pop_back();
        if (!name.empty() && name.back() == '\n') name.pop_back();
        NodePtr n = mk(Node::Kind::Global);
        n->sval = mod;
        n->extra = name;
        stack.push_back(n);
        break;
      }
      case 'R': {  // REDUCE
        NodePtr args = pop(), fn = pop();
        if (fn->kind != Node::Kind::Global)
          throw std::runtime_error("open_pth: REDUCE of non-global");
        if (fn->sval == "collections" && fn->extra == "OrderedDict") {
          if (args->kind != Node::Kind::Tuple || !args->items.empty())
            throw std::runtime_error(
                "open_pth: unexpected OrderedDict args");
          stack.push_back(mk(Node::Kind::Dict));
        } else if (fn->sval == "torch._utils" &&
                   fn->extra == "_rebuild_tensor_v2") {
          if (args->kind != Node::Kind::Tuple || args->items.size() != 6)
            throw std::runtime_error(
                "open_pth: bad _rebuild_tensor_v2 args");
          NodePtr t = mk(Node::Kind::Tensor);
          t->items = args->items;
          stack.push_back(t);
        } else {
          throw std::runtime_error("open_pth: unsupported REDUCE " + fn->sval +
                                   "." + fn->extra);
        }
        break;
      }
      case 'b': {  // BUILD
        NodePtr state = pop();
        if (stack.empty())
          throw std::runtime_error("open_pth: BUILD with empty stack");
        NodePtr inst = stack.back();
        if (state->kind == Node::Kind::None) break;  // no-op
        if (state->kind == Node::Kind::Dict &&
            inst->kind == Node::Kind::Dict) {
          // __dict__ state (e.g. {'version': 1}); informational only.
          break;
        }
        throw std::runtime_error("open_pth: unsupported BUILD state");
      }
      case 'Q': {  // BINPERSID
        NodePtr pid = pop();
        if (pid->kind != Node::Kind::Tuple || pid->items.size() != 5 ||
            pid->items[0]->kind != Node::Kind::Unicode ||
            pid->items[0]->sval != "storage" ||
            pid->items[1]->kind != Node::Kind::Global ||
            pid->items[2]->kind != Node::Kind::Unicode ||
            pid->items[4]->kind != Node::Kind::Int)
          throw std::runtime_error("open_pth: bad persistent id");
        NodePtr s = mk(Node::Kind::Storage);
        s->sval = pid->items[2]->sval;  // storage key
        s->ival = pid->items[4]->ival;  // numel
        s->extra = pid->items[1]->sval + "." + pid->items[1]->extra;
        stack.push_back(s);
        break;
      }
      case '}': stack.push_back(mk(Node::Kind::Dict)); break;  // EMPTY_DICT
      case ')': {  // EMPTY_TUPLE
        NodePtr n = mk(Node::Kind::Tuple);
        stack.push_back(n);
        break;
      }
      case 't': {  // TUPLE
        NodePtr n = mk(Node::Kind::Tuple);
        n->items = pop_mark();
        stack.push_back(n);
        break;
      }
      case 0x85: {  // TUPLE1
        NodePtr a = pop(), n = mk(Node::Kind::Tuple);
        n->items = {a};
        stack.push_back(n);
        break;
      }
      case 0x86: {  // TUPLE2
        NodePtr b = pop(), a = pop(), n = mk(Node::Kind::Tuple);
        n->items = {a, b};
        stack.push_back(n);
        break;
      }
      case 0x87: {  // TUPLE3
        NodePtr cc = pop(), b = pop(), a = pop(), n = mk(Node::Kind::Tuple);
        n->items = {a, b, cc};
        stack.push_back(n);
        break;
      }
      case '(': stack.push_back(mk(Node::Kind::Mark)); break;  // MARK
      case 's': {  // SETITEM
        NodePtr v = pop(), k = pop();
        if (stack.empty() || stack.back()->kind != Node::Kind::Dict)
          throw std::runtime_error("open_pth: SETITEM on non-dict");
        if (k->kind != Node::Kind::Unicode)
          throw std::runtime_error("open_pth: non-string dict key");
        stack.back()->dict.emplace_back(k->sval, v);
        break;
      }
      case 'u': {  // SETITEMS
        std::vector<NodePtr> items = pop_mark();
        if (items.size() % 2 != 0)
          throw std::runtime_error("open_pth: odd SETITEMS");
        if (stack.empty() || stack.back()->kind != Node::Kind::Dict)
          throw std::runtime_error("open_pth: SETITEMS on non-dict");
        for (size_t i = 0; i < items.size(); i += 2) {
          if (items[i]->kind != Node::Kind::Unicode)
            throw std::runtime_error("open_pth: non-string dict key");
          stack.back()->dict.emplace_back(items[i]->sval, items[i + 1]);
        }
        break;
      }
      case 'X': {  // BINUNICODE
        uint32_t n = c.u32();
        NodePtr s = mk(Node::Kind::Unicode);
        s->sval = c.bytes(n);
        stack.push_back(s);
        break;
      }
      case 'U': {  // SHORT_BINUNICODE
        uint8_t n = c.u8();
        NodePtr s = mk(Node::Kind::Unicode);
        s->sval = c.bytes(n);
        stack.push_back(s);
        break;
      }
      case 'J': {  // BININT
        NodePtr n = mk(Node::Kind::Int);
        n->ival = c.i32();
        stack.push_back(n);
        break;
      }
      case 'K': {  // BININT1
        NodePtr n = mk(Node::Kind::Int);
        n->ival = c.u8();
        stack.push_back(n);
        break;
      }
      case 'M': {  // BININT2
        NodePtr n = mk(Node::Kind::Int);
        n->ival = c.u16();
        stack.push_back(n);
        break;
      }
      case 'L': {  // LONG (decimal + 'L')
        std::string s = c.line();
        if (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (!s.empty() && s.back() == 'L') s.pop_back();
        NodePtr n = mk(Node::Kind::Int);
        try {
          n->ival = std::stoll(s);
        } catch (...) {
          throw std::runtime_error("open_pth: bad LONG '" + s + "'");
        }
        stack.push_back(n);
        break;
      }
      case 'G': {  // BINFLOAT (big-endian double)
        c.need(8, "double");
        uint64_t u = 0;
        for (int i = 0; i < 8; ++i) u = (u << 8) | (uint8_t)c.p[i];
        c.p += 8;
        double d;
        std::memcpy(&d, &u, 8);
        // Big-endian to host (host assumed little-endian like the file).
        uint64_t le = __builtin_bswap64(u);
        std::memcpy(&d, &le, 8);
        NodePtr n = mk(Node::Kind::Double);
        n->dval = d;
        stack.push_back(n);
        break;
      }
      case 'F': {  // FLOAT (ascii)
        std::string s = c.line();
        NodePtr n = mk(Node::Kind::Double);
        try {
          n->dval = std::stod(s);
        } catch (...) {
          throw std::runtime_error("open_pth: bad FLOAT");
        }
        stack.push_back(n);
        break;
      }
      case 'q': {  // BINPUT
        uint8_t idx = c.u8();
        if (memo.size() <= idx) memo.resize(idx + 1);
        memo[idx] = stack.back();
        break;
      }
      case 'r': {  // LONG_BINPUT
        uint32_t idx = c.u32();
        if (memo.size() <= idx) memo.resize(idx + 1);
        memo[idx] = stack.back();
        break;
      }
      case 'h': {  // BINGET
        uint8_t idx = c.u8();
        if (idx >= memo.size() || !memo[idx])
          throw std::runtime_error("open_pth: bad BINGET");
        stack.push_back(memo[idx]);
        break;
      }
      case 'j': {  // LONG_BINGET
        uint32_t idx = c.u32();
        if (idx >= memo.size() || !memo[idx])
          throw std::runtime_error("open_pth: bad LONG_BINGET");
        stack.push_back(memo[idx]);
        break;
      }
      case 'N': stack.push_back(mk(Node::Kind::None)); break;  // NONE
      case 0x88: {  // NEWTRUE
        NodePtr n = mk(Node::Kind::Bool);
        n->flag = true;
        stack.push_back(n);
        break;
      }
      case 0x89: {  // NEWFALSE
        NodePtr n = mk(Node::Kind::Bool);
        n->flag = false;
        stack.push_back(n);
        break;
      }
      case '.': done = true; break;  // STOP
      default: {
        char msg[64];
        std::snprintf(msg, sizeof(msg), "open_pth: unsupported opcode 0x%02x",
                      op);
        throw std::runtime_error(msg);
      }
    }
  }
  if (stack.size() != 1)
    throw std::runtime_error("open_pth: pickle left " +
                             std::to_string(stack.size()) + " stack items");
  root = stack.back();
}

// Walk the parsed structure, collecting tensor inventory with dotted names.
void collect(const NodePtr& n, const std::string& prefix,
             std::vector<PthTensorMeta>& out) {
  if (!n) return;
  if (n->kind == Node::Kind::Tensor) {
    if (n->items.size() != 6)
      throw std::runtime_error("open_pth: bad tensor record");
    NodePtr stor = n->items[0], off = n->items[1], sz = n->items[2],
            st = n->items[3];
    if (stor->kind != Node::Kind::Storage)
      throw std::runtime_error("open_pth: tensor without storage");
    if (sz->kind != Node::Kind::Tuple || st->kind != Node::Kind::Tuple)
      throw std::runtime_error("open_pth: bad tensor size/stride");
    PthTensorMeta m;
    m.name = prefix;
    m.storage_key = stor->sval;
    m.storage_numel = stor->ival;
    m.dtype = stor->extra;
    m.storage_offset = node_int(off, "storage_offset");
    for (auto& e : sz->items) m.shape.push_back(node_int(e, "size"));
    for (auto& e : st->items) m.stride.push_back(node_int(e, "stride"));
    m.numel = 1;
    for (auto d : m.shape) m.numel *= d;
    out.push_back(m);
    return;
  }
  if (n->kind == Node::Kind::Dict) {
    for (auto& kv : n->dict) {
      const std::string nm = prefix.empty() ? kv.first : prefix + "." + kv.first;
      collect(kv.second, nm, out);
    }
    return;
  }
  // Anything else at top level (e.g. a lone tensor handled above) is an
  // error unless it is the voice-pack single tensor... which is a Tensor.
  throw std::runtime_error("open_pth: unexpected top-level node");
}

}  // namespace

PthArchive open_pth(const std::string& path) {
  std::vector<char> buf = read_file(path);
  std::vector<ZipEntry> entries = zip_index(buf);
  // Discover the prefix: entry ending in "/data.pkl".
  std::string prefix;
  const ZipEntry* pkl = nullptr;
  for (auto& e : entries) {
    if (e.name.size() > 9 &&
        e.name.compare(e.name.size() - 9, 9, "/data.pkl") == 0) {
      prefix = e.name.substr(0, e.name.size() - 9);
      pkl = &e;
      break;
    }
  }
  if (!pkl) throw std::runtime_error("open_pth: no */data.pkl in '" + path + "'");
  // Byte order must be little (host assumption for the float records).
  bool have_bo = false;
  for (auto& e : entries) {
    if (e.name == prefix + "/byteorder") {
      auto r = zip_data_range(buf, e);
      if (r.second < 6 ||
          std::string(buf.data() + r.first, 6) != "little")
        throw std::runtime_error("open_pth: archive is not little-endian");
      have_bo = true;
    }
  }
  if (!have_bo) throw std::runtime_error("open_pth: no byteorder entry");
  auto pr = zip_data_range(buf, *pkl);
  std::vector<char> pkl_bytes(buf.data() + pr.first,
                              buf.data() + pr.first + pr.second);
  NodePtr root;
  parse_pickle(pkl_bytes, root);
  PthArchive a;
  a.path = path;
  a.prefix = prefix;
  if (root->kind == Node::Kind::Tensor) {
    // Single-tensor pack (voice .pt): inventory holds one nameless entry.
    std::vector<PthTensorMeta> tmp;
    collect(root, "", tmp);
    a.tensors = tmp;
  } else if (root->kind == Node::Kind::Dict) {
    collect(root, "", a.tensors);
  } else {
    throw std::runtime_error("open_pth: unexpected pickle root");
  }
  // Map storage records.
  const std::string dprefix = prefix + "/data/";
  for (auto& e : entries) {
    if (e.name.size() > dprefix.size() &&
        e.name.compare(0, dprefix.size(), dprefix) == 0) {
      auto r = zip_data_range(buf, e);
      a.records[e.name.substr(dprefix.size())] = r;
    }
  }
  return a;
}

int64_t pth_total_params(const PthArchive& archive) {
  int64_t n = 0;
  for (auto& t : archive.tensors) n += t.numel;
  return n;
}

std::vector<float> read_pth_storage(const PthArchive& archive,
                                    const std::string& storage_key) {
  auto it = archive.records.find(storage_key);
  if (it == archive.records.end())
    throw std::runtime_error("read_pth_storage: no record '" + storage_key +
                             "'");
  std::ifstream f(archive.path, std::ios::binary);
  if (!f) throw std::runtime_error("read_pth_storage: cannot reopen '" +
                                   archive.path + "'");
  f.seekg((long long)it->second.first, std::ios::beg);
  const size_t nbytes = (size_t)it->second.second;
  if (nbytes % 4 != 0)
    throw std::runtime_error("read_pth_storage: record '" + storage_key +
                             "' not a multiple of 4 bytes");
  std::vector<float> out(nbytes / 4);
  f.read(reinterpret_cast<char*>(out.data()), (long long)nbytes);
  if (!f) throw std::runtime_error("read_pth_storage: short read '" +
                                   storage_key + "'");
  return out;
}

}  // namespace Kokoro
}  // namespace KokkosDNN
