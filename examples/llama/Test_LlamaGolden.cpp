// Golden real-weight validation for the Llama example.
//
// README TODO "Real-weight validation": golden logits/perplexity check
// against HF transformers on 1-2 prompts. The tiny-random tests below run
// always (no weights needed); the real-weight tests are gated on env vars
// and SKIP gracefully when files are absent:
//
//   KOKKOSDNN_WEIGHTS         whitespace/colon/comma-separated list of
//                             .safetensors shard(s)
//   KOKKOSDNN_TOKENIZER_JSON  Hugging Face tokenizer.json
//   KOKKOSDNN_GOLDEN_JSON     golden file from examples/llama/generate_golden.py
//   KOKKOSDNN_GOLDEN_TOL      optional max-abs-diff override (float)
//   KOKKOSDNN_GOLDEN_FP32=1   also run the FP32 real-weight path (needs
//                             ~32 GiB RAM for 8B FP32 weights; otherwise only
//                             the bfloat16 path runs)
//
// NEVER download weights from tests: without the env vars (or with
// unreadable files) every real-weight test skips with an explanatory
// message.
//
// Tolerances (documented, overridable via KOKKOSDNN_GOLDEN_TOL):
//   BF16 real-weight: max-abs-diff over golden top-k <= 0.5, NLL within
//     0.5, prefill argmax must match exactly (greedy-decode correctness).
//   FP32 real-weight: max-abs-diff <= 0.05, NLL within 0.05, argmax exact.
// Rationale: BF16 has ~8 mantissa bits (eps ~0.0078); two independent
// kernels accumulate different rounding over 32 layers of 4096-wide
// reductions, so the absolute bar is generous while argmax equality stays
// strict. FP32 paths differ only by reduction order (~1e-6 relative on
// O(10)-magnitude logits), so that bar is tight.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_LlamaConfig.hpp"
#include "KokkosDNN_LlamaModel.hpp"
#include "KokkosDNN_LlamaWeights.hpp"
#include "KokkosDNN_Tokenizer.hpp"

namespace {

const char* getenv_str(const char* name) {
  const char* p = std::getenv(name);
  return (p && *p) ? p : nullptr;
}

// Split a file list on whitespace, ',', ';' or ':'.
std::vector<std::string> split_list(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',' ||
        c == ';' || c == ':') {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
    } else {
      cur += c;
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

bool file_readable(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

// ---- Minimal JSON parser (objects/arrays/strings/numbers/literals) ----
// Only what the golden schema needs; throws std::runtime_error on bad
// input. No third-party dependency (mirrors the loader's minimal-JSON
// approach in KokkosDNN_LlamaWeights.cpp).
struct JNode {
  enum class Type { Null, Bool, Num, Str, Arr, Obj } type = Type::Null;
  bool boolean = false;
  double num = 0;
  std::string str;
  std::vector<JNode> arr;
  std::vector<std::pair<std::string, JNode>> obj;

  bool has(const std::string& k) const {
    for (auto& kv : obj)
      if (kv.first == k) return true;
    return false;
  }
  const JNode& at(const std::string& k) const {
    for (auto& kv : obj)
      if (kv.first == k) return kv.second;
    throw std::runtime_error("golden: missing key '" + k + "'");
  }
};

class JParser {
 public:
  explicit JParser(const std::string& s) : s_(s) {}
  JNode parse() {
    JNode v = value();
    skip();
    if (pos_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  const std::string& s_;
  size_t pos_ = 0;
  [[noreturn]] void fail(const std::string& m) {
    throw std::runtime_error("golden: JSON parse error at byte " +
                             std::to_string(pos_) + ": " + m);
  }
  void skip() {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' ||
            s_[pos_] == '\r'))
      ++pos_;
  }
  char peek() {
    if (pos_ >= s_.size()) fail("unexpected end");
    return s_[pos_];
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos_;
  }
  JNode value() {
    skip();
    char c = peek();
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') {
      JNode n;
      n.type = JNode::Type::Str;
      n.str  = string();
      return n;
    }
    if (c == 't' || c == 'f') return boolean();
    if (c == 'n') return null();
    return number();
  }
  JNode object() {
    JNode n;
    n.type = JNode::Type::Obj;
    expect('{');
    skip();
    if (peek() == '}') {
      ++pos_;
      return n;
    }
    while (true) {
      skip();
      if (peek() != '"') fail("expected string key");
      std::string k = string();
      skip();
      expect(':');
      n.obj.emplace_back(k, value());
      skip();
      char c = peek();
      if (c == ',') {
        ++pos_;
        continue;
      }
      if (c == '}') {
        ++pos_;
        break;
      }
      fail("expected ',' or '}'");
    }
    return n;
  }
  JNode array() {
    JNode n;
    n.type = JNode::Type::Arr;
    expect('[');
    skip();
    if (peek() == ']') {
      ++pos_;
      return n;
    }
    while (true) {
      n.arr.push_back(value());
      skip();
      char c = peek();
      if (c == ',') {
        ++pos_;
        continue;
      }
      if (c == ']') {
        ++pos_;
        break;
      }
      fail("expected ',' or ']'");
    }
    return n;
  }
  void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
      out += (char)cp;
    } else if (cp < 0x800) {
      out += (char)(0xC0 | (cp >> 6));
      out += (char)(0x80 | (cp & 0x3F));
    } else {
      out += (char)(0xE0 | (cp >> 12));
      out += (char)(0x80 | ((cp >> 6) & 0x3F));
      out += (char)(0x80 | (cp & 0x3F));
    }
  }
  unsigned hex4() {
    if (pos_ + 4 > s_.size()) fail("bad \\u escape");
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = s_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
      else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
      else fail("bad \\u escape");
    }
    return v;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= s_.size()) fail("unterminated string");
      char c = s_[pos_++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= s_.size()) fail("bad escape");
      char e = s_[pos_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned cp = hex4();
          // Basic surrogate-pair support.
          if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 2 <= s_.size() &&
              s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
            pos_ += 2;
            unsigned lo = hex4();
            if (lo >= 0xDC00 && lo <= 0xDFFF)
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          if (cp >= 0x10000) {
            // Encode astral planes as 4-byte UTF-8.
            unsigned v = cp;
            out += (char)(0xF0 | (v >> 18));
            out += (char)(0x80 | ((v >> 12) & 0x3F));
            out += (char)(0x80 | ((v >> 6) & 0x3F));
            out += (char)(0x80 | (v & 0x3F));
          } else {
            append_utf8(out, cp);
          }
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }
  JNode boolean() {
    JNode n;
    n.type = JNode::Type::Bool;
    if (s_.compare(pos_, 4, "true") == 0) {
      n.boolean = true;
      pos_ += 4;
    } else if (s_.compare(pos_, 5, "false") == 0) {
      n.boolean = false;
      pos_ += 5;
    } else {
      fail("bad literal");
    }
    return n;
  }
  JNode null() {
    if (s_.compare(pos_, 4, "null") != 0) fail("bad literal");
    pos_ += 4;
    return JNode{};
  }
  JNode number() {
    const char* begin = s_.data() + pos_;
    char* end         = nullptr;
    double v          = std::strtod(begin, &end);
    if (end == begin) fail("bad number");
    pos_ = (size_t)(end - s_.data());
    JNode n;
    n.type = JNode::Type::Num;
    n.num  = v;
    return n;
  }
};

// ---- Golden file schema (written by examples/llama/generate_golden.py) ----
struct GoldenEntry {
  std::string prompt;
  std::vector<int> input_ids;
  std::vector<int> top_ids;
  std::vector<double> top_logits;
  int argmax   = -1;
  double nll   = 0;
  bool has_next = false;
  int next_id   = -1;
  std::vector<int> next_top_ids;
  std::vector<double> next_top_logits;
  double next_nll = 0;
};

struct GoldenFile {
  std::string model, dtype;
  int top_k = 0;
  std::vector<GoldenEntry> entries;
};

std::vector<int> jint_array(const JNode& n, const std::string& ctx) {
  if (n.type != JNode::Type::Arr)
    throw std::runtime_error("golden: '" + ctx + "' is not an array");
  std::vector<int> out;
  for (auto& e : n.arr) {
    if (e.type != JNode::Type::Num)
      throw std::runtime_error("golden: '" + ctx + "' has non-number entry");
    out.push_back((int)std::llround(e.num));
  }
  return out;
}

std::vector<double> jnum_array(const JNode& n, const std::string& ctx) {
  if (n.type != JNode::Type::Arr)
    throw std::runtime_error("golden: '" + ctx + "' is not an array");
  std::vector<double> out;
  for (auto& e : n.arr) {
    if (e.type != JNode::Type::Num)
      throw std::runtime_error("golden: '" + ctx + "' has non-number entry");
    out.push_back(e.num);
  }
  return out;
}

GoldenFile parse_golden(const std::string& text) {
  GoldenFile g;
  JNode root = JParser(text).parse();
  if (root.type != JNode::Type::Obj)
    throw std::runtime_error("golden: root is not an object");
  if (root.has("model") && root.at("model").type == JNode::Type::Str)
    g.model = root.at("model").str;
  if (root.has("dtype") && root.at("dtype").type == JNode::Type::Str)
    g.dtype = root.at("dtype").str;
  const JNode& entries = root.at("entries");
  if (entries.type != JNode::Type::Arr)
    throw std::runtime_error("golden: 'entries' is not an array");
  for (auto& en : entries.arr) {
    GoldenEntry e;
    if (en.type != JNode::Type::Obj)
      throw std::runtime_error("golden: entry is not an object");
    if (en.at("prompt").type != JNode::Type::Str)
      throw std::runtime_error("golden: entry prompt is not a string");
    e.prompt      = en.at("prompt").str;
    e.input_ids   = jint_array(en.at("input_ids"), "input_ids");
    e.top_ids     = jint_array(en.at("top_ids"), "top_ids");
    e.top_logits  = jnum_array(en.at("top_logits"), "top_logits");
    e.argmax      = (int)std::llround(en.at("argmax").num);
    e.nll         = en.at("nll").num;
    if (e.top_ids.size() != e.top_logits.size() || e.top_ids.empty())
      throw std::runtime_error("golden: top_ids/top_logits length mismatch");
    if (en.has("next_id")) {
      e.has_next         = true;
      e.next_id          = (int)std::llround(en.at("next_id").num);
      e.next_nll         = en.at("next_nll").num;
      e.next_top_ids     = jint_array(en.at("next_top_ids"), "next_top_ids");
      e.next_top_logits  = jnum_array(en.at("next_top_logits"),
                                      "next_top_logits");
      if (e.next_top_ids.size() != e.next_top_logits.size())
        throw std::runtime_error("golden: next_top length mismatch");
    }
    g.entries.push_back(std::move(e));
  }
  if (root.has("top_k")) g.top_k = (int)std::llround(root.at("top_k").num);
  if (g.entries.empty()) throw std::runtime_error("golden: no entries");
  return g;
}

GoldenFile load_golden_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("golden: cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return parse_golden(ss.str());
}

// ---- Comparison math (full-vocab host logits) ----
int argmax_host(const std::vector<float>& v) {
  int best = 0;
  for (size_t i = 1; i < v.size(); ++i)
    if (v[i] > v[best]) best = (int)i;
  return best;
}

// NLL of token id under logits (double precision): logsumexp - logit[id].
// Perplexity over the golden prompts = exp(mean NLL); reported, not
// asserted (the asserted quantities are argmax, top-k max-abs-diff, NLL).
double nll_of(const std::vector<float>& logits, int id) {
  double m = logits[0];
  for (float x : logits) m = std::max(m, (double)x);
  double se = 0;
  for (float x : logits) se += std::exp((double)x - m);
  return (m + std::log(se)) - (double)logits[(size_t)id];
}

// Run Model::forward on host ids, return full-vocab logits as floats.
template <class Scalar>
std::vector<float> forward_copy(
    KokkosDNN::Llama::Model<Kokkos::DefaultExecutionSpace, Scalar>& model,
    const std::vector<int>& ids) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<int*, Exec> toks("toks", ids.size());
  auto h = Kokkos::create_mirror_view(toks);
  for (size_t i = 0; i < ids.size(); ++i) h(i) = ids[i];
  Kokkos::deep_copy(toks, h);
  auto lg = model.forward(toks);
  auto hl = Kokkos::create_mirror_view(lg);
  Kokkos::deep_copy(hl, lg);
  std::vector<float> out(hl.extent(0));
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<float>(hl(i));
  return out;
}

// Shared prefill+decode comparison used by the real-weight tests (8B
// config) after weights are loaded.
template <class Scalar>
void expect_entry_match(const std::vector<float>& prefill_logits,
                        const GoldenEntry& ge, float tol, size_t idx) {
  int ours_argmax = argmax_host(prefill_logits);
  EXPECT_EQ(ours_argmax, ge.argmax)
      << "prefill argmax mismatch on entry " << idx << " ('" << ge.prompt
      << "'): greedy continuation would diverge from HF";
  double maxdiff = 0;
  for (size_t i = 0; i < ge.top_ids.size(); ++i) {
    int id = ge.top_ids[i];
    ASSERT_GE(id, 0);
    ASSERT_LT((size_t)id, prefill_logits.size());
    maxdiff =
        std::max(maxdiff, std::abs((double)prefill_logits[(size_t)id] -
                                   ge.top_logits[i]));
  }
  EXPECT_LE(maxdiff, (double)tol)
      << "max-abs-diff over golden top-" << ge.top_ids.size() << " on entry "
      << idx;
  EXPECT_NEAR(nll_of(prefill_logits, ge.argmax), ge.nll, (double)tol)
      << "NLL mismatch on entry " << idx;
}

// ---- Real-weight gated check ----
template <class Scalar>
void check_real_weights(float default_tol, const std::string& tag) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  const char* wenv = getenv_str("KOKKOSDNN_WEIGHTS");
  const char* tenv = getenv_str("KOKKOSDNN_TOKENIZER_JSON");
  const char* genv = getenv_str("KOKKOSDNN_GOLDEN_JSON");
  if (!wenv || !tenv || !genv) {
    GTEST_SKIP() << "real-weight golden check (" << tag << ") needs "
                 << "KOKKOSDNN_WEIGHTS + KOKKOSDNN_TOKENIZER_JSON + "
                 << "KOKKOSDNN_GOLDEN_JSON; skipping (tests never download "
                 << "weights). Generate the golden file with "
                 << "examples/llama/generate_golden.py.";
  }
  auto wfiles = split_list(wenv);
  if (wfiles.empty())
    GTEST_SKIP() << "KOKKOSDNN_WEIGHTS parsed to zero files; skipping.";
  for (auto& f : wfiles)
    if (!file_readable(f))
      GTEST_SKIP() << "weights file not readable, skipping: " << f;
  if (!file_readable(tenv))
    GTEST_SKIP() << "tokenizer file not readable, skipping: " << tenv;
  if (!file_readable(genv))
    GTEST_SKIP() << "golden file not readable, skipping: " << genv;

  float tol = default_tol;
  if (const char* t = getenv_str("KOKKOSDNN_GOLDEN_TOL")) tol = std::stof(t);

  GoldenFile golden = load_golden_file(genv);  // throws -> FAIL (corrupt file)
  Tokenizer tok(tenv);                         // throws -> FAIL
  Config cfg = Config::llama_3_1_8b();         // 8B shapes/dtypes coverage
  Weights<Exec, Scalar> w;
  load_safetensors(cfg, wfiles, w);  // throws -> FAIL
  Model<Exec, Scalar> model(cfg, w);

  std::printf("[golden:%s] model=%s golden=%s entries=%zu tol=%g\n",
              tag.c_str(), golden.model.c_str(), genv, golden.entries.size(),
              tol);
  double nll_sum = 0;
  int nll_n      = 0;
  const size_t n = std::min<size_t>(golden.entries.size(), 2);
  for (size_t e = 0; e < n; ++e) {
    const GoldenEntry& ge = golden.entries[e];
    // Tokenizer/prompt guard: encode must reproduce the golden input ids
    // (add_bos=true default matches generate_golden.py add_special_tokens).
    std::vector<int> ids = tok.encode(ge.prompt);
    ASSERT_EQ(ids, ge.input_ids)
        << "entry " << e << ": tokenizer output != golden input_ids "
        << "(wrong tokenizer.json or prompt mismatch?)";
    model.reset();
    std::vector<float> ours = forward_copy(model, ids);
    ASSERT_EQ(ours.size(), (size_t)cfg.vocab_size);
    expect_entry_match<Scalar>(ours, ge, tol, e);
    nll_sum += nll_of(ours, ge.argmax);
    ++nll_n;
    // Single-token decode step (T=1 GQA decode at full 8B geometry —
    // the FP32-cuDNN-fallback shape class noted in
    // KokkosDNN_Attention.hpp): the greedy continuation must match HF.
    if (ge.has_next) {
      std::vector<float> ours2 = forward_copy(model, {ge.argmax});
      ASSERT_EQ(ours2.size(), (size_t)cfg.vocab_size);
      EXPECT_EQ(argmax_host(ours2), ge.next_id)
          << "decode (T=1) argmax mismatch on entry " << e;
      EXPECT_NEAR(nll_of(ours2, ge.next_id), ge.next_nll, (double)tol)
          << "decode NLL mismatch on entry " << e;
      nll_sum += nll_of(ours2, ge.next_id);
      ++nll_n;
    }
  }
  std::printf("[golden:%s] mean NLL=%.4f perplexity=%.4f over %d next-token(s)\n",
              tag.c_str(), nll_sum / nll_n, std::exp(nll_sum / nll_n), nll_n);
}

}  // namespace

// 8B-shape/dtype coverage without weights: float vs bhalf forward parity
// on tiny_test (random weights, fixed seed). The real-weight tests below
// cover 8B shapes; this covers dtype parity where Serial can afford both.
TEST(LlamaGolden, DtypeParityTiny) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = Kokkos::Experimental::bhalf_t;
  using namespace KokkosDNN::Llama;
  auto cfg = Config::tiny_test();
  Weights<Exec, float> wf;
  Weights<Exec, Scalar> wb;
  random_init(cfg, wf, 1234);
  random_init(cfg, wb, 1234);
  Model<Exec, float> mf(cfg, wf);
  Model<Exec, Scalar> mb(cfg, wb);
  std::vector<int> prompt = {1, 2, 3, 4};
  std::vector<float> lf   = forward_copy(mf, prompt);
  std::vector<float> lb   = forward_copy(mb, prompt);
  ASSERT_EQ(lf.size(), lb.size());
  double maxdiff = 0;
  for (size_t i = 0; i < lf.size(); ++i)
    maxdiff = std::max(maxdiff, std::abs((double)lf[i] - (double)lb[i]));
  int top_f = argmax_host(lf), top_b = argmax_host(lb);
  std::printf("[parity] tiny vocab=%zu max-abs-diff=%.5f top1_f=%d top1_b=%d\n",
              lf.size(), maxdiff, top_f, top_b);
  // BF16 rounding over a 2-layer tiny model is ~0 here (measured 0.0 on
  // Serial); 0.1 leaves headroom across backends.
  EXPECT_LT(maxdiff, 0.1);
  EXPECT_EQ(top_f, top_b);
}

// FP32 T=1 GQA fallback shape class at 8B head geometry (Hq=32, Hkv=8,
// D=128; see the cuDNN <= 9.27 note in KokkosDNN_Attention.hpp):
// default-dispatched vs forced-native must agree. On Serial/CPU builds
// both take the native kernel (agreement = determinism); on CUDA+cuDNN
// builds the FP32 T=1 GQA case exercises the fallback guard — without it
// the composite path would corrupt non-leading group heads.
TEST(LlamaGolden, Attention8BDecodeFallbackMatchesNative) {
  using Exec = Kokkos::DefaultExecutionSpace;
  constexpr int Hq = 32, Hkv = 8, D = 128;
  const float scale = 0.0883883476f;  // 1/sqrt(128), matches Config
  auto check = [&](auto tag, int T, int Ktot, size_t q_off, float tol) {
    using Scalar = typename decltype(tag)::type;
    KokkosDNN::DNNHandle<Exec> h_on;  // default dispatch (cuDNN when built)
    KokkosDNN::DNNHandle<Exec> h_off;  // forced native
    h_off.use_cudnn = false;
    // LayoutRight throughout so a cuDNN build can engage (as in
    // Test_Attention.cpp's engagement test).
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> q("q", T, Hq, D),
        k("k", Ktot, Hkv, D), v("v", Ktot, Hkv, D), o_dis("odis", T, Hq, D),
        o_nat("onat", T, Hq, D);
    auto hq = Kokkos::create_mirror_view(q);
    auto hk = Kokkos::create_mirror_view(k);
    auto hv = Kokkos::create_mirror_view(v);
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < Hq; ++h)
        for (int d = 0; d < D; ++d)
          hq(t, h, d) =
              static_cast<Scalar>(0.05f * ((t * 7 + h * 13 + d * 3) % 17) -
                                  0.2f);
    for (int kk = 0; kk < Ktot; ++kk)
      for (int h = 0; h < Hkv; ++h)
        for (int d = 0; d < D; ++d) {
          hk(kk, h, d) =
              static_cast<Scalar>(0.05f * ((kk * 5 + h * 11 + d) % 19) - 0.2f);
          hv(kk, h, d) =
              static_cast<Scalar>(0.05f * ((kk * 3 + h * 17 + d * 2) % 23) -
                                  0.3f);
        }
    Kokkos::deep_copy(q, hq);
    Kokkos::deep_copy(k, hk);
    Kokkos::deep_copy(v, hv);
    KokkosDNN::attention_gqa_single<Exec, Scalar>(h_on, q, k, v, q_off, o_dis,
                                                  scale);
    KokkosDNN::attention_gqa_single<Exec, Scalar>(h_off, q, k, v, q_off, o_nat,
                                                  scale);
    Kokkos::fence();
    auto gd = Kokkos::create_mirror_view(o_dis);
    auto gn = Kokkos::create_mirror_view(o_nat);
    Kokkos::deep_copy(gd, o_dis);
    Kokkos::deep_copy(gn, o_nat);
    double maxdiff = 0;
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < Hq; ++h)
        for (int d = 0; d < D; ++d)
          maxdiff = std::max(maxdiff, std::abs((double)gd(t, h, d) -
                                               (double)gn(t, h, d)));
    std::printf("[fallback] T=%d K=%d q_off=%zu max|dispatched-native|=%.3g\n",
                T, Ktot, q_off, maxdiff);
    EXPECT_LE(maxdiff, (double)tol);
  };
  struct FTag {
    using type = float;
  };
  struct BTag {
    using type = Kokkos::Experimental::bhalf_t;
  };
  // T=1 GQA decode: THE fallback shape class (float), both scalars.
  check(FTag{}, 1, 17, 16, 1e-4f);
  check(BTag{}, 1, 17, 16, 5e-2f);
  // Short prefill at the same geometry for completeness.
  check(FTag{}, 4, 20, 16, 1e-4f);
  check(BTag{}, 4, 20, 16, 5e-2f);
}

// Parser self-test (always runs): guards schema drift between
// generate_golden.py and parse_golden without needing weights.
TEST(LlamaGolden, GoldenParserFixture) {
  const std::string js = R"({
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "dtype": "bfloat16",
    "top_k": 4,
    "entries": [
      {"prompt": "Hi \"there\"\n",
       "input_ids": [128000, 9906],
       "top_ids": [1, 2, 3, 4],
       "top_logits": [9.5, 8.25, -1.0, 0.0],
       "argmax": 1, "nll": 0.75,
       "next_id": 2, "next_top_ids": [2, 1],
       "next_top_logits": [7.0, 6.5], "next_nll": 0.5},
      {"prompt": "caf\u00e9",
       "input_ids": [128000, 936, 59958],
       "top_ids": [5, 6],
       "top_logits": [3.0, 2.0],
       "argmax": 5, "nll": 1.25}
    ]
  })";
  GoldenFile g = parse_golden(js);
  ASSERT_EQ(g.entries.size(), 2u);
  EXPECT_EQ(g.model, "meta-llama/Llama-3.1-8B-Instruct");
  EXPECT_EQ(g.entries[0].prompt, "Hi \"there\"\n");
  EXPECT_EQ(g.entries[0].input_ids, (std::vector<int>{128000, 9906}));
  EXPECT_EQ(g.entries[0].top_ids, (std::vector<int>{1, 2, 3, 4}));
  EXPECT_DOUBLE_EQ(g.entries[0].top_logits[1], 8.25);
  EXPECT_EQ(g.entries[0].argmax, 1);
  EXPECT_DOUBLE_EQ(g.entries[0].nll, 0.75);
  ASSERT_TRUE(g.entries[0].has_next);
  EXPECT_EQ(g.entries[0].next_id, 2);
  EXPECT_DOUBLE_EQ(g.entries[0].next_nll, 0.5);
  EXPECT_EQ(g.entries[1].prompt, "caf\xc3\xa9");  // \u00e9 -> UTF-8
  EXPECT_FALSE(g.entries[1].has_next);
  EXPECT_THROW(parse_golden(R"({"entries": []})"), std::runtime_error);
  EXPECT_THROW(parse_golden("{not json"), std::runtime_error);
}

// Harness dry-run (always runs): the comparison math (argmax / top-k
// max-abs-diff / NLL) is self-consistent on a tiny random model, where the
// golden entry is derived via independent long-double reference math.
TEST(LlamaGolden, HarnessDryRunTiny) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto cfg = Config::tiny_test();
  Weights<Exec, float> w;
  random_init(cfg, w, 1234);
  Model<Exec, float> m(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};
  std::vector<float> ours = forward_copy(m, prompt);
  // Independent reference in long double.
  long double mx = ours[0];
  for (float x : ours) mx = std::max(mx, (long double)x);
  long double se = 0;
  for (float x : ours) se += expl((long double)x - mx);
  long double ref_lse = mx + logl(se);
  int ref_argmax      = 0;
  for (size_t i = 1; i < ours.size(); ++i)
    if (ours[i] > ours[(size_t)ref_argmax]) ref_argmax = (int)i;
  GoldenEntry ge;
  ge.prompt    = "dry-run";
  ge.input_ids = prompt;
  ge.argmax    = ref_argmax;
  ge.nll       = (double)(ref_lse - (long double)ours[(size_t)ref_argmax]);
  // top-8 via a simple selection (independent of any helper).
  std::vector<int> idx(ours.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
  std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(),
                    [&](int a, int b) { return ours[(size_t)a] > ours[(size_t)b]; });
  for (int i = 0; i < 8; ++i) {
    ge.top_ids.push_back(idx[(size_t)i]);
    ge.top_logits.push_back((double)ours[(size_t)idx[(size_t)i]]);
  }
  expect_entry_match<float>(ours, ge, 1e-6f, 0);
  EXPECT_NEAR(nll_of(ours, ge.argmax), ge.nll, 1e-9);
}

// Real-weight golden checks (gated; SKIP without files, never download).
TEST(LlamaGolden, RealWeightsBHalf) {
  check_real_weights<Kokkos::Experimental::bhalf_t>(0.5f, "bf16");
}

TEST(LlamaGolden, RealWeightsFp32) {
  const char* f = getenv_str("KOKKOSDNN_GOLDEN_FP32");
  if (!f || std::string(f) != "1") {
    GTEST_SKIP() << "set KOKKOSDNN_GOLDEN_FP32=1 to run the FP32 real-weight "
                 << "path (needs ~32 GiB RAM for 8B FP32 weights).";
  }
  check_real_weights<float>(0.05f, "fp32");
}
