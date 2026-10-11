// Golden real-weight validation for the DINOv3 example.
//
// README TODO "Golden test vs HF": CLS + patch-token check against HF
// transformers on facebook/dinov3-vitb16-pretrain-lvd1689m. The parser and
// harness tests below run always (no weights needed); the real-weight test
// is gated on env vars and SKIPs gracefully when files are absent:
//
//   KOKKOSDNN_DINOV3_WEIGHTS  .safetensors checkpoint (whitespace/colon/
//                             comma-separated shard list; the ViT-B
//                             checkpoint is a single shard)
//   KOKKOSDNN_DINOV3_GOLDEN_JSON
//                             golden file from examples/dinov3/generate_golden.py
//   KOKKOSDNN_DINOV3_TOL      optional max-abs-diff override (float)
//   KOKKOSDNN_DINOV3_CONFIG   optional HF config.json overlay (defaults to
//                             Config::vitb16(), which already matches the
//                             checkpoint arch)
//
// NEVER download weights from tests: without the env vars (or with
// unreadable files) the real-weight test skips with an explanatory
// message.
//
// Tolerances (documented, overridable via KOKKOSDNN_DINOV3_TOL):
//   FP32 real-weight: max-abs-diff over CLS + first-4 patch rows +
//     patch mean <= 1e-3. Rationale: both sides run FP32 with the same
//     math (LayerNorm eps=1e-5, exact-erf GELU, FP32 streaming softmax),
//     so differences are only reduction-order rounding over 12 layers of
//     768-wide reductions; 1e-3 leaves ~100x headroom over the observed
//     ~1e-5-level diffs.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "KokkosDNN_Dinov3Config.hpp"
#include "KokkosDNN_Dinov3Model.hpp"
#include "KokkosDNN_Dinov3Weights.hpp"

namespace {

namespace D3 = KokkosDNN::Dinov3;

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
// approach).
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
          if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 2 <= s_.size() &&
              s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
            pos_ += 2;
            unsigned lo = hex4();
            if (lo >= 0xDC00 && lo <= 0xDFFF)
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          if (cp >= 0x10000) {
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

// ---- Golden file schema (written by examples/dinov3/generate_golden.py) ----
struct GoldenFile {
  std::string model, dtype, input, synthetic_formula;
  int image_size = 0, hidden_size = 0;
  int num_register_tokens = 0, num_patch_tokens = 0;
  int pix_c = 0, pix_h = 0, pix_w = 0;
  std::vector<double> pixels;  // flat CHW float32 reference input
  std::vector<double> cls;     // [D]
  std::vector<std::vector<double>> patch_first4;  // 4 x [D]
  std::vector<double> patch_mean;                  // [D]
};

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
  if (root.has("input") && root.at("input").type == JNode::Type::Str)
    g.input = root.at("input").str;
  if (root.has("synthetic_formula") &&
      root.at("synthetic_formula").type == JNode::Type::Str)
    g.synthetic_formula = root.at("synthetic_formula").str;
  g.image_size          = (int)std::llround(root.at("image_size").num);
  g.hidden_size         = (int)std::llround(root.at("hidden_size").num);
  g.num_register_tokens = (int)std::llround(root.at("num_register_tokens").num);
  g.num_patch_tokens    = (int)std::llround(root.at("num_patch_tokens").num);
  const JNode& pix      = root.at("pixels");
  g.pix_c               = (int)std::llround(pix.at("channels").num);
  g.pix_h               = (int)std::llround(pix.at("height").num);
  g.pix_w               = (int)std::llround(pix.at("width").num);
  g.pixels              = jnum_array(pix.at("values"), "pixels.values");
  g.cls                 = jnum_array(root.at("cls"), "cls");
  const JNode& pf       = root.at("patch_first4");
  if (pf.type != JNode::Type::Arr || pf.arr.size() != 4)
    throw std::runtime_error("golden: 'patch_first4' must have 4 rows");
  for (auto& row : pf.arr) g.patch_first4.push_back(jnum_array(row, "patch row"));
  g.patch_mean = jnum_array(root.at("patch_mean"), "patch_mean");
  if ((int)g.cls.size() != g.hidden_size)
    throw std::runtime_error("golden: cls length != hidden_size");
  if ((int)g.patch_mean.size() != g.hidden_size)
    throw std::runtime_error("golden: patch_mean length != hidden_size");
  if ((int)g.pixels.size() != g.pix_c * g.pix_h * g.pix_w)
    throw std::runtime_error("golden: pixels size != CxHxW");
  return g;
}

GoldenFile load_golden_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("golden: cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return parse_golden(ss.str());
}

// Canonical synthetic pixels (must match dinov3_encode.cpp,
// Test_Dinov3Smoke.cpp and generate_golden.py --synthetic):
//   v(c,y,x) = ((y*S + x) % 256) / 255 + 0.01 * c
template <class Exec, class Scalar>
Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> make_synthetic_pixels(
    int C, int S) {
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> pixels("pixels", (size_t)C,
                                                            (size_t)S,
                                                            (size_t)S);
  auto dst = pixels;
  Kokkos::parallel_for(
      "dinov3_synthetic",
      Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>({0, 0, 0}, {(size_t)C,
                                                              (size_t)S,
                                                              (size_t)S}),
      KOKKOS_LAMBDA(size_t c, size_t y, size_t x) {
        float v =
            static_cast<float>((y * (size_t)S + x) % 256) * (1.0f / 255.0f) +
            0.01f * static_cast<float>(c);
        dst(c, y, x) = static_cast<Scalar>(v);
      });
  Kokkos::fence();
  return pixels;
}

double max_abs_diff(const std::vector<float>& a,
                    const std::vector<double>& b) {
  if (a.size() != b.size())
    throw std::runtime_error("max_abs_diff: size mismatch");
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs((double)a[i] - b[i]));
  return m;
}

// ---- Real-weight gated check (FP32; the checkpoint torch_dtype) ----
void check_real_weights_fp32(float default_tol) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const char* wenv = getenv_str("KOKKOSDNN_DINOV3_WEIGHTS");
  const char* genv = getenv_str("KOKKOSDNN_DINOV3_GOLDEN_JSON");
  if (!wenv || !genv) {
    GTEST_SKIP() << "real-weight golden check needs KOKKOSDNN_DINOV3_WEIGHTS "
                 << "+ KOKKOSDNN_DINOV3_GOLDEN_JSON; skipping (tests never "
                 << "download weights). Generate the golden file with "
                 << "examples/dinov3/generate_golden.py.";
  }
  auto wfiles = split_list(wenv);
  if (wfiles.empty())
    GTEST_SKIP() << "KOKKOSDNN_DINOV3_WEIGHTS parsed to zero files; skipping.";
  for (auto& f : wfiles)
    if (!file_readable(f))
      GTEST_SKIP() << "weights file not readable, skipping: " << f;
  if (!file_readable(genv))
    GTEST_SKIP() << "golden file not readable, skipping: " << genv;

  float tol = default_tol;
  if (const char* t = getenv_str("KOKKOSDNN_DINOV3_TOL")) tol = std::stof(t);

  GoldenFile golden = load_golden_file(genv);  // throws -> FAIL (corrupt file)
  D3::Config cfg    = D3::Config::vitb16();
  if (const char* c = getenv_str("KOKKOSDNN_DINOV3_CONFIG"))
    cfg = D3::Config::from_json_file(c, D3::Config::vitb16());
  cfg.validate();
  // Golden/config consistency guards (fail loudly instead of comparing
  // shifted rows).
  ASSERT_EQ(cfg.image_size, golden.image_size);
  ASSERT_EQ(cfg.hidden_size, golden.hidden_size);
  ASSERT_EQ(cfg.num_register_tokens, golden.num_register_tokens);
  ASSERT_EQ(cfg.num_patch_tokens(), golden.num_patch_tokens);

  D3::Weights<Exec, Scalar> w;
  D3::load_safetensors(cfg, wfiles, w);  // throws -> FAIL
  D3::Model<Exec, Scalar> model(cfg, w);

  // Input: recompute the canonical synthetic pixels, or take the stored
  // preprocessed tensor for --image goldens (exact same floats the HF
  // reference saw).
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> pixels;
  if (golden.input == "synthetic") {
    pixels = make_synthetic_pixels<Exec, Scalar>(cfg.num_channels,
                                                 cfg.image_size);
    // Cross-check the recomputation against the stored reference input.
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pixels);
    ASSERT_EQ((int)(h.extent(0) * h.extent(1) * h.extent(2)),
              (int)golden.pixels.size());
    double in_diff = 0;
    size_t k       = 0;
    for (size_t c = 0; c < h.extent(0); ++c)
      for (size_t y = 0; y < h.extent(1); ++y)
        for (size_t x = 0; x < h.extent(2); ++x, ++k)
          in_diff = std::max(in_diff,
                             std::abs((double)h(c, y, x) - golden.pixels[k]));
    EXPECT_LE(in_diff, 1e-6)
        << "synthetic pixel recomputation diverged from the golden input";
  } else {
    pixels = Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec>(
        "pixels", (size_t)golden.pix_c, (size_t)golden.pix_h,
        (size_t)golden.pix_w);
    auto h = Kokkos::create_mirror_view(pixels);
    size_t k = 0;
    for (size_t c = 0; c < h.extent(0); ++c)
      for (size_t y = 0; y < h.extent(1); ++y)
        for (size_t x = 0; x < h.extent(2); ++x, ++k)
          h(c, y, x) = static_cast<Scalar>(golden.pixels[k]);
    Kokkos::deep_copy(pixels, h);
    std::printf("[golden] image input '%s' taken from golden pixels\n",
                golden.input.c_str());
  }

  auto out = model.forward(pixels);
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  const size_t T = host.extent(0), D = host.extent(1);
  const size_t R = (size_t)cfg.num_register_tokens;
  const size_t Np = (size_t)cfg.num_patch_tokens();
  ASSERT_EQ(T, 1 + R + Np);
  ASSERT_EQ(D, (size_t)cfg.hidden_size);

  std::vector<float> cls(D);
  for (size_t d = 0; d < D; ++d) cls[d] = static_cast<float>(host(0, d));
  double cls_diff = max_abs_diff(cls, golden.cls);

  double rows_diff = 0;
  for (int r = 0; r < 4; ++r) {
    std::vector<float> row(D);
    for (size_t d = 0; d < D; ++d)
      row[d] = static_cast<float>(host(1 + R + (size_t)r, d));
    rows_diff = std::max(rows_diff,
                         max_abs_diff(row, golden.patch_first4[(size_t)r]));
  }

  // Patch mean in double precision (independent of any helper).
  std::vector<double> mean(D, 0.0);
  for (size_t p = 0; p < Np; ++p)
    for (size_t d = 0; d < D; ++d)
      mean[d] += static_cast<double>(host(1 + R + p, d));
  for (size_t d = 0; d < D; ++d) mean[d] /= (double)Np;
  double mean_diff = 0;
  for (size_t d = 0; d < D; ++d)
    mean_diff = std::max(mean_diff, std::abs(mean[d] - golden.patch_mean[d]));

  std::printf("[golden] model=%s input=%s max|d| cls=%.6g patch_first4=%.6g "
              "patch_mean=%.6g tol=%g\n",
              golden.model.c_str(), golden.input.c_str(), cls_diff, rows_diff,
              mean_diff, tol);
  EXPECT_LE(cls_diff, (double)tol) << "CLS max-abs-diff over [768]";
  EXPECT_LE(rows_diff, (double)tol) << "patch-first4 max-abs-diff over 4x[768]";
  EXPECT_LE(mean_diff, (double)tol) << "patch-mean max-abs-diff over [768]";
}

}  // namespace

// Parser self-test (always runs): guards schema drift between
// generate_golden.py and parse_golden without needing weights.
TEST(Dinov3Golden, GoldenParserFixture) {
  const std::string js = R"({
    "model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dtype": "float32", "image_size": 224, "hidden_size": 4,
    "num_register_tokens": 1, "num_patch_tokens": 2,
    "input": "synthetic \"test\"\n",
    "synthetic_formula": "caf\u00e9",
    "pixels": {"channels": 1, "height": 2, "width": 2,
               "values": [0.0, 0.5, 1.0, -2.0]},
    "cls": [1.0, 2.0, 3.0, 4.0],
    "patch_first4": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    "patch_mean": [0.25, 0.25, 0.25, 0.25]
  })";
  GoldenFile g = parse_golden(js);
  EXPECT_EQ(g.model, "facebook/dinov3-vitb16-pretrain-lvd1689m");
  EXPECT_EQ(g.dtype, "float32");
  EXPECT_EQ(g.image_size, 224);
  EXPECT_EQ(g.hidden_size, 4);
  EXPECT_EQ(g.num_register_tokens, 1);
  EXPECT_EQ(g.num_patch_tokens, 2);
  EXPECT_EQ(g.input, "synthetic \"test\"\n");
  EXPECT_EQ(g.synthetic_formula, "caf\xc3\xa9");  // \u00e9 -> UTF-8
  ASSERT_EQ(g.pixels.size(), 4u);
  EXPECT_DOUBLE_EQ(g.pixels[1], 0.5);
  ASSERT_EQ(g.cls.size(), 4u);
  EXPECT_DOUBLE_EQ(g.cls[3], 4.0);
  ASSERT_EQ(g.patch_first4.size(), 4u);
  EXPECT_DOUBLE_EQ(g.patch_first4[2][2], 1.0);
  ASSERT_EQ(g.patch_mean.size(), 4u);
  EXPECT_DOUBLE_EQ(g.patch_mean[0], 0.25);
  EXPECT_THROW(parse_golden(R"({"image_size": 1})"), std::runtime_error);
  EXPECT_THROW(parse_golden("{not json"), std::runtime_error);
}

// Harness dry-run (always runs): the comparison math (max-abs-diff over
// CLS-sized vectors, patch-mean reduction) is self-consistent on a tiny
// random model, where the golden entry is derived via independent
// long-double reference math.
TEST(Dinov3Golden, HarnessDryRunTiny) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = D3::DefaultScalar;
  D3::Config cfg = D3::Config::tiny_test();
  D3::Weights<Exec, Scalar> w;
  D3::random_init(cfg, w, 1234);
  D3::Model<Exec, Scalar> model(cfg, w);
  const int C = cfg.num_channels, S = cfg.image_size;
  auto pixels = make_synthetic_pixels<Exec, Scalar>(C, S);
  auto out    = model.forward(pixels);
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  const size_t T = host.extent(0), D = host.extent(1);
  const size_t R = (size_t)cfg.num_register_tokens;
  const size_t Np = (size_t)cfg.num_patch_tokens();
  ASSERT_EQ(T, 1 + R + Np);

  // Independent long-double reference: CLS L2 norm + patch mean.
  long double n2 = 0;
  for (size_t d = 0; d < D; ++d) {
    long double v = (long double)host(0, d);
    n2 += v * v;
  }
  long double ref_norm = sqrtl(n2);
  std::vector<float> cls(D);
  for (size_t d = 0; d < D; ++d) cls[d] = static_cast<float>(host(0, d));
  long double n2f = 0;
  for (float v : cls) n2f += (long double)v * (long double)v;
  EXPECT_NEAR((double)sqrtl(n2f), (double)ref_norm, 1e-9);

  std::vector<long double> ref_mean(D, 0.0L);
  for (size_t p = 0; p < Np; ++p)
    for (size_t d = 0; d < D; ++d)
      ref_mean[d] += (long double)host(1 + R + p, d);
  for (size_t d = 0; d < D; ++d) ref_mean[d] /= (long double)Np;
  std::vector<double> mean_d(D);
  for (size_t d = 0; d < D; ++d) mean_d[d] = (double)ref_mean[d];
  std::vector<float> mean_f(D);
  for (size_t d = 0; d < D; ++d) {
    double acc = 0;
    for (size_t p = 0; p < Np; ++p) acc += (double)host(1 + R + p, d);
    mean_f[d] = static_cast<float>(acc / (double)Np);
  }
  double md = max_abs_diff(mean_f, mean_d);
  std::printf("[dry-run] tiny T=%zu D=%zu CLS_L2=%.6Lf mean_maxdiff=%.3g\n", T,
              D, ref_norm, md);
  EXPECT_LT(md, 1e-6);
  // Self-comparison of the CLS vector is exactly zero.
  std::vector<double> cls_d(D);
  for (size_t d = 0; d < D; ++d) cls_d[d] = (double)cls[d];
  EXPECT_EQ(max_abs_diff(cls, cls_d), 0.0);
}

// Real-weight golden check (gated; SKIP without files, never download).
TEST(Dinov3Golden, RealWeightsFp32) { check_real_weights_fp32(1e-3f); }
