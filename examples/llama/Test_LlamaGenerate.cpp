// End-to-end tiny Llama: random weights, prefill + greedy decode.
// Exercises both bfloat16 (default) and fp32 scalar paths.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_LlamaConfig.hpp"
#include "KokkosDNN_LlamaModel.hpp"
#include "KokkosDNN_LlamaWeights.hpp"
#include "KokkosDNN_Tokenizer.hpp"

template <class Scalar>
void check_generate() {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg = KokkosDNN::Llama::Config::tiny_test();
  KokkosDNN::Llama::Weights<Exec, Scalar> w;
  KokkosDNN::Llama::random_init(cfg, w, 1234);
  KokkosDNN::Llama::Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};
  auto out                = model.generate(prompt, 4);
  ASSERT_EQ(out.size(), prompt.size() + 4);
  for (auto id : out) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, cfg.vocab_size);
  }
  // Determinism: rebuild and compare.
  KokkosDNN::Llama::Model<Exec, Scalar> model2(cfg, w);
  auto out2 = model2.generate(prompt, 4);
  EXPECT_EQ(out, out2);
}

TEST(LlamaGenerate, Float) { check_generate<float>(); }

TEST(LlamaGenerate, BHalf) { check_generate<Kokkos::Experimental::bhalf_t>(); }

template <class Scalar>
void check_multi_stop() {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg = KokkosDNN::Llama::Config::tiny_test();
  KokkosDNN::Llama::Weights<Exec, Scalar> w;
  KokkosDNN::Llama::random_init(cfg, w, 1234);
  KokkosDNN::Llama::Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};

  // Full run: default Instruct stops are out of tiny vocab, so no early stop.
  auto full = model.generate(prompt, 4, std::vector<int>{});
  ASSERT_EQ(full.size(), prompt.size() + 4);

  // Re-run stopping on the known first generated token: must stop after
  // exactly one new token (stop token included).
  const int first = full[prompt.size()];
  KokkosDNN::Llama::Model<Exec, Scalar> model2(cfg, w);
  auto early =
      model2.generate(prompt, 4, std::vector<int>{first, first});  // dupes ok
  ASSERT_EQ(early.size(), prompt.size() + 1);
  EXPECT_EQ(early.back(), first);

  // Non-triggering stop set: full length again.
  KokkosDNN::Llama::Model<Exec, Scalar> model3(cfg, w);
  auto no_stop = model3.generate(prompt, 4, std::vector<int>{cfg.vocab_size + 7});
  EXPECT_EQ(no_stop, full);

  // Back-compat single-eos overload agrees with the vector overload.
  KokkosDNN::Llama::Model<Exec, Scalar> model4(cfg, w);
  auto via_int = model4.generate(prompt, 4, first);
  EXPECT_EQ(via_int, early);
}

TEST(LlamaGenerateMultiStop, Float) { check_multi_stop<float>(); }

TEST(LlamaGenerateMultiStop, BHalf) {
  check_multi_stop<Kokkos::Experimental::bhalf_t>();
}

// Persistent scratch: reusing one model across generate() calls (reset +
// scratch reuse, growth when a later prefill is longer) reproduces fresh
// models exactly.
template <class Scalar>
void check_scratch_reuse() {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg = KokkosDNN::Llama::Config::tiny_test();
  KokkosDNN::Llama::Weights<Exec, Scalar> w;
  KokkosDNN::Llama::random_init(cfg, w, 1234);
  KokkosDNN::Llama::Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};
  auto first              = model.generate(prompt, 4);
  auto second             = model.generate(prompt, 4);
  EXPECT_EQ(first, second);
  // Longer prefill after short runs grows scratch; must still agree with
  // a fresh model.
  KokkosDNN::Llama::Model<Exec, Scalar> fresh(cfg, w);
  std::vector<int> long_prompt = {5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_EQ(model.generate(long_prompt, 3), fresh.generate(long_prompt, 3));
}

TEST(LlamaScratchReuse, Float) { check_scratch_reuse<float>(); }

TEST(LlamaScratchReuse, BHalf) {
  check_scratch_reuse<Kokkos::Experimental::bhalf_t>();
}

// Chunked prefill: splitting a prompt into chunks matches single-shot
// (prefill_chunk_size <= 0) logits/generate exactly, while scratch stays
// O(chunk·dim) instead of O(T·dim).
template <class Scalar>
void check_chunked_prefill() {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg   = KokkosDNN::Llama::Config::tiny_test();
  cfg.max_seq_len = 32;
  KokkosDNN::Llama::Weights<Exec, Scalar> w;
  KokkosDNN::Llama::random_init(cfg, w, 1234);
  std::vector<int> prompt = {1, 2, 3, 4, 5, 6, 7, 8};

  auto forward_logits = [&](int chunk) {
    KokkosDNN::Llama::Model<Exec, Scalar> m(cfg, w);
    m.cfg.prefill_chunk_size = chunk;
    Kokkos::View<int*, Exec> toks("toks", prompt.size());
    auto ht = Kokkos::create_mirror_view(toks);
    for (size_t i = 0; i < prompt.size(); ++i) ht(i) = prompt[i];
    Kokkos::deep_copy(toks, ht);
    auto lg = m.forward(toks);
    auto hl = Kokkos::create_mirror_view(lg);
    Kokkos::deep_copy(hl, lg);
    std::vector<float> out(hl.extent(0));
    for (size_t i = 0; i < hl.extent(0); ++i)
      out[i] = static_cast<float>(hl(i));
    return std::make_pair(out, m.scratch_cap_);
  };

  auto ref = forward_logits(0);  // single-shot reference
  for (int chunk : {1, 2, 3}) {
    auto got = forward_logits(chunk);
    ASSERT_EQ(got.first.size(), ref.first.size());
    for (size_t i = 0; i < ref.first.size(); ++i)
      EXPECT_FLOAT_EQ(got.first[i], ref.first[i]);
    // Scratch never exceeds the chunk size (decode slot is 1).
    EXPECT_LE(got.second, (size_t)chunk);
  }
  EXPECT_EQ(ref.second, prompt.size());  // single-shot sizes to T

  // End-to-end generate() agrees across chunkings.
  KokkosDNN::Llama::Model<Exec, Scalar> m1(cfg, w), m2(cfg, w);
  m1.cfg.prefill_chunk_size = 0;
  m2.cfg.prefill_chunk_size = 2;
  EXPECT_EQ(m2.generate(prompt, 4), m1.generate(prompt, 4));
}

TEST(LlamaChunkedPrefill, Float) { check_chunked_prefill<float>(); }

TEST(LlamaChunkedPrefill, BHalf) {
  check_chunked_prefill<Kokkos::Experimental::bhalf_t>();
}

// ---- O(1) reset(): rewind position counter, no cache wipe ----
// forward() overwrites KV at [cpos, cpos+C) before reading prefix
// [0, cpos+C), so stale entries are never observed after reset().
// Always-on tiny-config tests (no weight downloads).
namespace {

template <class Exec, class Scalar>
Kokkos::View<int*, Exec> make_tok_view(
    KokkosDNN::Llama::Model<Exec, Scalar>& m, const std::vector<int>& ids) {
  (void)m;
  Kokkos::View<int*, Exec> toks("toks", ids.size());
  auto h = Kokkos::create_mirror_view(toks);
  for (size_t i = 0; i < ids.size(); ++i) h(i) = ids[i];
  Kokkos::deep_copy(toks, h);
  return toks;
}

template <class Exec, class Scalar>
std::vector<float> logits_vec(
    const Kokkos::View<Scalar*, Exec>& lg) {
  auto h = Kokkos::create_mirror_view(lg);
  Kokkos::deep_copy(h, lg);
  std::vector<float> out(h.extent(0));
  for (size_t i = 0; i < h.extent(0); ++i)
    out[i] = static_cast<float>(h(i));
  return out;
}

template <class Scalar>
void expect_logits_eq(const std::vector<float>& a,
                      const std::vector<float>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) EXPECT_FLOAT_EQ(a[i], b[i]);
}

// Prefill + two greedy decode steps, returning per-step logits. Drives
// forward() directly so the KV cache (not generate()'s own reset) is
// exercised.
template <class Scalar>
std::vector<std::vector<float>> run_prefill_decode(
    KokkosDNN::Llama::Model<Kokkos::DefaultExecutionSpace, Scalar>& m,
    const std::vector<int>& prompt) {
  using Exec = Kokkos::DefaultExecutionSpace;
  std::vector<std::vector<float>> all;
  auto toks = make_tok_view(m, prompt);
  auto lg   = m.forward(toks);
  all.push_back(logits_vec(lg));
  int nxt = KokkosDNN::sample_greedy<Exec, Scalar>(lg);
  // Decode two steps from single-token slots (argmax on device).
  for (int s = 0; s < 2; ++s) {
    Kokkos::View<int*, Exec> one("one", 1);
    auto h = Kokkos::create_mirror_view(one);
    h(0) = nxt;
    Kokkos::deep_copy(one, h);
    auto lg2 = m.forward(one);
    all.push_back(logits_vec(lg2));
    nxt = KokkosDNN::sample_greedy<Exec, Scalar>(lg2);
  }
  return all;
}

}  // namespace

template <class Scalar>
void check_reset_determinism() {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto cfg = Config::tiny_test();
  Weights<Exec, Scalar> w;
  random_init(cfg, w, 1234);
  Model<Exec, Scalar> m(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};

  auto first = run_prefill_decode(m, prompt);
  EXPECT_EQ(m.seq_len, prompt.size() + 2);
  m.reset();
  EXPECT_EQ(m.seq_len, 0u);
  auto second = run_prefill_decode(m, prompt);
  ASSERT_EQ(first.size(), second.size());
  for (size_t i = 0; i < first.size(); ++i)
    expect_logits_eq<Scalar>(first[i], second[i]);

  // Fresh model agrees bit-for-bit.
  Model<Exec, Scalar> fresh(cfg, w);
  auto ref = run_prefill_decode(fresh, prompt);
  ASSERT_EQ(ref.size(), second.size());
  for (size_t i = 0; i < ref.size(); ++i)
    expect_logits_eq<Scalar>(ref[i], second[i]);
}

TEST(LlamaReset, DeterminismVsFreshFloat) { check_reset_determinism<float>(); }

TEST(LlamaReset, DeterminismVsFreshBHalf) {
  check_reset_determinism<Kokkos::Experimental::bhalf_t>();
}

template <class Scalar>
void check_reset_multiple() {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto cfg = Config::tiny_test();
  Weights<Exec, Scalar> w;
  random_init(cfg, w, 1234);
  Model<Exec, Scalar> m(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};
  // Interleave different-length sequences between resets.
  std::vector<int> other = {9, 10, 11, 12, 13, 14};
  Model<Exec, Scalar> fresh(cfg, w);
  auto expect_a = run_prefill_decode(fresh, prompt);
  for (int i = 0; i < 3; ++i) {
    auto got_other = run_prefill_decode(m, other);
    Model<Exec, Scalar> f2(cfg, w);
    expect_logits_eq<Scalar>(got_other[0], logits_vec(f2.forward(
                                                 make_tok_view(f2, other))));
    m.reset();
    auto got = run_prefill_decode(m, prompt);
    ASSERT_EQ(got.size(), expect_a.size());
    for (size_t k = 0; k < got.size(); ++k)
      expect_logits_eq<Scalar>(got[k], expect_a[k]);
    m.reset();
  }
}

TEST(LlamaReset, MultipleResetsFloat) { check_reset_multiple<float>(); }

TEST(LlamaReset, MultipleResetsBHalf) {
  check_reset_multiple<Kokkos::Experimental::bhalf_t>();
}

template <class Scalar>
void check_reset_after_partial() {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto cfg = Config::tiny_test();
  Weights<Exec, Scalar> w;
  random_init(cfg, w, 1234);
  Model<Exec, Scalar> m(cfg, w);
  // Partial fill then reset: stale prefix must not leak into the rerun.
  auto part = make_tok_view(m, std::vector<int>{7, 8});
  (void)logits_vec(m.forward(part));
  EXPECT_EQ(m.seq_len, 2u);
  m.reset();
  EXPECT_EQ(m.seq_len, 0u);
  std::vector<int> prompt = {1, 2, 3, 4};
  auto got = run_prefill_decode(m, prompt);
  Model<Exec, Scalar> fresh(cfg, w);
  auto ref = run_prefill_decode(fresh, prompt);
  ASSERT_EQ(got.size(), ref.size());
  for (size_t i = 0; i < got.size(); ++i)
    expect_logits_eq<Scalar>(got[i], ref[i]);
}

TEST(LlamaReset, AfterPartialFillFloat) {
  check_reset_after_partial<float>();
}

TEST(LlamaReset, AfterPartialFillBHalf) {
  check_reset_after_partial<Kokkos::Experimental::bhalf_t>();
}

template <class Scalar>
void check_reset_chunked() {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto cfg = Config::tiny_test();
  Weights<Exec, Scalar> w;
  random_init(cfg, w, 1234);
  std::vector<int> prompt = {1, 2, 3, 4, 5, 6, 7, 8};

  Model<Exec, Scalar> m(cfg, w);
  m.cfg.prefill_chunk_size = 2;
  auto toks = make_tok_view(m, prompt);
  auto first = logits_vec(m.forward(toks));
  m.reset();
  auto second = logits_vec(m.forward(make_tok_view(m, prompt)));
  expect_logits_eq<Scalar>(first, second);

  // Chunked-after-reset matches single-shot fresh reference.
  Model<Exec, Scalar> ref(cfg, w);
  ref.cfg.prefill_chunk_size = 0;
  auto expect = logits_vec(ref.forward(make_tok_view(ref, prompt)));
  expect_logits_eq<Scalar>(second, expect);
}

TEST(LlamaReset, ChunkedPathFloat) { check_reset_chunked<float>(); }

TEST(LlamaReset, ChunkedPathBHalf) {
  check_reset_chunked<Kokkos::Experimental::bhalf_t>();
}

// ---- Instruct chat template + BOS wiring + stop-strings (README TODO) ----
// Always-on unit tests over golden strings and a tiny inline tokenizer
// fixture (no weight downloads). Deterministic: pure string/id logic,
// no sampling.
namespace {

std::string chat_fixture_json() {
  return R"({
  "model": {"type": "BPE", "vocab": {
    "a": 0, "b": 1, "H": 2, "i": 3, "\u0120": 4, "S": 5, "y": 6, "s": 7,
    "e": 8, "m": 9, "u": 10, "r": 11, "t": 12, "o": 13, "n": 14, "A": 15,
    "\u010a": 16
  }, "merges": []},
  "added_tokens": [
    {"id": 100, "content": "<|begin_of_text|>", "special": true},
    {"id": 101, "content": "<|end_of_text|>", "special": true},
    {"id": 102, "content": "<|start_header_id|>", "special": true},
    {"id": 103, "content": "<|end_header_id|>", "special": true},
    {"id": 104, "content": "<|eot_id|>", "special": true},
    {"id": 105, "content": "<|eom_id|>", "special": true}
  ]
})";
}

std::string write_chat_fixture() {
  const std::string p = "/tmp/kokkosdnn_chat_fixture.json";
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("cannot write " + p);
  f << chat_fixture_json();
  f.close();
  return p;
}

}  // namespace

TEST(LlamaChatTemplate, GoldenStrings) {
  using KokkosDNN::Llama::ChatMessage;
  using KokkosDNN::Llama::make_chat_messages;
  using KokkosDNN::Llama::render_chat_template;

  // User-only single turn + generation prompt.
  EXPECT_EQ(render_chat_template({{"user", "Hi"}}, true),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");

  // System + user via helper.
  auto msgs = make_chat_messages("Hello", "Be brief.", "user");
  ASSERT_EQ(msgs.size(), 2u);
  EXPECT_EQ(msgs[0].role, "system");
  EXPECT_EQ(render_chat_template(msgs, true),
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Be brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");

  // Multi-turn without generation prompt ends with <|eot_id|>.
  std::vector<ChatMessage> multi = {{"user", "Hi"},
                                    {"assistant", "Hello!"},
                                    {"user", "Bye"}};
  EXPECT_EQ(render_chat_template(multi, false),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "Hello!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "Bye<|eot_id|>");

  // Empty role defaults to user; empty message list renders BOS only
  // (+ optional generation prompt).
  EXPECT_EQ(make_chat_messages("Hi", "", "").front().role, "user");
  EXPECT_EQ(render_chat_template({}, true),
            "<|begin_of_text|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n");
  EXPECT_EQ(render_chat_template({}, false), "<|begin_of_text|>");
}

TEST(LlamaChatBos, ExactlyOneBos) {
  using KokkosDNN::Llama::encode_chat;
  using KokkosDNN::Llama::encode_rendered_chat;
  using KokkosDNN::Llama::ensure_bos;
  using KokkosDNN::Llama::render_chat_template;

  // Pure id-level helper: collapse runs / strip / empty edge.
  EXPECT_EQ(ensure_bos({100, 100, 5}, 100, true), (std::vector<int>{100, 5}));
  EXPECT_EQ(ensure_bos({5, 100}, 100, true), (std::vector<int>{100, 5, 100}));
  EXPECT_EQ(ensure_bos({100, 100}, 100, true), (std::vector<int>{100}));
  EXPECT_EQ(ensure_bos({}, 100, true), (std::vector<int>{100}));
  EXPECT_EQ(ensure_bos({100, 100, 5}, 100, false), (std::vector<int>{5}));
  EXPECT_EQ(ensure_bos({5}, 100, false), (std::vector<int>{5}));

  // Through the fixture tokenizer (BOS id 100): chat render already
  // starts with the BOS literal, so naive encode(add_bos=true) would
  // double it; encode_chat must yield exactly one.
  KokkosDNN::Llama::Tokenizer tok(write_chat_fixture());
  EXPECT_EQ(tok.bos(), 100);
  std::vector<KokkosDNN::Llama::ChatMessage> msgs = {{"user", "Hi"}};
  auto rendered = render_chat_template(msgs, true);
  ASSERT_TRUE(rendered.rfind("<|begin_of_text|>", 0) == 0);

  auto naive = tok.encode(rendered, true);  // double-BOS guard check
  ASSERT_GE(naive.size(), 2u);
  EXPECT_EQ(naive[0], 100);
  EXPECT_EQ(naive[1], 100);  // proves the double-count exists without fix

  auto fixed = encode_chat(tok, msgs, true, true);
  ASSERT_FALSE(fixed.empty());
  EXPECT_EQ(fixed[0], 100);
  size_t lead = 0;
  while (lead < fixed.size() && fixed[lead] == 100) ++lead;
  EXPECT_EQ(lead, 1u);  // exactly one BOS
  // Same body as encode(add_bos=false) plus one BOS.
  auto no_bos = encode_chat(tok, msgs, true, false);
  EXPECT_EQ(fixed.size(), no_bos.size() + 1u);
  EXPECT_EQ(std::vector<int>(fixed.begin() + 1, fixed.end()), no_bos);

  // Raw text starting with the BOS literal is also collapsed to one.
  auto dup = encode_rendered_chat(tok, "<|begin_of_text|>Hi", true);
  EXPECT_EQ(dup.front(), 100);
  size_t d = 0;
  while (d < dup.size() && dup[d] == 100) ++d;
  EXPECT_EQ(d, 1u);
  // add_bos=false strips the literal's BOS entirely.
  auto stripped = tok.encode("<|begin_of_text|>Hi", false);
  EXPECT_EQ(ensure_bos(stripped, tok.bos(), false)[0], 2);  // 'H'
}

TEST(LlamaStopStrings, Truncation) {
  using KokkosDNN::Llama::find_stop_string;
  using KokkosDNN::Llama::truncate_at_stop_strings;

  EXPECT_EQ(truncate_at_stop_strings("hello world", {}), "hello world");
  EXPECT_EQ(truncate_at_stop_strings("hello world", {""}), "hello world");
  EXPECT_EQ(truncate_at_stop_strings("hello world", {"zzz"}), "hello world");
  // Earliest match wins across the list.
  EXPECT_EQ(truncate_at_stop_strings("aaSTOPbbENDcc", {"END", "STOP"}),
            "aa");
  EXPECT_EQ(truncate_at_stop_strings("foo<|eot_id|>bar", {"<|eot_id|>"}),
            "foo");
  // Match at position 0 truncates to empty; matched string reported.
  EXPECT_EQ(truncate_at_stop_strings("STOPnow", {"STOP"}), "");
  std::string hit;
  EXPECT_EQ(find_stop_string("abCDcd", {"cd", "CD"}, &hit), 2u);
  EXPECT_EQ(hit, "CD");
  EXPECT_EQ(find_stop_string("nothing", {"x"}, &hit), std::string::npos);
  EXPECT_TRUE(hit.empty());
  // Decoded-text scenario: generation suffix truncated, prompt untouched.
  // (Driver decodes gen ids only; here the pure-string equivalent.)
  std::string gen = "Paris is great. User: hi";
  EXPECT_EQ(truncate_at_stop_strings(gen, {"User:"}), "Paris is great. ");
}

// ---- config.json parsing (README TODO "Model config") ----
// Always-on unit tests over inline JSON strings + temp files (no downloads).
// A real config.json path is exercised when KOKKOSDNN_LLAMA_CONFIG_JSON is
// set (gated, SKIP otherwise).
namespace {

using KokkosDNN::Llama::Config;

const char* kTinyJson = R"({
  "hidden_size": 64,
  "num_hidden_layers": 2,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "intermediate_size": 128,
  "vocab_size": 256,
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "rope_scaling": null,
  "max_position_embeddings": 32,
  "tie_word_embeddings": false,
  "bos_token_id": 128000,
  "eos_token_id": 128001
})";

void write_tmp_json(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << text;
}

}  // namespace

TEST(LlamaConfig, DefaultsPreserved) {
  auto full = Config::llama_3_3_8b();
  EXPECT_EQ(full.dim, 4096);
  EXPECT_EQ(full.n_layers, 32);
  EXPECT_EQ(full.n_heads, 32);
  EXPECT_EQ(full.n_kv_heads, 8);
  EXPECT_EQ(full.vocab_size, 128256);
  EXPECT_EQ(full.ffn_dim, 14336);
  EXPECT_EQ(full.head_dim, 128);
  EXPECT_FLOAT_EQ(full.rms_norm_eps, 1e-5f);
  EXPECT_FLOAT_EQ(full.rope_theta, 500000.0f);
  EXPECT_EQ(full.max_seq_len, 8192);
  EXPECT_FALSE(full.tie_word_embeddings);
  EXPECT_TRUE(full.rope_scaling_type.empty());
  EXPECT_NO_THROW(full.validate());

  auto tiny = Config::tiny_test();
  EXPECT_EQ(tiny.dim, 64);
  EXPECT_EQ(tiny.n_layers, 2);
  EXPECT_EQ(tiny.head_dim, 8);
  EXPECT_EQ(tiny.max_seq_len, 32);
  EXPECT_NO_THROW(tiny.validate());
}

TEST(LlamaConfig, OverrideFromJson) {
  Config got = Config::from_json_string(kTinyJson);
  EXPECT_EQ(got.dim, 64);
  EXPECT_EQ(got.n_layers, 2);
  EXPECT_EQ(got.n_heads, 8);
  EXPECT_EQ(got.n_kv_heads, 4);
  EXPECT_EQ(got.vocab_size, 256);
  EXPECT_EQ(got.ffn_dim, 128);
  EXPECT_FLOAT_EQ(got.rms_norm_eps, 1e-5f);
  EXPECT_FLOAT_EQ(got.rope_theta, 500000.0f);
  EXPECT_EQ(got.max_seq_len, 32);
  EXPECT_FALSE(got.tie_word_embeddings);
  EXPECT_EQ(got.bos_token, 128000);
  EXPECT_EQ(got.eos_token, 128001);
  // Derived quantities recomputed, not trusted from the file.
  EXPECT_EQ(got.head_dim, 8);  // 64 / 8
  EXPECT_FLOAT_EQ(got.attn_scale, 1.0f / std::sqrt(8.0f));
  EXPECT_NO_THROW(got.validate());

  // Non-default overrides flow through (incl. theta/eps/tie/max aliases).
  const std::string js = R"({
    "hidden_size": 128, "num_hidden_layers": 3,
    "num_attention_heads": 8, "num_key_value_heads": 2,
    "intermediate_size": 256, "vocab_size": 512,
    "rms_norm_eps": 1e-06, "rope_theta": 10000.0,
    "max_position_embeddings": 256, "tie_word_embeddings": true,
    "bos_token_id": 1, "eos_token_id": [2, 3]
  })";
  Config o = Config::from_json_string(js);
  EXPECT_EQ(o.dim, 128);
  EXPECT_EQ(o.head_dim, 16);  // 128 / 8 recomputed
  EXPECT_FLOAT_EQ(o.attn_scale, 0.25f);
  EXPECT_FLOAT_EQ(o.rms_norm_eps, 1e-6f);
  EXPECT_FLOAT_EQ(o.rope_theta, 10000.0f);
  EXPECT_EQ(o.max_seq_len, 256);
  EXPECT_TRUE(o.tie_word_embeddings);
  EXPECT_EQ(o.bos_token, 1);
  EXPECT_EQ(o.eos_token, 2);  // array form: first id wins
  EXPECT_NO_THROW(o.validate());

  // "max_seq_len" alias wins over "max_position_embeddings" when both given.
  Config m = Config::from_json_string(
      R"({"max_position_embeddings": 512, "max_seq_len": 96})",
      Config::tiny_test());
  EXPECT_EQ(m.max_seq_len, 96);
}

TEST(LlamaConfig, MissingFieldsFallback) {
  // Empty object: base untouched.
  Config base = Config::tiny_test();
  Config got  = Config::from_json_string("{}", base);
  EXPECT_EQ(got.dim, base.dim);
  EXPECT_EQ(got.n_layers, base.n_layers);
  EXPECT_EQ(got.max_seq_len, base.max_seq_len);
  EXPECT_EQ(got.rope_theta, base.rope_theta);
  // Partial object: only present keys change.
  Config part = Config::from_json_string(R"({"rope_theta": 1234.0})", base);
  EXPECT_FLOAT_EQ(part.rope_theta, 1234.0f);
  EXPECT_EQ(part.dim, base.dim);
  // Unknown keys ignored.
  EXPECT_NO_THROW(Config::from_json_string(
      R"({"hidden_size": 64, "quantization_config": {"x": 1}})", base));
}

TEST(LlamaConfig, TieWordEmbeddingsHonored) {
  Config t = Config::from_json_string(R"({"tie_word_embeddings": true})");
  EXPECT_TRUE(t.tie_word_embeddings);
  Config f = Config::from_json_string(R"({"tie_word_embeddings": false})");
  EXPECT_FALSE(f.tie_word_embeddings);
  // Absent key keeps the base value (both directions).
  Config bt = Config::from_json_string("{}", Config::tiny_test());
  EXPECT_FALSE(bt.tie_word_embeddings);
  Config base = Config::llama_3_3_8b();
  base.tie_word_embeddings = true;
  EXPECT_TRUE(Config::from_json_string("{}", base).tie_word_embeddings);
}

TEST(LlamaConfig, RopeScalingRejectedOrNull) {
  // Absent / explicit null: fine.
  EXPECT_NO_THROW(Config::from_json_string("{}", Config::tiny_test()));
  EXPECT_NO_THROW(Config::from_json_string(R"({"rope_scaling": null})",
                                           Config::tiny_test()));
  // Llama-3-style YaRN object: parsed, then rejected with a clear error.
  EXPECT_THROW(Config::from_json_string(R"({"rope_scaling": {
      "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
      "original_max_position_embeddings": 8192, "rope_type": "llama3"}})"),
               std::runtime_error);
  // Linear variant (alternate "type" key): also rejected.
  EXPECT_THROW(
      Config::from_json_string(
          R"({"rope_scaling": {"type": "linear", "factor": 2.0}})"),
      std::runtime_error);
  // A manually-armed scaling flag fails validation too.
  Config c = Config::tiny_test();
  c.rope_scaling_type   = "llama3";
  c.rope_scaling_factor = 8.0;
  EXPECT_THROW(c.validate(), std::runtime_error);
}

TEST(LlamaConfig, MaxSeqLenValidation) {
  // Supported window is [1, 131072] (arch limit; PoC default stays 8192).
  EXPECT_NO_THROW(
      Config::from_json_string(R"({"max_position_embeddings": 131072})"));
  EXPECT_NO_THROW(Config::from_json_string(R"({"max_position_embeddings": 1})",
                                           Config::tiny_test()));
  EXPECT_THROW(Config::from_json_string(R"({"max_position_embeddings": 0})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"max_position_embeddings": -8})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(
                   R"({"max_position_embeddings": 1000000})"),
               std::exception);
  Config c = Config::tiny_test();
  c.max_seq_len = 0;
  EXPECT_THROW(c.validate(), std::exception);
  c.max_seq_len = 131072 + 1;
  EXPECT_THROW(c.validate(), std::exception);
  c.max_seq_len = 131072;  // 128k rents fine as far as validation goes
  EXPECT_NO_THROW(c.validate());
}

TEST(LlamaConfig, IncompatibleDimsRejected) {
  // hidden_size not divisible by num_attention_heads.
  EXPECT_THROW(Config::from_json_string(
                   R"({"hidden_size": 65, "num_attention_heads": 8})",
                   Config::tiny_test()),
               std::exception);
  // More kv heads than query heads.
  EXPECT_THROW(Config::from_json_string(
                   R"({"num_attention_heads": 4, "num_key_value_heads": 8})",
                   Config::tiny_test()),
               std::exception);
  // Query heads not a multiple of kv heads (GQA groups uneven).
  EXPECT_THROW(Config::from_json_string(
                   R"({"num_attention_heads": 7, "num_key_value_heads": 3})",
                   Config::tiny_test()),
               std::exception);
  // Non-positive dims.
  EXPECT_THROW(Config::from_json_string(R"({"vocab_size": 0})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"num_hidden_layers": 0})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"rope_theta": -1.0})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"rms_norm_eps": 0.0})",
                                        Config::tiny_test()),
               std::exception);
  // Malformed JSON / wrong root type.
  EXPECT_THROW(Config::from_json_string("{not json}", Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string("[1,2,3]", Config::tiny_test()),
               std::exception);
}

TEST(LlamaConfig, FromJsonFile) {
  const std::string path = "/tmp/kokkosdnn_test_llama_config.json";
  write_tmp_json(path, kTinyJson);
  Config got = Config::from_json_file(path);
  EXPECT_EQ(got.dim, 64);
  EXPECT_EQ(got.max_seq_len, 32);
  // File overlay keeps the caller's base for missing keys.
  write_tmp_json(path, R"({"rope_theta": 777.0})");
  Config over = Config::from_json_file(path, Config::tiny_test());
  EXPECT_FLOAT_EQ(over.rope_theta, 777.0f);
  EXPECT_EQ(over.dim, 64);
  EXPECT_EQ(over.max_seq_len, 32);
  std::remove(path.c_str());
  // Missing file: clear error, no crash.
  EXPECT_THROW(Config::from_json_file("/nonexistent_config_12345.json"),
               std::runtime_error);
}

// theta/eps overrides actually reach the model (RoPE inv_freq), and the
// Model constructor enforces validation before renting the KV cache.
TEST(LlamaConfig, OverridesReachModel) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg           = Config::tiny_test();
  cfg.rope_theta     = 10000.0f;
  cfg.rms_norm_eps   = 1e-6f;
  cfg.max_seq_len    = 48;
  KokkosDNN::Llama::Weights<Exec, float> w;
  KokkosDNN::Llama::random_init(cfg, w, 7);
  KokkosDNN::Llama::Model<Exec, float> m(cfg, w);
  EXPECT_EQ((int)m.inv_freq.extent(0), cfg.head_dim / 2);
  auto h = Kokkos::create_mirror_view(m.inv_freq);
  Kokkos::deep_copy(h, m.inv_freq);
  // inv_freq[1] = theta^(-2/head_dim): proves the parsed theta was used.
  EXPECT_NEAR(h(1), 1.0f / std::pow(10000.0f, 2.0f / cfg.head_dim), 1e-6f);

  // Invalid configs fail fast in the constructor (before allocation).
  auto bad       = Config::tiny_test();
  bad.max_seq_len = 0;
  KokkosDNN::Llama::Weights<Exec, float> w2;
  KokkosDNN::Llama::random_init(Config::tiny_test(), w2, 7);
  EXPECT_THROW((KokkosDNN::Llama::Model<Exec, float>(bad, w2)),
               std::exception);
}

// Real config.json, gated on env (never downloaded from tests).
TEST(LlamaConfig, RealConfigFileGated) {
  const char* p = std::getenv("KOKKOSDNN_LLAMA_CONFIG_JSON");
  if (!p || !*p)
    GTEST_SKIP() << "set KOKKOSDNN_LLAMA_CONFIG_JSON=<config.json> to check "
                 << "a real HF config (e.g. Llama-3.3-8B-Instruct); skipping.";
  std::ifstream f(p, std::ios::binary);
  if (!f) GTEST_SKIP() << "config file not readable, skipping: " << p;
  Config cfg = Config::from_json_file(p);  // throws -> FAIL (corrupt file)
  cfg.validate();
  std::printf("[config %s] dim=%d layers=%d heads=%d kv=%d ffn=%d vocab=%d "
              "eps=%g theta=%g max_seq=%d tie=%d scaling='%s'\n",
              p, cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
              cfg.ffn_dim, cfg.vocab_size, (double)cfg.rms_norm_eps,
              (double)cfg.rope_theta, cfg.max_seq_len,
              (int)cfg.tie_word_embeddings, cfg.rope_scaling_type.c_str());
  // A real Llama 3.x config must at least look like one.
  EXPECT_GT(cfg.dim, 0);
  EXPECT_GT(cfg.vocab_size, 32000);
  EXPECT_LE(cfg.max_seq_len, 131072);
}

// Tied lm_head: with tie_word_embeddings=true a checkpoint without any
// lm_head tensor loads by aliasing the embedding table; without the flag
// the same checkpoint throws.
namespace {

void write_safetensors_no_lm_head(const std::string& path) {
  const int D = 8, Hq = 8, Hkv = 4, F = 16, V = 16;
  auto f32 = [&](std::vector<float> f) {
    std::vector<char> b(f.size() * 4);
    std::memcpy(b.data(), f.data(), b.size());
    return b;
  };
  std::vector<float> ones_D(D, 1.0f), seq(V * D), wq(Hq * D), wkv(Hkv * D),
      wo(D * Hq), gate(F * D), down(D * F);
  for (size_t i = 0; i < seq.size(); ++i) seq[i] = 0.06f;
  for (size_t i = 0; i < wq.size(); ++i) wq[i] = 0.01f;
  for (size_t i = 0; i < wkv.size(); ++i) wkv[i] = 0.02f;
  for (size_t i = 0; i < wo.size(); ++i) wo[i] = 0.03f;
  for (size_t i = 0; i < gate.size(); ++i) gate[i] = 0.04f;
  for (size_t i = 0; i < down.size(); ++i) down[i] = 0.05f;
  struct E {
    std::string name, dtype, shape;
    std::vector<char> data;
  };
  std::vector<E> es = {
      {"model.embed_tokens.weight", "F32", "[16,8]", f32(seq)},
      {"model.norm.weight", "F32", "[8]", f32(ones_D)},
      // NOTE: no lm_head.* tensor (tied checkpoint layout).
      {"model.layers.0.self_attn.q_proj.weight", "F32", "[8,8]", f32(wq)},
      {"model.layers.0.self_attn.k_proj.weight", "F32", "[4,8]", f32(wkv)},
      {"model.layers.0.self_attn.v_proj.weight", "F32", "[4,8]", f32(wkv)},
      {"model.layers.0.self_attn.o_proj.weight", "F32", "[8,8]", f32(wo)},
      {"model.layers.0.mlp.gate_proj.weight", "F32", "[16,8]", f32(gate)},
      {"model.layers.0.mlp.up_proj.weight", "F32", "[16,8]", f32(gate)},
      {"model.layers.0.mlp.down_proj.weight", "F32", "[8,16]", f32(down)},
      {"model.layers.0.input_layernorm.weight", "F32", "[8]", f32(ones_D)},
      {"model.layers.0.post_attention_layernorm.weight", "F32", "[8]",
       f32(ones_D)},
  };
  std::string js = "{";
  size_t off     = 0;
  for (size_t i = 0; i < es.size(); ++i) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[%zu,"
                  "%zu]}%s",
                  es[i].name.c_str(), es[i].dtype.c_str(), es[i].shape.c_str(),
                  off, off + es[i].data.size(),
                  i + 1 < es.size() ? "," : "");
    js += buf;
    off += es[i].data.size();
  }
  js += "}";
  std::ofstream f(path, std::ios::binary);
  uint64_t n = js.size();
  unsigned char hb[8];
  for (int i = 0; i < 8; ++i) hb[i] = (unsigned char)((n >> (8 * i)) & 0xFF);
  f.write((char*)hb, 8);
  f.write(js.data(), (std::streamsize)js.size());
  for (auto& e : es)
    f.write(e.data.data(), (std::streamsize)e.data.size());
}

KokkosDNN::Llama::Config tied_test_config(bool tie) {
  KokkosDNN::Llama::Config cfg;
  cfg.dim = 8;
  cfg.n_layers = 1;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.vocab_size = 16;
  cfg.ffn_dim = 16;
  cfg.head_dim = 2;
  cfg.max_seq_len = 16;
  cfg.tie_word_embeddings = tie;
  return cfg;
}

}  // namespace

TEST(LlamaConfig, TiedLmHeadLoader) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const std::string path = "/tmp/kokkosdnn_test_tied.safetensors";
  write_safetensors_no_lm_head(path);
  // Untied: missing lm_head is an error (previous behavior preserved).
  {
    auto cfg = tied_test_config(false);
    KokkosDNN::Llama::Weights<Exec, float> w;
    EXPECT_THROW(KokkosDNN::Llama::load_safetensors(cfg, {path}, w),
                 std::runtime_error);
  }
  // Tied: loads, lm_head aliases the embedding table.
  {
    auto cfg = tied_test_config(true);
    KokkosDNN::Llama::Weights<Exec, float> w;
    EXPECT_NO_THROW(KokkosDNN::Llama::load_safetensors(cfg, {path}, w));
    EXPECT_EQ(w.lm_head.extent(0), w.embed.extent(0));
    EXPECT_EQ(w.lm_head.extent(1), w.embed.extent(1));
    EXPECT_EQ(w.lm_head.data(), w.embed.data());
    // Forward smoke through the tied weights.
    KokkosDNN::Llama::Model<Exec, float> model(cfg, w);
    Kokkos::View<int*, Exec> toks("toks", 2);
    auto ht = Kokkos::create_mirror_view(toks);
    ht(0) = 1;
    ht(1) = 2;
    Kokkos::deep_copy(toks, ht);
    EXPECT_EQ(model.forward(toks).extent(0), (size_t)cfg.vocab_size);
  }
  std::remove(path.c_str());
}

// ---- Sampling (temperature / top-k / top-p / min-p, penalty, seed) + ----
// ---- streaming decode: always-on, no weight downloads. ----
template <class Scalar>
void check_sampled_generate() {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg = KokkosDNN::Llama::Config::tiny_test();
  KokkosDNN::Llama::Weights<Exec, Scalar> w;
  KokkosDNN::Llama::random_init(cfg, w, 1234);
  std::vector<int> prompt = {1, 2, 3, 4};

  // Default options stay greedy: identical to the legacy greedy path.
  KokkosDNN::Llama::Model<Exec, Scalar> m0(cfg, w);
  auto greedy_out = m0.generate(prompt, 4);
  KokkosDNN::Llama::Model<Exec, Scalar> m1(cfg, w);
  KokkosDNN::SampleOptions gopt;  // temp=0
  KokkosDNN::HostRNG r0(0);
  auto via_opt = m1.generate(prompt, 4, std::vector<int>{}, gopt, &r0);
  EXPECT_EQ(via_opt, greedy_out);

  // Fixed seed reproduces: same seed + same prompt => identical output.
  KokkosDNN::SampleOptions sopt;
  sopt.temperature = 1.0f;
  sopt.top_k = 8;
  sopt.seed = 777;
  KokkosDNN::Llama::Model<Exec, Scalar> ma(cfg, w), mb(cfg, w);
  KokkosDNN::HostRNG ra(sopt.seed), rb(sopt.seed);
  auto oa = ma.generate(prompt, 4, std::vector<int>{}, sopt, &ra);
  auto ob = mb.generate(prompt, 4, std::vector<int>{}, sopt, &rb);
  EXPECT_EQ(oa, ob);
  ASSERT_EQ(oa.size(), prompt.size() + 4);

  // Null RNG with a fixed seed is equally deterministic.
  KokkosDNN::Llama::Model<Exec, Scalar> mc(cfg, w), md(cfg, w);
  auto oc = mc.generate(prompt, 4, std::vector<int>{}, sopt, nullptr);
  auto od = md.generate(prompt, 4, std::vector<int>{}, sopt, nullptr);
  EXPECT_EQ(oc, od);

  // Streaming callback: invoked once per new token, in order, matching the
  // returned suffix.
  KokkosDNN::Llama::Model<Exec, Scalar> ms(cfg, w);
  KokkosDNN::HostRNG rs(sopt.seed);
  std::vector<int> streamed;
  auto full =
      ms.generate(prompt, 4, std::vector<int>{}, sopt, &rs,
                  [&](int id) { streamed.push_back(id); });
  ASSERT_EQ(streamed.size(), 4u);
  EXPECT_EQ(streamed,
            std::vector<int>(full.begin() + (long)prompt.size(), full.end()));

  // Greedy streaming also fires per token (temperature=0 path).
  KokkosDNN::Llama::Model<Exec, Scalar> mg(cfg, w);
  std::vector<int> gstream;
  auto gfull = mg.generate(prompt, 4, std::vector<int>{}, gopt, nullptr,
                           [&](int id) { gstream.push_back(id); });
  EXPECT_EQ(gfull, greedy_out);
  ASSERT_EQ(gstream.size(), 4u);
  EXPECT_EQ(gstream,
            std::vector<int>(gfull.begin() + (long)prompt.size(), gfull.end()));

  // Repetition penalty flows through generate (smoke: valid ids, full len).
  KokkosDNN::SampleOptions popt;
  popt.temperature = 1.0f;
  popt.repetition_penalty = 1.5f;
  popt.seed = 11;
  KokkosDNN::Llama::Model<Exec, Scalar> mp(cfg, w);
  KokkosDNN::HostRNG rp(popt.seed);
  auto op = mp.generate(prompt, 4, std::vector<int>{}, popt, &rp);
  ASSERT_EQ(op.size(), prompt.size() + 4);
  for (auto id : op) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, cfg.vocab_size);
  }
}

TEST(LlamaSampledGenerate, Float) { check_sampled_generate<float>(); }

TEST(LlamaSampledGenerate, BHalf) {
  check_sampled_generate<Kokkos::Experimental::bhalf_t>();
}
