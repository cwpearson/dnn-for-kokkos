// SmolLM2 tokenizer regression tests (synthetic GPT-2 fixtures only, no
// model downloads). Covers: base/instruct BOS-EOS conventions, Digits
// individual-digits splitting, GPT-2 ByteLevel space handling, specials +
// set_bos/set_eos overrides, sibling tokenizer_config.json override, and
// the ChatML render/encode path.
//
// NOTE: this file is pure ASCII by design. Non-ASCII appears only as JSON
// \uXXXX escapes (fixture), so the \u-escape parser itself is covered too.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "KokkosDNN_SmolLM2Tokenizer.hpp"

namespace {

// SmolLM2/GPT-2-style fixture: Sequence[Digits, ByteLevel] pre_tokenizer;
// byte-mapped space is U+0120 ("G-dot"); merges use plain ASCII pairs;
// specials mirror SmolLM2 (<|endoftext|> = base BOS/EOS,
// <|im_start|>/<|im_end|> = Instruct BOS/EOS+stop).
std::string gpt2_fixture_json() {
  return R"({
  "model": {"type": "BPE", "vocab": {
    "a": 0, "b": 1, "o": 5, "n": 6, "d": 3,
    "1": 10, "2": 11, "12": 12,
    "ab": 20, "don": 22, "'t": 23,
    "c": 30, "e": 31, "f": 32, "g": 33, "h": 34, "i": 35, "j": 36,
    "k": 37, "l": 38, "m": 39, "p": 40, "q": 41, "r": 42, "s": 43,
    "t": 44, "u": 45, "v": 46, "w": 47, "x": 48, "y": 49, "z": 50,
    "\u010a": 60,
    "\u0120": 2
  }, "merges": ["a b", "1 2", "d o", "do n", "' t"]},
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
    {"type": "Digits", "individual_digits": true},
    {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}
  ]},
  "added_tokens": [
    {"id": 100, "content": "<|endoftext|>", "special": true},
    {"id": 101, "content": "<|im_start|>", "special": true},
    {"id": 102, "content": "<|im_end|>", "special": true}
  ]
})";
}

std::string write_tmp(const std::string& body) {
  std::string p = "/tmp/kokkosdnn_smollm2_tok_fixture_" +
                  std::to_string(::getpid()) + ".json";
  std::ofstream f(p, std::ios::binary);
  f << body;
  f.close();
  return p;
}

}  // namespace

TEST(SmolLM2Tokenizer, Gpt2AutoDetect) {
  KokkosDNN::SmolLM2::Tokenizer tok(write_tmp(gpt2_fixture_json()));
  EXPECT_EQ(tok.vocab_size(), 103);  // max id 102 (specials extend vocab)
  // Base-model convention: bos=eos=<|endoftext|>; eot=<|im_end|> (stop).
  // In particular no 128000+ Llama fallback leaks into this small vocab.
  EXPECT_EQ(tok.bos(), 100);
  EXPECT_EQ(tok.eos(), 100);
  EXPECT_EQ(tok.eot(), 102);
  EXPECT_TRUE(tok.is_special(100));
  EXPECT_TRUE(tok.is_special(101));
  EXPECT_TRUE(tok.is_special(102));
  EXPECT_LT(tok.bos(), tok.vocab_size());
  EXPECT_LT(tok.eos(), tok.vocab_size());
  EXPECT_LT(tok.eot(), tok.vocab_size());
}

TEST(SmolLM2Tokenizer, Gpt2DigitsSingle) {
  KokkosDNN::SmolLM2::Tokenizer tok(write_tmp(gpt2_fixture_json()));
  // Digits(individual_digits=true) isolates each digit, so the "1 2" merge
  // never fires even though it exists in `merges`.
  EXPECT_EQ(tok.encode("12", false), (std::vector<int>{10, 11}));
  EXPECT_EQ(tok.encode("a12", false), (std::vector<int>{0, 10, 11}));
}

TEST(SmolLM2Tokenizer, Gpt2SpaceHandling) {
  KokkosDNN::SmolLM2::Tokenizer tok(write_tmp(gpt2_fixture_json()));
  // GPT-2 regex: "a b" -> ["a", " b"]; " b" byte-maps to U+0120+"b" with
  // no applicable merge, hence per-char ids.
  EXPECT_EQ(tok.encode("a b", false), (std::vector<int>{0, 2, 1}));
  // "  a" -> [" ", " a"] -> [U+0120, U+0120+"a"] -> [2, 2, 0].
  EXPECT_EQ(tok.encode("  a", false), (std::vector<int>{2, 2, 0}));
  // Letters still merge and roundtrip exactly through the byte mapping.
  EXPECT_EQ(tok.encode("ab", false), (std::vector<int>{20}));
  EXPECT_EQ(tok.decode(tok.encode("a b", false)), "a b");
  EXPECT_EQ(tok.encode("don't", false), (std::vector<int>{22, 23}));
}

TEST(SmolLM2Tokenizer, Gpt2SpecialsAndBos) {
  KokkosDNN::SmolLM2::Tokenizer tok(write_tmp(gpt2_fixture_json()));
  EXPECT_EQ(tok.encode("a<|im_end|>b", false), (std::vector<int>{0, 102, 1}));
  EXPECT_EQ(tok.encode("a", true), (std::vector<int>{100, 0}));
  EXPECT_EQ(tok.decode({100, 0, 102, 1}), "ab");
  EXPECT_EQ(tok.decode({100, 0, 102, 1}, false), "<|endoftext|>a<|im_end|>b");
  // Explicit overrides (SmolLM2-Instruct bos=1 eos=2 without a config file).
  tok.set_bos(101);
  tok.set_eos(102);
  EXPECT_EQ(tok.bos(), 101);
  EXPECT_EQ(tok.eos(), 102);
  EXPECT_EQ(tok.encode("a", true), (std::vector<int>{101, 0}));
}

TEST(SmolLM2Tokenizer, Gpt2SiblingConfigOverride) {
  namespace fs = std::filesystem;
  fs::path dir =
      fs::temp_directory_path() /
      ("kokkosdnn_smollm2_tok_cfg_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  std::ofstream(dir / "tokenizer.json", std::ios::binary)
      << gpt2_fixture_json();
  // SmolLM2-Instruct choice: bos=<|im_start|>, eos=pad=<|im_end|>.
  std::ofstream(dir / "tokenizer_config.json", std::ios::binary)
      << R"({"bos_token": "<|im_start|>", "eos_token": "<|im_end|>"})";
  KokkosDNN::SmolLM2::Tokenizer tok((dir / "tokenizer.json").string());
  EXPECT_EQ(tok.bos(), 101);
  EXPECT_EQ(tok.eos(), 102);
  fs::remove_all(dir);
}

TEST(SmolLM2Tokenizer, ChatmlRender) {
  using KokkosDNN::SmolLM2::ChatMessage;
  using KokkosDNN::SmolLM2::render_chatml_template;
  // Default-system injection + generation prompt.
  EXPECT_EQ(render_chatml_template({{ "user", "Hi" }}),
            "<|im_start|>system\n"
            "You are a helpful AI assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "Hi<|im_end|>\n"
            "<|im_start|>assistant\n");
  // Explicit system suppresses injection; no generation prompt on request.
  EXPECT_EQ(render_chatml_template({{ "system", "S" }, { "user", "Hi" }},
                                   false),
            "<|im_start|>system\nS<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n");
  // Custom default system; empty default disables injection.
  EXPECT_EQ(render_chatml_template({{ "user", "Hi" }}, false, "Custom."),
            "<|im_start|>system\nCustom.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n");
  EXPECT_EQ(render_chatml_template({{ "user", "Hi" }}, false, ""),
            "<|im_start|>user\nHi<|im_end|>\n");
}

TEST(SmolLM2Tokenizer, ChatmlEncodeRoundtrip) {
  using KokkosDNN::SmolLM2::ChatMessage;
  KokkosDNN::SmolLM2::Tokenizer tok(write_tmp(gpt2_fixture_json()));
  // All-lowercase content stays inside the fixture's single-char vocab.
  std::vector<ChatMessage> msgs = {{"system", "don"}, {"user", "add on"}};
  std::vector<int> ids =
      KokkosDNN::SmolLM2::encode_chatml(tok, msgs, true, true);
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids[0], 100);  // exactly-one-BOS (base <|endoftext|>)
  std::string back = tok.decode(ids);
  EXPECT_NE(back.find("add on"), std::string::npos);
  EXPECT_NE(back.find("don"), std::string::npos);
  // Literal specials survive the special-token split path.
  EXPECT_EQ(tok.decode(ids, false).substr(0, 27),
            "<|endoftext|><|im_start|>sy");
}
