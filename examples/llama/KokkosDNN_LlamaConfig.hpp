#ifndef KOKKOSDNN_LLAMA_CONFIG_HPP
#define KOKKOSDNN_LLAMA_CONFIG_HPP

#include <cstddef>

namespace KokkosDNN {
namespace Llama {

// Hyperparameters for Llama 3.3 8B (Meta, Dec 2024; same arch as 3.1 8B).
// Reference: HF meta-llama/Llama-3.3-70B-Instruct config pattern scaled to
// 8B + llama3 paper. Values below match config.json of
// meta-llama/Llama-3.3-8B-Instruct.
struct Config {
  int dim            = 4096;    // hidden size
  int n_layers       = 32;
  int n_heads        = 32;      // query heads
  int n_kv_heads     = 8;       // GQA kv heads
  int vocab_size     = 128256;
  int ffn_dim        = 14336;   // SwiGLU intermediate
  int head_dim       = 128;     // dim / n_heads
  float rms_norm_eps = 1e-5f;
  float rope_theta   = 500000.0f;
  float attn_scale   = 0.0883883476f;  // 1/sqrt(128)
  int max_seq_len    = 8192;           // PoC runtime cap (arch supports 128k)
  int bos_token      = 128000;
  int eos_token      = 128001;
  // Instruct stop tokens (HF generation_config eos_token_id
  // [128001, 128008, 128009]): <|eom_id|> and <|eot_id|>.
  int eom_token      = 128008;
  int eot_token      = 128009;

  static Config llama_3_3_8b() { return Config{}; }

  // Tiny config for unit tests (no 16GB download needed).
  static Config tiny_test() {
    Config c;
    c.dim = 64; c.n_layers = 2; c.n_heads = 8; c.n_kv_heads = 4;
    c.vocab_size = 256; c.ffn_dim = 128; c.head_dim = 8;
    c.max_seq_len = 32;
    return c;
  }
};

}  // namespace Llama
}  // namespace KokkosDNN

#endif
