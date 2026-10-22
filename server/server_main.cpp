// kokkosdnn_server: OpenAI-compatible HTTP front-end (stub).
//
// Honest stub: parses --help/--version, otherwise initializes Kokkos,
// prints the baked-in configuration, notes that the REST routes are TODO,
// and finalizes. It binds no port and constructs no model session yet;
// session construction and all sampling/chat logic belong in models/*.

#include <Kokkos_Core.hpp>

#include "KokkosDNN_LFM25Config.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

constexpr const char* kVersion = "0.1.0";
constexpr int kDefaultPort = 8000;

void print_help(const char* prog) {
  std::printf(
      "Usage: %s [--port N] [--model ID] [--help] [--version]\n"
      "\n"
      "OpenAI-compatible inference server stub for KokkosDNN (LFM2.5).\n"
      "\n"
      "Options:\n"
      "  --port N     TCP port to listen on (default: %d; stub binds none yet)\n"
      "  --model ID   model identifier to serve (default: lfm2_5)\n"
      "  --help       print this message and exit\n"
      "  --version    print the server version and exit\n"
      "\n"
      "Intended routes (TODO, see server/README.md):\n"
      "  GET  /v1/models\n"
      "  POST /v1/chat/completions\n",
      prog, kDefaultPort);
}

}  // namespace

int main(int argc, char* argv[]) {
  int port = kDefaultPort;
  std::string model = "lfm2_5";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0) {
      print_help(argv[0]);
      return 0;
    }
    if (std::strcmp(argv[i], "--version") == 0) {
      std::printf("kokkosdnn_server %s\n", kVersion);
      return 0;
    }
    if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      port = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model = argv[++i];
    } else {
      std::printf("Unknown argument: %s\n", argv[i]);
      print_help(argv[0]);
      return 1;
    }
  }

  Kokkos::initialize(argc, argv);
  {
    // Reference the linked model component so the wiring is real:
    // config-header constant from KokkosDNN::lfm25.
    std::printf("kokkosdnn_server %s (stub)\n", kVersion);
    std::printf("  model: %s (lfm2_5 max_seq_len=%d)\n", model.c_str(),
                KokkosDNN::LFM25::kMaxSupportedSeqLen);
    std::printf("  port: %d (no listener yet; routes TODO)\n", port);
    std::printf("  TODO: GET /v1/models, POST /v1/chat/completions\n");
  }
  Kokkos::finalize();
  return 0;
}
