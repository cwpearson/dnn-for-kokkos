// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#include "KokkosDNN_ExampleReadline.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <unistd.h>

#ifdef KOKKOSDNN_EXAMPLE_HAVE_READLINE
#include <readline/history.h>
#include <readline/readline.h>
#endif

namespace KokkosDNN::Example::Readline {

namespace {
#ifdef KOKKOSDNN_EXAMPLE_HAVE_READLINE
// Staging for rl_startup_hook: readline calls PrefillHook() before accepting
// input, so the previous line can be offered for editing. Copied (not a
// pointer into the caller's string) so the caller's lifetime cannot dangle.
std::string g_prefill;

int PrefillHook() {
  if (!g_prefill.empty()) {
    (void)rl_insert_text(g_prefill.c_str());
  }
  return 0;
}

std::once_flag g_hist_once;
#endif
}  // namespace

bool Available() {
#ifdef KOKKOSDNN_EXAMPLE_HAVE_READLINE
  return true;
#else
  return false;
#endif
}

bool StdinIsTty() { return ::isatty(STDIN_FILENO) != 0; }

bool PromptLine(const char* prompt, std::string& out,
                const std::string* prefill, bool add_to_history) {
#ifdef KOKKOSDNN_EXAMPLE_HAVE_READLINE
  if (::isatty(STDIN_FILENO) != 0) {
    std::call_once(g_hist_once, [] { ::using_history(); });
    g_prefill = (prefill != nullptr) ? *prefill : std::string{};
    rl_startup_hook = g_prefill.empty() ? nullptr : &PrefillHook;
    char* line = ::readline(prompt);
    rl_startup_hook = nullptr;
    g_prefill.clear();
    if (line == nullptr) {
      return false;  // EOF (Ctrl-D)
    }
    out.assign(line);
    std::free(line);
    if (!out.empty() && out.back() == '\r') {
      out.pop_back();
    }
    if (add_to_history && !out.empty()) {
      (void)::add_history(out.c_str());
    }
    return true;
  }
  (void)prefill;
  (void)add_to_history;
#else
  (void)prefill;
  (void)add_to_history;
#endif
  std::fputs(prompt, stdout);
  std::fflush(stdout);
  if (!std::getline(std::cin, out)) {
    return false;
  }
  if (!out.empty() && out.back() == '\r') {
    out.pop_back();
  }
  return true;
}

}  // namespace KokkosDNN::Example::Readline
