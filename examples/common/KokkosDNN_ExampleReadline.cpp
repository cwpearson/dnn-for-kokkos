// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#include "KokkosDNN_ExampleReadline.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#ifdef KOKKOSDNN_EXAMPLE_HAVE_READLINE
#include <readline/history.h>
#include <readline/readline.h>
#else
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
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

// ESC-cancel: ESC then Enter (CR and LF spellings) aborts the current
// PromptLine read. Bound once below. Sets rl_done so ::readline returns at
// once; PromptLine reports the cancel through esc_pressed and drops the
// partial line instead of adding it to history. A two-key sequence only, so
// arrows (ESC [ A ...) and Meta combos are unaffected.
bool g_esc_hit = false;

int EscAbort(int, int) {
  g_esc_hit = true;
  rl_done = 1;
  return 0;
}

void install_once() {
  (void)::using_history();
  (void)::rl_bind_keyseq("\x1b\r", EscAbort);
  (void)::rl_bind_keyseq("\x1b\n", EscAbort);
}

std::once_flag g_init_once;
#else
// In-memory-only prompt history for the fallback editor (never persisted).
std::vector<std::string> g_hist;

// termios raw-mode guard: ICANON|ECHO off (byte-at-a-time reading, no kernel
// echo), ISIG left on so Ctrl-C/Ctrl-Z keep their usual job-control meaning.
class TermRaw {
 public:
  explicit TermRaw(int fd) : fd_(fd), engaged_(false) {
    if (::tcgetattr(fd_, &saved_) != 0) return;
    struct termios raw = saved_;
    raw.c_lflag &= static_cast<unsigned>(~(ICANON | ECHO));
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (::tcsetattr(fd_, TCSANOW, &raw) != 0) return;
    engaged_ = true;
  }
  ~TermRaw() {
    if (engaged_) (void)::tcsetattr(fd_, TCSANOW, &saved_);
  }
  TermRaw(const TermRaw&) = delete;
  TermRaw& operator=(const TermRaw&) = delete;
  bool engaged() const { return engaged_; }

 private:
  int fd_;
  struct termios saved_{};
  bool engaged_;
};

// True when fd has input waiting within ms (ESC-sequence peeking).
bool input_waiting(int fd, int ms) {
  fd_set set;
  FD_ZERO(&set);
  FD_SET(fd, &set);
  struct timeval tv;
  tv.tv_sec = ms / 1000;
  tv.tv_usec = static_cast<suseconds_t>((ms % 1000) * 1000);
  const int rc = ::select(fd + 1, &set, nullptr, nullptr, &tv);
  return rc > 0;
}

bool read_byte(int fd, unsigned char& c) {
  ssize_t n = 0;
  do {
    n = ::read(fd, &c, 1);
  } while (n < 0 && errno == EINTR);
  return n == 1;
}

// Minimal TTY line editor: printable input, Backspace, Left/Right,
// Home/End, Delete, UP/DN through the in-memory history, Enter to accept,
// Ctrl-D on an empty line for EOF, and ESC to cancel (reported through
// esc_pressed, partial line dropped). UTF-8 passes through byte-wise; the
// readline build handles multibyte editing properly.
bool prompt_fallback_tty(const char* prompt, std::string& out,
                         const std::string* prefill, bool add_to_history,
                         bool* esc_pressed) {
  TermRaw raw(STDIN_FILENO);
  if (!raw.engaged()) {
    // Terminal setup failed: plain getline, no editing or ESC handling.
    if (!std::getline(std::cin, out)) return false;
    if (!out.empty() && out.back() == '\r') out.pop_back();
    if (add_to_history && !out.empty()) g_hist.push_back(out);
    return true;
  }
  std::string buf = (prefill != nullptr) ? *prefill : std::string{};
  size_t cursor = buf.size();
  std::string draft = buf;
  size_t hist = g_hist.size();  // one-past-newest: editing the draft
  std::fputs(prompt, stdout);
  std::fwrite(buf.data(), 1, buf.size(), stdout);
  std::fflush(stdout);
  const auto redraw = [&] {
    std::string s("\r");
    s += prompt;
    s += buf;
    s += "\x1b[K";  // clear to end of line
    if (buf.size() > cursor) {
      s += "\x1b[";
      s += std::to_string(buf.size() - cursor);
      s += "D";
    }
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
  };
  const auto cancel = [&] {
    out.clear();
    if (esc_pressed != nullptr) *esc_pressed = true;
    std::fputs("\n", stdout);
    std::fflush(stdout);
    return true;
  };
  while (true) {
    unsigned char c = 0;
    if (!read_byte(STDIN_FILENO, c)) {
      std::fputs("\n", stdout);
      std::fflush(stdout);
      return false;  // EOF/error
    }
    if (c == '\r' || c == '\n') {
      std::fputs("\n", stdout);
      std::fflush(stdout);
      if (add_to_history && !buf.empty()) g_hist.push_back(buf);
      out = buf;
      return true;
    }
    if (c == 0x04) {  // Ctrl-D: EOF on an empty line, else delete at cursor
      if (buf.empty()) {
        std::fputs("\n", stdout);
        std::fflush(stdout);
        return false;
      }
      if (cursor < buf.size()) {
        buf.erase(cursor, 1);
        redraw();
      }
      continue;
    }
    if (c == 0x7f || c == 0x08) {  // Backspace/DEL
      if (cursor > 0) {
        buf.erase(cursor - 1, 1);
        --cursor;
        redraw();
      }
      continue;
    }
    if (c == 0x1b) {  // ESC: arrows/Home/End/Delete, else cancel the line
      if (!input_waiting(STDIN_FILENO, 50)) return cancel();  // lone ESC
      unsigned char e1 = 0;
      if (!read_byte(STDIN_FILENO, e1)) {
        std::fputs("\n", stdout);
        std::fflush(stdout);
        return false;
      }
      if (e1 != '[') return cancel();  // ESC + key (e.g. ESC then Enter)
      if (!input_waiting(STDIN_FILENO, 50)) continue;  // stray ESC [, ignore
      unsigned char e2 = 0;
      if (!read_byte(STDIN_FILENO, e2)) {
        std::fputs("\n", stdout);
        std::fflush(stdout);
        return false;
      }
      switch (e2) {
        case 'A':  // UP: older history
          if (!g_hist.empty() && hist > 0) {
            if (hist == g_hist.size()) draft = buf;
            --hist;
            buf = g_hist[hist];
            cursor = buf.size();
            redraw();
          }
          break;
        case 'B':  // DOWN: newer history, then back to the draft
          if (hist < g_hist.size()) {
            ++hist;
            buf = (hist == g_hist.size()) ? draft : g_hist[hist];
            cursor = buf.size();
            redraw();
          }
          break;
        case 'C':  // RIGHT
          if (cursor < buf.size()) {
            ++cursor;
            redraw();
          }
          break;
        case 'D':  // LEFT
          if (cursor > 0) {
            --cursor;
            redraw();
          }
          break;
        case 'H':  // Home
          cursor = 0;
          redraw();
          break;
        case 'F':  // End
          cursor = buf.size();
          redraw();
          break;
        case '3':  // Delete: expect '~'
          if (input_waiting(STDIN_FILENO, 50)) {
            unsigned char e3 = 0;
            if (read_byte(STDIN_FILENO, e3) && e3 == '~' &&
                cursor < buf.size()) {
              buf.erase(cursor, 1);
              redraw();
            }
          }
          break;
        default:  // unknown sequence: ignore
          break;
      }
      continue;
    }
    if (c >= 0x20) {  // printable ASCII + UTF-8 bytes
      buf.insert(cursor, 1, static_cast<char>(c));
      ++cursor;
      redraw();
      continue;
    }
    // Any other control byte: ignore (Ctrl-C/Ctrl-Z arrive via ISIG).
  }
}
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
                const std::string* prefill, bool add_to_history,
                bool* esc_pressed) {
  if (::isatty(STDIN_FILENO) != 0) {
#ifdef KOKKOSDNN_EXAMPLE_HAVE_READLINE
    std::call_once(g_init_once, install_once);
    g_prefill = (prefill != nullptr) ? *prefill : std::string{};
    rl_startup_hook = g_prefill.empty() ? nullptr : &PrefillHook;
    g_esc_hit = false;
    char* line = ::readline(prompt);
    rl_startup_hook = nullptr;
    g_prefill.clear();
    rl_done = 0;
    if (line == nullptr) {
      return false;  // EOF (Ctrl-D)
    }
    const bool esc = g_esc_hit;
    g_esc_hit = false;
    if (esc) {
      std::free(line);  // partial line is dropped, never added to history
      std::fputs("\n", stdout);
      std::fflush(stdout);
      out.clear();
      if (esc_pressed != nullptr) *esc_pressed = true;
      return true;
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
#else
    return prompt_fallback_tty(prompt, out, prefill, add_to_history,
                               esc_pressed);
#endif
  }
  // Non-TTY (piped/redirected): plain getline, no editing, history, or ESC.
  if (esc_pressed != nullptr) *esc_pressed = false;
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
