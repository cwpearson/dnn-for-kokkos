// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Shared example-REPL line reader.
//
// PromptLine() reads one interactive line: GNU readline (line editing,
// history, optional prefill text for edit-and-resubmit, ESC-cancel) when
// this component was built against it (see Available()), else a minimal
// built-in TTY editor with the same in-memory history and ESC behavior, and
// plain std::getline off-TTY, so examples build and run without new
// mandatory dependencies.
#pragma once

#include <string>

namespace KokkosDNN::Example::Readline {

// True when built against GNU readline (editing + history + prefill work).
bool Available();

// True when stdin is a TTY.
bool StdinIsTty();

// Print `prompt`, read one line into `out`. When `prefill` is non-null and
// non-empty (TTY stdin only) its contents are inserted into the editing
// buffer so the caller can offer the previous input for editing. When
// `add_to_history` is set, non-empty accepted lines are added to the
// in-memory history (never persisted anywhere); UP/DN move through it on
// later reads (native readline support when built against it, otherwise the
// built-in fallback editor with the same history behavior). When
// `esc_pressed` is non-null, ESC cancels the line: ESC then Enter under GNU
// readline, a lone ESC in the fallback editor. A cancel clears `out`, never
// touches history, returns true with `*esc_pressed` set, and the caller
// decides what cancelling means (re-ask, back out, ...). Returns false on
// EOF/error. A trailing CR is stripped so CRLF consoles behave.
bool PromptLine(const char* prompt, std::string& out,
                const std::string* prefill = nullptr,
                bool add_to_history = true, bool* esc_pressed = nullptr);

}  // namespace KokkosDNN::Example::Readline
