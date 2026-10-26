// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Shared example-REPL line reader.
//
// PromptLine() reads one interactive line: GNU readline (line editing,
// history, optional prefill text for edit-and-resubmit) when this component
// was built against it (see Available()), else a plain std::getline fallback
// so examples build and run without new mandatory dependencies.
#pragma once

#include <string>

namespace KokkosDNN::Example::Readline {

// True when built against GNU readline (editing + history + prefill work).
bool Available();

// True when stdin is a TTY.
bool StdinIsTty();

// Print `prompt`, read one line into `out`. When `prefill` is non-null and
// non-empty (readline build, TTY stdin only) its contents are inserted into
// the editing buffer so the caller can offer the previous input for editing.
// When `add_to_history` is set, non-empty lines are added to the in-memory
// history. Returns false on EOF/error. A trailing CR is stripped so
// CRLF consoles behave.
bool PromptLine(const char* prompt, std::string& out,
                const std::string* prefill = nullptr,
                bool add_to_history = true);

}  // namespace KokkosDNN::Example::Readline
