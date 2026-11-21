// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Out-of-line special members for MossImage::Config (declared in
// KokkosDNN_MossImage.hpp). The std::vector member copy must never inline
// into callers: GCC 15's -Warray-bounds false-positives on the inlined
// memmove once it proves skip_layers has one element, and -Werror promotes
// it to a build failure. Defaulting them here compiles the vector copy
// once, with opaque sizes, so the warning cannot fire; callers only emit a
// plain call. Semantics are exactly the implicitly-defined ones.
#include "KokkosDNN_MossImage.hpp"

namespace KokkosDNN {
namespace MossImage {

Config::Config() = default;
Config::Config(const Config&) = default;
Config& Config::operator=(const Config&) = default;
Config::Config(Config&&) noexcept = default;
Config& Config::operator=(Config&&) noexcept = default;

}  // namespace MossImage
}  // namespace KokkosDNN
