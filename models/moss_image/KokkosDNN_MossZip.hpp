// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_MOSS_ZIP_HPP
#define KOKKOSDNN_MOSS_ZIP_HPP

// Thin compatibility wrapper over the central ZIP reader
// (src/KokkosDNN_Zip.hpp, next to KokkosDNN_GGUF.hpp). New code should
// include KokkosDNN_Zip.hpp directly and use ::KokkosDNN::Zip; this header
// preserves the historical KokkosDNN::Moss::Zip names (FileReader/ZipArchive/ZipEntry,
// ld16/ld32/ld64) and the "moss" error prefix so existing diagnostics are unchanged.

#include "KokkosDNN_Zip.hpp"

namespace KokkosDNN {
namespace Moss {
namespace Zip {

using ::KokkosDNN::Zip::ld16;
using ::KokkosDNN::Zip::ld32;
using ::KokkosDNN::Zip::ld64;

// mmap window (historical name; central Zip::MappedFile with "moss" errors).
struct FileReader : ::KokkosDNN::Zip::MappedFile {
  explicit FileReader(const std::string& p)
      : ::KokkosDNN::Zip::MappedFile(p, "moss") {}
};

// Stored-entry central-directory record (historical name; central Entry).
struct ZipEntry : ::KokkosDNN::Zip::Entry {
  ZipEntry() = default;
  ZipEntry(const ::KokkosDNN::Zip::Entry& e)
      : ::KokkosDNN::Zip::Entry(e) {}
};

// Minimal ZIP central-directory reader (stored entries; historical name).
struct ZipArchive : ::KokkosDNN::Zip::Archive {
  explicit ZipArchive(const FileReader& r)
      : ::KokkosDNN::Zip::Archive(r, "moss") {}
};

}  // namespace Zip
}  // namespace Moss
}  // namespace KokkosDNN

#endif
