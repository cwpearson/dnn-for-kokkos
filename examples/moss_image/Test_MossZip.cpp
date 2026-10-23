// Synthetic-ZIP unit test for the MOSS torch.save (.pth) central-directory
// reader (KokkosDNN_MossZip.hpp). No real checkpoint needed: the test builds
// a minimal stored-entries ZIP in memory, writes it to a temp file, and
// checks that ZipArchive reports the true compressed/uncompressed sizes.
//
// Regression cover: the reader used to take compressed size from offset +24
// (the uncompressed-size field) and uncompressed size from offset +28 (the
// file-name-length field), so for an entry like name "ab" (len 2) with a
// 23-byte payload it reported uncomp_size 2 and read_stored returned 2
// garbage bytes; every .pth load then failed its size validation.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "KokkosDNN_MossZip.hpp"

namespace {

void put16(std::string& s, uint16_t v) {
  s.push_back(static_cast<char>(v & 0xFF));
  s.push_back(static_cast<char>((v >> 8) & 0xFF));
}
void put32(std::string& s, uint32_t v) {
  for (int i = 0; i < 4; ++i)
    s.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}

// One stored entry: local header + payload. Returns the local-header offset.
uint64_t emit_local(std::string& zip, const std::string& name,
                    const std::string& payload, const std::string& extra) {
  const uint64_t off = static_cast<uint64_t>(zip.size());
  zip += "PK\x03\x04";
  put16(zip, 20);  // version needed
  put16(zip, 0);   // flags
  put16(zip, 0);   // method 0 = stored
  put16(zip, 0);   // mod time
  put16(zip, 0);   // mod date
  put32(zip, 0);   // crc (unchecked by the reader)
  put32(zip, static_cast<uint32_t>(payload.size()));
  put32(zip, static_cast<uint32_t>(payload.size()));
  put16(zip, static_cast<uint16_t>(name.size()));
  put16(zip, static_cast<uint16_t>(extra.size()));
  zip += name;
  zip += extra;
  zip += payload;
  return off;
}

void emit_central(std::string& zip, const std::string& name,
                  const std::string& payload, const std::string& extra,
                  uint64_t local_off) {
  zip += "PK\x01\x02";
  put16(zip, 63);  // version made by (arbitrary)
  put16(zip, 20);  // version needed
  put16(zip, 0);   // flags
  put16(zip, 0);   // method 0 = stored
  put16(zip, 0);   // mod time
  put16(zip, 0);   // mod date
  put32(zip, 0);   // crc (unchecked by the reader)
  put32(zip, static_cast<uint32_t>(payload.size()));
  put32(zip, static_cast<uint32_t>(payload.size()));
  put16(zip, static_cast<uint16_t>(name.size()));
  put16(zip, static_cast<uint16_t>(extra.size()));
  put16(zip, 0);  // comment length
  put16(zip, 0);  // disk number
  put16(zip, 0);  // internal attrs
  put32(zip, 0);  // external attrs
  put32(zip, static_cast<uint32_t>(local_off));
  zip += name;
  zip += extra;
}

std::string build_zip(uint64_t& ab_off, uint64_t& data_off) {
  std::string zip;
  // Regression shape from the report: 2-char name, 23-byte payload. The
  // buggy reader reported uncomp_size 2 (the name length) here.
  const std::string ab(23, 'x');
  const std::string data(100, 'y');
  const std::string extra("\xCA\xFE\x00\x00", 4);
  ab_off = emit_local(zip, "ab", ab, "");
  data_off = emit_local(zip, "data/0", data, extra);
  const uint64_t cd_off = static_cast<uint64_t>(zip.size());
  emit_central(zip, "ab", ab, "", ab_off);
  emit_central(zip, "data/0", data, extra, data_off);
  const uint64_t cd_size = static_cast<uint64_t>(zip.size()) - cd_off;
  zip += "PK\x05\x06";
  put16(zip, 0);  // disk
  put16(zip, 0);  // cd disk
  put16(zip, 2);  // entries on disk
  put16(zip, 2);  // total entries
  put32(zip, static_cast<uint32_t>(cd_size));
  put32(zip, static_cast<uint32_t>(cd_off));
  put16(zip, 0);  // comment length
  return zip;
}

std::string write_temp(const std::string& bytes) {
  const std::string path =
      std::string(::testing::TempDir()) + "moss_zip_test.zip";
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write temp zip");
  const size_t n =
      std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
  if (n != bytes.size()) throw std::runtime_error("short write of temp zip");
  return path;
}

TEST(MossZip, StoredSizesAndRoundTrip) {
  using KokkosDNN::Moss::Zip::FileReader;
  using KokkosDNN::Moss::Zip::ZipArchive;
  uint64_t ab_off = 0, data_off = 0;
  const std::string path = write_temp(build_zip(ab_off, data_off));
  const FileReader fr(path);
  const ZipArchive za(fr);
  ASSERT_EQ(za.entries.size(), static_cast<size_t>(2));

  const auto& ab = za.entries.at("ab");
  EXPECT_EQ(ab.method, static_cast<uint16_t>(0));
  EXPECT_EQ(ab.comp_size, static_cast<uint64_t>(23));
  EXPECT_EQ(ab.uncomp_size, static_cast<uint64_t>(23));
  EXPECT_EQ(ab.local_off, ab_off);
  EXPECT_EQ(za.read_stored("ab"), std::string(23, 'x'));

  // Extra-field bytes in both headers must be skipped, not misparsed.
  const auto& d0 = za.entries.at("data/0");
  EXPECT_EQ(d0.method, static_cast<uint16_t>(0));
  EXPECT_EQ(d0.comp_size, static_cast<uint64_t>(100));
  EXPECT_EQ(d0.uncomp_size, static_cast<uint64_t>(100));
  EXPECT_EQ(d0.local_off, data_off);
  EXPECT_EQ(za.read_stored("data/0"), std::string(100, 'y'));

  std::remove(path.c_str());
}

TEST(MossZip, MissingEntryThrows) {
  using KokkosDNN::Moss::Zip::FileReader;
  using KokkosDNN::Moss::Zip::ZipArchive;
  uint64_t ab_off = 0, data_off = 0;
  const std::string path = write_temp(build_zip(ab_off, data_off));
  const FileReader fr(path);
  const ZipArchive za(fr);
  EXPECT_THROW(za.read_stored("nope"), std::runtime_error);
  std::remove(path.c_str());
}

TEST(MossZip, NonZipThrows) {
  using KokkosDNN::Moss::Zip::FileReader;
  using KokkosDNN::Moss::Zip::ZipArchive;
  const std::string path = write_temp(std::string(64, 'z'));
  const FileReader fr(path);
  EXPECT_THROW({ ZipArchive za(fr); }, std::runtime_error);
  std::remove(path.c_str());
}

}  // namespace
