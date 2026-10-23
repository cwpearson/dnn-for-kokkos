// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Tests for Common::IncrementalDetokenizer (streaming detokenizer with
// stop-string handling). All pieces are fed as raw byte strings, exactly
// as Tokenizer::decode({id}) would produce them.
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "KokkosDNN_Detokenizer.hpp"

namespace {

using KokkosDNN::Common::IncrementalDetokenizer;

IncrementalDetokenizer im_end_detok() {
  return IncrementalDetokenizer({"<|im_end|>"});
}

TEST(Detokenizer, AsciiPassthrough) {
  auto d = im_end_detok();
  auto s1 = d.push(1, "Hello, ");
  EXPECT_EQ(s1.delta, "Hello, ");
  EXPECT_FALSE(s1.stop_hit);
  auto s2 = d.push(2, "world");
  EXPECT_EQ(s2.delta, "world");
  EXPECT_FALSE(s2.stop_hit);
  EXPECT_EQ(d.buffered(), "");
  EXPECT_EQ(d.flush(), "");
}

TEST(Detokenizer, StopSplitAcrossThreePushes) {
  auto d = im_end_detok();
  // Each fragment is a proper prefix of "<|im_end|>" once accumulated, so
  // nothing is released until the match completes.
  auto s1 = d.push(1, "<|");
  EXPECT_EQ(s1.delta, "");
  EXPECT_FALSE(s1.stop_hit);
  EXPECT_EQ(d.buffered(), "<|");
  auto s2 = d.push(2, "im_en");
  EXPECT_EQ(s2.delta, "");
  EXPECT_FALSE(s2.stop_hit);
  auto s3 = d.push(3, "d|>");
  EXPECT_TRUE(s3.stop_hit);
  EXPECT_EQ(s3.matched, "<|im_end|>");
  EXPECT_EQ(s3.delta, "");
  EXPECT_TRUE(d.stopped());
  // Sticky: later pushes emit nothing; flush is empty.
  auto s4 = d.push(4, "more");
  EXPECT_TRUE(s4.stop_hit);
  EXPECT_EQ(s4.delta, "");
  EXPECT_EQ(d.flush(), "");
}

TEST(Detokenizer, StopTruncatesLeadingText) {
  auto d = im_end_detok();
  auto s1 = d.push(1, "Hi <|");
  EXPECT_EQ(s1.delta, "Hi ");  // safe part released, "<|" held back
  EXPECT_FALSE(s1.stop_hit);
  EXPECT_EQ(d.buffered(), "<|");
  auto s2 = d.push(2, "im_end|>");
  EXPECT_TRUE(s2.stop_hit);
  EXPECT_EQ(s2.matched, "<|im_end|>");
  EXPECT_EQ(s2.delta, "");
}

TEST(Detokenizer, SinglePushStopTruncatesAndDiscardsTail) {
  auto d = im_end_detok();
  auto s = d.push(1, "reply<|im_end|>extra");
  EXPECT_TRUE(s.stop_hit);
  EXPECT_EQ(s.matched, "<|im_end|>");
  EXPECT_EQ(s.delta, "reply");
  EXPECT_EQ(d.flush(), "");
}

TEST(Detokenizer, EarliestMatchWins) {
  IncrementalDetokenizer d({"STOP", "<|im_end|>"});
  auto s = d.push(1, "a STOP b<|im_end|>c");
  EXPECT_TRUE(s.stop_hit);
  EXPECT_EQ(s.matched, "STOP");
  EXPECT_EQ(s.delta, "a ");
}

TEST(Detokenizer, PartialUtf8HeldUntilComplete) {
  auto d = im_end_detok();
  // U+00E9 (e-acute) = 0xC3 0xA9, split across two pushes.
  auto s1 = d.push(1, "\xC3");
  EXPECT_EQ(s1.delta, "");
  EXPECT_FALSE(s1.stop_hit);
  auto s2 = d.push(2, "\xA9");
  EXPECT_FALSE(s2.stop_hit);
  EXPECT_EQ(s2.delta, "\xC3\xA9");
  // 3-byte U+20AC (euro) = 0xE2 0x82 0xAC, split across three pushes.
  auto t1 = d.push(3, "price \xE2");
  EXPECT_EQ(t1.delta, "price ");
  auto t2 = d.push(4, "\x82");
  EXPECT_EQ(t2.delta, "");
  auto t3 = d.push(5, "\xAC!");
  EXPECT_EQ(t3.delta, "\xE2\x82\xAC!");
}

TEST(Detokenizer, FlushEmitsRemainder) {
  auto d = im_end_detok();
  auto s = d.push(1, "abc<|");
  EXPECT_EQ(s.delta, "abc");
  EXPECT_EQ(d.buffered(), "<|");
  // End of stream: no more bytes are coming, so the held-back "<|" is
  // data, not a signal.
  EXPECT_EQ(d.flush(), "<|");
  EXPECT_EQ(d.buffered(), "");
}

TEST(Detokenizer, ResetClears) {
  auto d = im_end_detok();
  auto s = d.push(1, "x<|im_end|>");
  EXPECT_TRUE(s.stop_hit);
  d.reset();
  EXPECT_FALSE(d.stopped());
  EXPECT_EQ(d.buffered(), "");
  auto s2 = d.push(2, "ok");
  EXPECT_FALSE(s2.stop_hit);
  EXPECT_EQ(s2.delta, "ok");
}

TEST(Detokenizer, EmptyStopListPassesThrough) {
  IncrementalDetokenizer d({});
  auto s = d.push(1, "anything <| goes");
  EXPECT_FALSE(s.stop_hit);
  EXPECT_EQ(s.delta, "anything <| goes");
  EXPECT_EQ(d.flush(), "");
}

TEST(Detokenizer, EmptyStopStringsIgnored) {
  IncrementalDetokenizer d({"", "<|im_end|>"});
  auto s = d.push(1, "clean text");
  EXPECT_FALSE(s.stop_hit);
  EXPECT_EQ(s.delta, "clean text");
}
}  // namespace
