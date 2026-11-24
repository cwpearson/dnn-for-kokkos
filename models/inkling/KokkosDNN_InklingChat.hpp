// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLING_CHAT_HPP
#define KOKKOSDNN_INKLING_CHAT_HPP

// Inkling 2 Mini chat template (thinkingmachines/Inkling-2-Mini
// chat_template.jinja semantics, header-only).
//
// Role tokens: user -> <|message_user|>, assistant -> <|message_model|>,
// system -> <|message_system|>, tool -> <|message_tool|> (tool messages
// append the tool name, resolved via tool_call_id against prior assistant
// tool_calls). Per-message bodies use <|content_text|>, each closed by
// <|end_message|>. Assistant reasoning_content replays as
// '<|message_model|><|content_thinking|>...<|end_message|>'. A
// thinking-effort system message
// ('<|message_system|><|content_text|>Thinking effort level: N<|end_message|>')
// is emitted before the first non-system message; reasoning_effort maps
// none/minimal/low/medium/high/xhigh/max -> 0/0.1/0.2/0.7/0.9/0.99/0.99
// (default 0.9), numeric strings pass through when in [0.0, 0.99],
// anything else throws std::invalid_argument. Tools are declared as
// '<|message_system|>tool_declare<|content_xml|>{json}<|end_message|>'
// (sorted-keys compact JSON specs); calls render as
// '<|message_model|>NAME<|content_invoke_tool_json|>{"name":...,"args":{sorted}}
// <|end_message|>' (args must be an object with sorted keys, never a JSON
// string). Every assistant message appends
// '<|content_model_end_sampling|>' (id 200006, the generation-config eos).
// No trailing '<|message_model|>' unless add_generation_prompt.
//
// Default stops: ids {200006}, strings {'<|content_model_end_sampling|>'}.

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "KokkosDNN_ChatStop.hpp"
#include "KokkosDNN_InklingTokenizer.hpp"

namespace KokkosDNN {
namespace Inkling {

// A single tool invocation. `arguments` is a JSON object string (OpenAI
// `function.arguments` wire form).
struct ToolCall {
  std::string id;         // OpenAI tool_call id ("" when unknown/unparsed)
  std::string name;       // function name
  std::string arguments;  // JSON object string ("" or "{}" when none)
};

struct ChatMessage {
  std::string role;     // "system" | "user" | "assistant" | "tool"
  std::string content;
  std::string thinking;  // assistant reasoning trace (replays as the
                         // <|content_thinking|> block)
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
  std::string name;  // tool message function name (may be empty)
};

namespace Impl {

// ---- Minimal JSON value model (tools + tool-call args canonicalization) ----
struct JsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool boolean = false;
  std::string number;  // raw literal, preserved verbatim
  std::string str;
  std::vector<JsonValue> arr;
  std::map<std::string, JsonValue> obj;  // sorted by key
};

struct JsonParser {
  const std::string& s;
  size_t p = 0;
  explicit JsonParser(const std::string& text) : s(text) {}

  [[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("inkling chat: malformed JSON (" + what + ")");
  }

  void ws() {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' ||
                            s[p] == '\r'))
      ++p;
  }

  char peek() {
    if (p >= s.size()) fail("unexpected end");
    return s[p];
  }

  JsonValue parse_value() {
    ws();
    char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') {
      JsonValue v;
      v.type = JsonValue::Type::String;
      v.str  = parse_string();
      return v;
    }
    if (c == 't' || c == 'f' || c == 'n') return parse_literal();
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    fail("unexpected character");
  }

  std::string parse_string() {
    if (s[p] != '"') fail("expected string");
    ++p;
    std::string out;
    while (true) {
      if (p >= s.size()) fail("unterminated string");
      char c = s[p++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (p >= s.size()) fail("bad escape");
      char e = s[p++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (p + 4 > s.size()) fail("bad \\u escape");
          unsigned cp = 0;
          for (int i = 0; i < 4; ++i) {
            char h = s[p++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
            else fail("bad \\u escape");
          }
          if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 <= s.size() &&
              s[p] == '\\' && s[p + 1] == 'u') {
            p += 2;
            unsigned lo = 0;
            for (int i = 0; i < 4; ++i) {
              char h = s[p++];
              lo <<= 4;
              if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
              else if (h >= 'a' && h <= 'f')
                lo |= (unsigned)(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F')
                lo |= (unsigned)(h - 'A' + 10);
              else fail("bad \\u escape");
            }
            if (lo >= 0xDC00 && lo <= 0xDFFF)
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          if (cp < 0x80) {
            out += (char)cp;
          } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
          } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
          } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
          }
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }

  JsonValue parse_literal() {
    JsonValue v;
    if (s.compare(p, 4, "true") == 0) {
      p += 4;
      v.type    = JsonValue::Type::Bool;
      v.boolean = true;
      return v;
    }
    if (s.compare(p, 5, "false") == 0) {
      p += 5;
      v.type    = JsonValue::Type::Bool;
      v.boolean = false;
      return v;
    }
    if (s.compare(p, 4, "null") == 0) {
      p += 4;
      v.type = JsonValue::Type::Null;
      return v;
    }
    fail("bad literal");
  }

  JsonValue parse_number() {
    const size_t b = p;
    if (p < s.size() && s[p] == '-') ++p;
    if (p >= s.size()) fail("bad number");
    if (s[p] == '0') {
      ++p;
    } else if (s[p] >= '1' && s[p] <= '9') {
      while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
    } else {
      fail("bad number");
    }
    if (p < s.size() && s[p] == '.') {
      ++p;
      if (p >= s.size() || s[p] < '0' || s[p] > '9') fail("bad number");
      while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
    }
    if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
      ++p;
      if (p < s.size() && (s[p] == '+' || s[p] == '-')) ++p;
      if (p >= s.size() || s[p] < '0' || s[p] > '9') fail("bad number");
      while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
    }
    JsonValue v;
    v.type   = JsonValue::Type::Number;
    v.number = s.substr(b, p - b);
    return v;
  }

  JsonValue parse_array() {
    JsonValue v;
    v.type = JsonValue::Type::Array;
    ++p;  // [
    ws();
    if (p < s.size() && s[p] == ']') {
      ++p;
      return v;
    }
    while (true) {
      v.arr.push_back(parse_value());
      ws();
      if (p >= s.size()) fail("unbalanced array");
      if (s[p] == ',') {
        ++p;
        continue;
      }
      if (s[p] == ']') {
        ++p;
        return v;
      }
      fail("expected ',' or ']'");
    }
  }

  JsonValue parse_object() {
    JsonValue v;
    v.type = JsonValue::Type::Object;
    ++p;  // {
    ws();
    if (p < s.size() && s[p] == '}') {
      ++p;
      return v;
    }
    while (true) {
      ws();
      if (p >= s.size() || s[p] != '"') fail("expected string key");
      std::string key = parse_string();
      ws();
      if (p >= s.size() || s[p] != ':') fail("expected ':'");
      ++p;
      v.obj[key] = parse_value();
      ws();
      if (p >= s.size()) fail("unbalanced object");
      if (s[p] == ',') {
        ++p;
        continue;
      }
      if (s[p] == '}') {
        ++p;
        return v;
      }
      fail("expected ',' or '}'");
    }
  }
};

inline JsonValue parse_json(const std::string& text) {
  JsonParser ps(text);
  JsonValue v = ps.parse_value();
  ps.ws();
  if (ps.p != text.size())
    ps.fail("trailing characters after top-level value");
  return v;
}

// JSON string escaping matching Jinja2's tojson (htmlsafe): short escapes,
// \u00XX for other controls, \u003c/\u003e/\u0026/\u0027 for <>&',
// UTF-8 passthrough otherwise.
inline void append_json_string(const std::string& in, std::string& out) {
  static const char* hex = "0123456789abcdef";
  out += '"';
  for (unsigned char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out += hex[(c >> 4) & 0xF];
          out += hex[c & 0xF];
        } else if (c == '<') {
          out += "\\u003c";
        } else if (c == '>') {
          out += "\\u003e";
        } else if (c == '&') {
          out += "\\u0026";
        } else if (c == '\'') {
          out += "\\u0027";
        } else {
          out += (char)c;
        }
    }
  }
  out += '"';
}

// Compact serialization with all object keys sorted (std::map order),
// matching `tojson(sort_keys=true, separators=(",", ":"))`.
inline void append_json_canonical(const JsonValue& v, std::string& out) {
  switch (v.type) {
    case JsonValue::Type::Null: out += "null"; return;
    case JsonValue::Type::Bool: out += (v.boolean ? "true" : "false"); return;
    case JsonValue::Type::Number: out += v.number; return;
    case JsonValue::Type::String: append_json_string(v.str, out); return;
    case JsonValue::Type::Array: {
      out += '[';
      bool first = true;
      for (const auto& e : v.arr) {
        if (!first) out += ',';
        first = false;
        append_json_canonical(e, out);
      }
      out += ']';
      return;
    }
    case JsonValue::Type::Object: {
      out += '{';
      bool first = true;
      for (const auto& [k, e] : v.obj) {
        if (!first) out += ',';
        first = false;
        append_json_string(k, out);
        out += ':';
        append_json_canonical(e, out);
      }
      out += '}';
      return;
    }
  }
}

inline std::string json_canonical(const JsonValue& v) {
  std::string out;
  append_json_canonical(v, out);
  return out;
}

inline std::string trim_ws(const std::string& t) {
  size_t a = 0;
  while (a < t.size() && (t[a] == ' ' || t[a] == '\t' || t[a] == '\n' ||
                          t[a] == '\r'))
    ++a;
  size_t b = t.size();
  while (b > a && (t[b - 1] == ' ' || t[b - 1] == '\t' || t[b - 1] == '\n' ||
                   t[b - 1] == '\r'))
    --b;
  return t.substr(a, b - a);
}

// Shortest round-trip double formatting ("0.9", "0.99", ...), matching the
// template's `{{ num }}` rendering for values in [0, 0.99].
inline std::string format_effort_number(double f) {
  char buf[32];
  auto res = std::to_chars(buf, buf + sizeof(buf), f);
  if (res.ec == std::errc() && res.ptr != buf)
    return std::string(buf, res.ptr);
  std::snprintf(buf, sizeof(buf), "%.17g", f);
  return std::string(buf);
}

}  // namespace Impl

// Resolve a reasoning_effort cue to its rendered number string
// ("0", "0.1", "0.2", "0.7", "0.9", "0.99"). "" selects the 0.9 default.
// Matching is case-insensitive (consistent with the server-side
// normalization; the upstream Jinja map itself is lowercase). Numeric
// strings pass through when in [0.0, 0.99] (0.0 renders "0").
// Anything else throws std::invalid_argument.
inline std::string resolve_reasoning_effort(const std::string& effort) {
  std::string k = Impl::trim_ws(effort);
  for (auto& ch : k)
    ch = (char)std::tolower((unsigned char)ch);
  if (k.empty() || k == "high") return "0.9";
  if (k == "none") return "0";
  if (k == "minimal") return "0.1";
  if (k == "low") return "0.2";
  if (k == "medium") return "0.7";
  if (k == "xhigh" || k == "max") return "0.99";
  try {
    size_t n = 0;
    double f = std::stod(k, &n);
    if (n == k.size() && f >= 0.0 && f <= 0.99) {
      if (f == 0.0) return "0";
      return Impl::format_effort_number(f);
    }
  } catch (...) {
  }
  throw std::invalid_argument("unknown reasoning_effort '" + effort +
                              "' (expected none, minimal, low, medium, high,"
                              " xhigh, max, or a float in [0.0, 0.99])");
}

// Build the standard single-turn messages for a raw prompt: optional
// system message first, then one `role` (default "user") turn.
inline std::vector<ChatMessage> make_chat_messages(
    const std::string& prompt, const std::string& system = "",
    const std::string& role = "user") {
  return ::KokkosDNN::Common::Chat::make_chat_messages<ChatMessage>(
      prompt, system, role);
}

// Render messages with tool support, following the upstream Inkling 2 Mini
// chat template (see the header comment). Throws std::invalid_argument on
// unknown roles/effort values or malformed tool JSON; never throws on
// message content (rendered verbatim).
//
// `expert_present` is reserved for future use (currently no template
// effect); it is accepted so the server-side encode signature stays
// stable.
inline std::string render_inkling_chatml_with_tools(
    const std::vector<ChatMessage>& messages, const std::string& tools_json,
    const std::string& system_suffix = "", bool add_generation_prompt = true,
    bool /*expert_present*/ = true,
    const std::string& reasoning_strength = "high") {
  const std::string effort = resolve_reasoning_effort(reasoning_strength);
  std::string out;

  // Tool catalogue (sorted-keys compact JSON), when tools are given.
  bool have_tools = false;
  {
    std::string t = Impl::trim_ws(tools_json);
    if (!t.empty() && t != "null") {
      Impl::JsonValue tools = Impl::parse_json(t);
      if (tools.type != Impl::JsonValue::Type::Array)
        throw std::invalid_argument(
            "inkling chat: 'tools' must be a JSON array");
      have_tools = !tools.arr.empty();
      if (have_tools) {
        Impl::JsonValue specs;
        specs.type = Impl::JsonValue::Type::Array;
        for (const auto& tool : tools.arr) {
          const Impl::JsonValue* fn = &tool;
          if (tool.type == Impl::JsonValue::Type::Object) {
            auto it = tool.obj.find("function");
            if (it != tool.obj.end() &&
                it->second.type == Impl::JsonValue::Type::Object)
              fn = &it->second;
          }
          Impl::JsonValue spec;
          spec.type = Impl::JsonValue::Type::Object;
          auto field = [&](const char* key, const Impl::JsonValue& v) {
            spec.obj[key] = v;
          };
          Impl::JsonValue empty_str, empty_obj, func_type;
          empty_str.type = Impl::JsonValue::Type::String;
          empty_obj.type = Impl::JsonValue::Type::Object;
          func_type.type = Impl::JsonValue::Type::String;
          func_type.str  = "function";
          if (fn->type == Impl::JsonValue::Type::Object) {
            auto it = fn->obj.find("description");
            field("description", (it != fn->obj.end() &&
                                  it->second.type ==
                                      Impl::JsonValue::Type::String)
                                     ? it->second
                                     : empty_str);
            it = fn->obj.find("name");
            if (it != fn->obj.end())
              field("name", it->second);
            else {
              Impl::JsonValue null;
              field("name", null);
            }
            it = fn->obj.find("parameters");
            field("parameters", (it != fn->obj.end() &&
                                 it->second.type ==
                                     Impl::JsonValue::Type::Object)
                                    ? it->second
                                    : empty_obj);
          } else {
            Impl::JsonValue null;
            field("description", empty_str);
            field("name", null);
            field("parameters", empty_obj);
          }
          if (tool.type == Impl::JsonValue::Type::Object) {
            auto it = tool.obj.find("type");
            field("type", (it != tool.obj.end() &&
                           it->second.type == Impl::JsonValue::Type::String)
                              ? it->second
                              : func_type);
          } else {
            field("type", func_type);
          }
          specs.arr.push_back(std::move(spec));
        }
        out += "<|message_system|>tool_declare<|content_xml|>";
        out += Impl::json_canonical(specs);
        out += "<|end_message|>";
      }
    }
  }

  // Effective messages: system_suffix rides on the last system message
  // (or opens the conversation when there is none).
  std::vector<ChatMessage> msgs = messages;
  if (!system_suffix.empty()) {
    bool folded = false;
    for (size_t i = msgs.size(); i-- > 0;) {
      if (msgs[i].role == "system") {
        if (!msgs[i].content.empty()) msgs[i].content += "\n\n";
        msgs[i].content += system_suffix;
        folded = true;
        break;
      }
    }
    if (!folded) {
      ChatMessage m;
      m.role    = "system";
      m.content = system_suffix;
      msgs.insert(msgs.begin(), std::move(m));
    }
  }

  auto role_token = [](const std::string& role) -> std::string {
    if (role == "user") return "<|message_user|>";
    if (role == "assistant") return "<|message_model|>";
    if (role == "system") return "<|message_system|>";
    if (role == "tool") return "<|message_tool|>";
    throw std::invalid_argument("inkling chat: unknown message role '" +
                                role + "'");
  };

  bool effort_emitted = false;
  for (size_t i = 0; i < msgs.size(); ++i) {
    const ChatMessage& m = msgs[i];
    // Validate the role even for system messages (unknown roles raise).
    std::string header = role_token(m.role);
    if (!effort_emitted && m.role != "system") {
      out += "<|message_system|><|content_text|>Thinking effort level: ";
      out += effort;
      out += "<|end_message|>";
      effort_emitted = true;
    }
    if (m.role == "tool") {
      if (!m.name.empty()) {
        header += m.name;
      } else if (!m.tool_call_id.empty()) {
        // Resolve via tool_call_id against prior assistant tool_calls.
        for (size_t j = 0; j < i; ++j) {
          if (msgs[j].role != "assistant") continue;
          for (const auto& tc : msgs[j].tool_calls) {
            if (!tc.id.empty() && tc.id == m.tool_call_id && !tc.name.empty())
              header += tc.name;
          }
        }
      }
    } else if (m.role == "assistant" && !m.name.empty()) {
      header += m.name;
    }

    if (m.role == "assistant" && !m.thinking.empty()) {
      out += header;
      out += "<|content_thinking|>";
      out += m.thinking;
      out += "<|end_message|>";
    }

    {
      const bool has_calls =
          (m.role == "assistant" && !m.tool_calls.empty());
      const bool has_reason =
          (m.role == "assistant" && !m.thinking.empty());
      if (!m.content.empty() || !(has_calls || has_reason)) {
        out += header;
        out += "<|content_text|>";
        out += m.content;
        out += "<|end_message|>";
      }
    }

    if (m.role == "assistant") {
      for (const auto& tc : m.tool_calls) {
        if (tc.name.empty())
          throw std::invalid_argument(
              "inkling chat: tool call function name must be a string");
        if (!m.name.empty() && m.name != tc.name)
          throw std::invalid_argument(
              "inkling chat: assistant name '" + m.name +
              "' does not match its tool call '" + tc.name + "'");
        std::string args = "{}";
        {
          std::string t = Impl::trim_ws(tc.arguments);
          if (!t.empty() && t != "{}" && t != "null") {
            Impl::JsonValue v = Impl::parse_json(t);
            if (v.type != Impl::JsonValue::Type::Object)
              throw std::invalid_argument(
                  "inkling chat: tool call arguments must be an object");
            args = Impl::json_canonical(v);
          }
        }
        out += "<|message_model|>";
        out += tc.name;
        out += "<|content_invoke_tool_json|>";
        out += "{\"name\":";
        Impl::append_json_string(tc.name, out);
        out += ",\"args\":";
        out += args;
        out += "}";
        out += "<|end_message|>";
      }
      out += "<|content_model_end_sampling|>";
    }
  }
  if (!effort_emitted) {
    out += "<|message_system|><|content_text|>Thinking effort level: ";
    out += effort;
    out += "<|end_message|>";
  }
  if (add_generation_prompt) out += "<|message_model|>";
  return out;
}

// Render + encode with tool support. The rendered template carries no BOS
// marker (upstream adds none), so the text encodes directly with
// add_bos=false.
inline std::vector<int> encode_chatml_with_tools(
    const Tokenizer& tok, const std::vector<ChatMessage>& messages,
    const std::string& tools_json, const std::string& system_suffix = "",
    bool add_generation_prompt = true, bool expert_present = true,
    const std::string& reasoning_strength = "high") {
  (void)expert_present;  // reserved; no template effect (see above)
  return tok.encode(render_inkling_chatml_with_tools(
                        messages, tools_json, system_suffix,
                        add_generation_prompt, expert_present,
                        reasoning_strength),
                    false);
}

// Default stop strings: the chat turn terminator
// <|content_model_end_sampling|> (the generation-config eos id 200006 stays
// covered by the stop-id side of the Session policy). <|end_message|> is a
// per-message structural boundary, never a stop.
inline std::vector<std::string> default_stop_strings() {
  return {"<|content_model_end_sampling|>"};
}

// ---------------------------------------------------------------------------
// Tool-call parsing (API parity with LFM25/Glimmer parse_tool_calls /
// ToolCallStreamParser so generic server code compiles unchanged).
//
// Splits a raw completion (decoded with skip_special_tokens=false) into
// display text, the <|content_thinking|> reasoning trace, and tool calls
// parsed from <|content_invoke_tool_json|> blocks, following the upstream
// response_template in tokenizer_config.json:
//
//   reasoning_content: open (?:<|message_model|>...)?<|content_thinking|>
//     close <|end_message|> (repeats, joined)
//   content: open (?:<|message_model|>...)?<|content_(text|tool_error|
//     invoke_tool_text)|> close <|end_message|> (repeats, joined)
//   tool_calls: open (?:<|message_model|>...)?<|content_invoke_tool_json|>
//     close <|end_message|> (repeats; content is JSON
//     {"name":...,"args":{...}}, transformed to name={content.name}
//     arguments={content.args})
//
// Channel headers/footers (<|message_model|> plus its NAME suffix,
// <|content_*|> openers, <|end_message|> closers) never leak into either
// text view. The display text is additionally cut at the first stop string
// (with <|content_model_end_sampling|> always applying, independent of
// `stops`); <|end_message|> is structural, never a stop. A length-capped
// turn cut mid-marker has its longest trailing partial marker head
// stripped, so fragments never leak into `content`. Never throws on model
// output (malformed invoke blocks are dropped).
// ---------------------------------------------------------------------------
struct ToolParseResult {
  std::string text;
  std::string reasoning;  // <|content_thinking|> trace ("" when none)
  std::vector<ToolCall> calls;
};

struct StreamDeltas {
  std::string display;
  std::string reasoning;
};

namespace Impl {

// Inkling channel markers (fixed literals). <|unused_...|> fillers are
// handled by prefix, not listed here.
inline bool inkling_starts_with(const std::string& s, size_t pos,
                                const char* prefix, size_t len) {
  return s.size() - pos >= len && s.compare(pos, len, prefix, len) == 0;
}

// Longest suffix of s that is a strict prefix of any token in toks
// (bounded by max_len). Used for hold-backs and length-cap stripping.
inline size_t inkling_suffix_overlap(const std::string& s,
                                     const std::vector<std::string>& toks,
                                     size_t max_len) {
  const size_t n = s.size();
  if (n == 0) return 0;
  size_t best = 0;
  const size_t cap = (n < max_len) ? n : max_len;
  for (size_t k = 1; k <= cap; ++k) {
    for (const auto& t : toks) {
      if (t.size() > k && t.compare(0, k, s, n - k, k) == 0) {
        best = k;
        break;
      }
    }
  }
  return best;
}

// Length of a trailing incomplete UTF-8 sequence (0 when the tail ends on
// a character boundary).
inline size_t inkling_utf8_hold(const std::string& s) {
  size_t n = 0;
  for (size_t i = s.size(); i > 0 && n < 4; --i, ++n) {
    const unsigned char c = static_cast<unsigned char>(s[i - 1]);
    if ((c & 0x80) == 0) return 0;
    if ((c & 0xC0) == 0xC0) {
      size_t need = 1;
      if ((c & 0xE0) == 0xC0)
        need = 2;
      else if ((c & 0xF0) == 0xE0)
        need = 3;
      else if ((c & 0xF8) == 0xF0)
        need = 4;
      return (n + 1 >= need) ? 0 : (n + 1);
    }
  }
  return s.size();
}

// Parse one <|content_invoke_tool_json|> body into a call. The body must
// be a JSON object with a non-empty string "name" and an object "args"
// (missing/null "args" means "{}"). Returns true with `out` set on
// success; malformed bodies report false (caller drops the block).
inline bool parse_invoke_json(const std::string& body, ToolCall& out) {
  std::string t = trim_ws(body);
  if (t.empty()) return false;
  JsonValue v;
  try {
    v = parse_json(t);
  } catch (...) {
    return false;
  }
  if (v.type != JsonValue::Type::Object) return false;
  auto nit = v.obj.find("name");
  if (nit == v.obj.end() ||
      nit->second.type != JsonValue::Type::String ||
      nit->second.str.empty())
    return false;
  std::string args = "{}";
  auto ait = v.obj.find("args");
  if (ait != v.obj.end() && ait->second.type != JsonValue::Type::Null) {
    if (ait->second.type != JsonValue::Type::Object) return false;
    args = json_canonical(ait->second);
  }
  out.id.clear();
  out.name = nit->second.str;
  out.arguments = args.empty() ? std::string("{}") : args;
  return true;
}

}  // namespace Impl

class ToolCallStreamParser {
 public:
  explicit ToolCallStreamParser(
      const std::vector<std::string>& stop_strings = {});
  // One parser step for one raw piece: returns display + reasoning text
  // newly safe to stream (either may be empty = "hold"). Prefer this over
  // push() on hot paths.
  StreamDeltas feed(const std::string& raw_piece);
  // Returns display text newly safe to stream (may be empty = "hold").
  // Stashes the reasoning delta for the next poll_reasoning() call.
  std::string push(const std::string& raw_piece);
  // Returns reasoning text accumulated since the last poll (may be empty).
  std::string poll_reasoning();
  // Tail text + tail reasoning + parsed calls. Idempotent.
  ToolParseResult finish();

 private:
  enum class Channel { kOutside, kThinking, kDisplay, kInvoke };

  void drive(bool end_of_input);
  void route_text(const std::string& s);
  void release_display();
  void release_reasoning();
  StreamDeltas emit_deltas();
  // Parse invoke_buf_ into calls_ when valid (always clears the buffer).
  void close_invoke();

  Channel cur_ = Channel::kOutside;
  std::string hold_;       // undecided bytes (bounded, see drive())
  bool awaiting_name_ = false;  // just saw <|message_model|>; Next [^<]*
                                // may be the NAME prefix of an invoke block
  std::string name_buf_;   // buffered NAME candidate (dropped before any
                           // <|content_*|> opener, else flushed as display)
  std::string invoke_buf_;  // bytes of the current invoke-JSON element
  std::string acc_d_;       // display bytes (append-only)
  std::string acc_r_;       // reasoning bytes (append-only)
  size_t rel_d_ = 0;        // settled display frontier
  size_t rel_r_ = 0;        // settled reasoning frontier
  size_t emitted_d_ = 0;    // display chars already returned
  size_t emitted_r_ = 0;    // reasoning chars already returned
  std::string stash_r_;  // reasoning delta held for poll_reasoning()
  std::vector<ToolCall> calls_;
  bool stopped_ = false;  // display cut (stop found or eos seen)
  bool eot_seen_ = false;  // structural turn end seen (drop all)
  size_t eot_at_ = std::string::npos;
  bool finished_ = false;
  ToolParseResult tail_;  // cached finish() result
  std::vector<std::string> stops_;  // configured stop strings (no empties)
  size_t max_stop_len_ = 1;
  std::vector<std::string> marker_hold_;  // release hold-back markers
};

inline ToolCallStreamParser::ToolCallStreamParser(
    const std::vector<std::string>& stop_strings) {
  for (const auto& s : stop_strings) {
    if (s.empty()) continue;
    stops_.push_back(s);
  }
  // The turn terminator always ends the turn, even with custom stops
  // (mirrors the Session stop policy union).
  if (std::find(stops_.begin(), stops_.end(),
                "<|content_model_end_sampling|>") == stops_.end())
    stops_.push_back("<|content_model_end_sampling|>");
  max_stop_len_ = 1;
  for (const auto& s : stops_)
    max_stop_len_ = std::max(max_stop_len_, s.size());
  static const char* kHold[] = {
      "<|message_model|>",   "<|message_user|>",   "<|message_system|>",
      "<|message_tool|>",    "<|content_thinking|>", "<|content_text|>",
      "<|content_tool_error|>", "<|content_invoke_tool_text|>",
      "<|content_invoke_tool_json|>", "<|content_xml|>", "<|content_image|>",
      "<|content_model_end_sampling|>", "<|begin_of_text|>", "<|end_message|>",
      "<|endoftext|>",       "<|unused",            "<|content_"};
  for (const char* h : kHold) marker_hold_.emplace_back(h);
}

inline void ToolCallStreamParser::route_text(const std::string& s) {
  if (s.empty()) return;
  if (cur_ == Channel::kThinking) {
    acc_r_ += s;
    return;
  }
  if (cur_ == Channel::kInvoke) {
    // Invoke JSON accumulates even after a display stop so a call straddling
    // the cut still parses; its bytes never reach display.
    invoke_buf_ += s;
    return;
  }
  if (stopped_) return;
  acc_d_ += s;
}

inline void ToolCallStreamParser::close_invoke() {
  ToolCall c;
  if (Impl::parse_invoke_json(invoke_buf_, c)) calls_.push_back(std::move(c));
  invoke_buf_.clear();
}

inline void ToolCallStreamParser::release_display() {
  // The turn ends at the earliest of a custom stop and the structural
  // terminator: when the terminator fired first (eot_at_ set), a custom
  // stop earlier in the turn still wins.
  const size_t limit = (eot_at_ == std::string::npos)
                           ? acc_d_.size()
                           : std::min(eot_at_, acc_d_.size());
  const size_t from = (rel_d_ >= max_stop_len_) ? rel_d_ - max_stop_len_ + 1 : 0;
  size_t best = std::string::npos;
  for (const auto& s : stops_) {
    const size_t p = acc_d_.find(s, from);
    if (p != std::string::npos && p < limit &&
        (best == std::string::npos || p < best))
      best = p;
  }
  if (best != std::string::npos) {
    acc_d_.resize(best);
    stopped_ = true;
    rel_d_ = acc_d_.size();
    if (emitted_d_ > rel_d_) emitted_d_ = rel_d_;
    return;
  }
  if (stopped_) {
    rel_d_ = acc_d_.size();
    return;
  }
  const size_t hold_stop =
      Impl::inkling_suffix_overlap(acc_d_, stops_, max_stop_len_);
  const size_t hold_mark =
      Impl::inkling_suffix_overlap(acc_d_, marker_hold_, 64);
  const size_t hold =
      std::max(hold_stop, std::max(hold_mark, Impl::inkling_utf8_hold(acc_d_)));
  const size_t rel = acc_d_.size() - hold;
  if (rel > rel_d_) rel_d_ = rel;
}

inline void ToolCallStreamParser::release_reasoning() {
  // Stops apply to display only; reasoning holds back partial UTF-8 plus
  // marker-prefix tails (same strip agreement as display).
  const size_t hold = std::max(
      Impl::inkling_suffix_overlap(acc_r_, marker_hold_, 64),
      Impl::inkling_utf8_hold(acc_r_));
  const size_t rel = acc_r_.size() - hold;
  if (rel > rel_r_) rel_r_ = rel;
}

inline StreamDeltas ToolCallStreamParser::emit_deltas() {
  release_display();
  release_reasoning();
  StreamDeltas out;
  out.display = acc_d_.substr(emitted_d_, rel_d_ - emitted_d_);
  out.reasoning = acc_r_.substr(emitted_r_, rel_r_ - emitted_r_);
  emitted_d_ = rel_d_;
  emitted_r_ = rel_r_;
  return out;
}

inline void ToolCallStreamParser::drive(bool end_of_input) {
  static const char* kMsgModel = "<|message_model|>";
  static const char* kThink = "<|content_thinking|>";
  static const char* kText = "<|content_text|>";
  static const char* kToolError = "<|content_tool_error|>";
  static const char* kInvokeText = "<|content_invoke_tool_text|>";
  static const char* kInvokeJson = "<|content_invoke_tool_json|>";
  static const char* kXml = "<|content_xml|>";
  static const char* kImage = "<|content_image|>";
  static const char* kEndSampling = "<|content_model_end_sampling|>";
  static const char* kEndMessage = "<|end_message|>";
  static const char* kEndOfText = "<|endoftext|>";
  static const char* kBegin = "<|begin_of_text|>";
  static const char* kMsgUser = "<|message_user|>";
  static const char* kMsgSystem = "<|message_system|>";
  static const char* kMsgTool = "<|message_tool|>";
  // Fixed markers, longest first so a shared prefix resolves to the longest
  // match.
  static const char* kFixed[] = {
      kInvokeJson, kEndSampling, kInvokeText, kToolError, kThink,
      kMsgModel,  kMsgSystem,    kMsgTool,    kMsgUser,   kText,
      kEndMessage, kEndOfText,   kBegin,      kXml,       kImage};
  static const size_t kNumFixed = 15;
  static const size_t kNameMax = 256;
  static const size_t kHoldMax = 4096;

  while (true) {
    if (eot_seen_) {
      hold_.clear();
      return;
    }
    // Buffer a NAME candidate following <|message_model|>: everything up to
    // the next '<' (bounded). It resolves once the next marker is known.
    if (awaiting_name_) {
      const size_t lt = hold_.find('<');
      if (lt == std::string::npos) {
        if (end_of_input) {
          // Length-cap fragment of a cut invoke header: structural, drop.
          name_buf_.clear();
          awaiting_name_ = false;
          hold_.clear();
          continue;
        }
        if (name_buf_.size() + hold_.size() > kNameMax) {
          // Not a NAME prefix (unbounded text): flush as display.
          route_text(name_buf_);
          name_buf_.clear();
          awaiting_name_ = false;
          route_text(hold_);
          hold_.clear();
          continue;
        }
        name_buf_ += hold_;
        hold_.clear();
        return;
      }
      if (lt > 0) {
        if (name_buf_.size() + lt > kNameMax) {
          route_text(name_buf_);
          name_buf_.clear();
          awaiting_name_ = false;
          route_text(hold_.substr(0, lt));
          hold_.erase(0, lt);
          continue;
        }
        name_buf_ += hold_.substr(0, lt);
        hold_.erase(0, lt);
      }
      // Fall through with hold_ starting at '<': the marker match below
      // decides whether the NAME is structural (dropped) or text.
    }
    // Route text before the next '<' (no marker can hide inside it).
    if (!hold_.empty() && hold_[0] != '<') {
      const size_t lt = hold_.find('<');
      const std::string run =
          (lt == std::string::npos) ? hold_ : hold_.substr(0, lt);
      route_text(run);
      hold_.erase(0, run.size());
      continue;
    }
    if (hold_.empty()) return;
    // hold_ starts with '<' here. Fixed markers, longest first.
    bool consumed = false;
    for (size_t m = 0; m < kNumFixed; ++m) {
      const size_t n = std::strlen(kFixed[m]);
      if (!Impl::inkling_starts_with(hold_, 0, kFixed[m], n)) continue;
      consumed = true;
      const std::string mark(kFixed[m]);
      hold_.erase(0, n);
      const bool is_content = (mark == kThink || mark == kText ||
                               mark == kToolError || mark == kInvokeText ||
                               mark == kInvokeJson || mark == kXml ||
                               mark == kImage);
      if (awaiting_name_) {
        // A NAME prefix is structural only before a content opener; any
        // other marker means the buffered text was stray display.
        if (is_content) {
          name_buf_.clear();
        } else {
          route_text(name_buf_);
          name_buf_.clear();
        }
        awaiting_name_ = false;
      }
      if (mark == kMsgModel) {
        // A previous NAME still pending (back-to-back headers) flushes as
        // display above; start buffering the new NAME candidate.
        awaiting_name_ = true;
        name_buf_.clear();
      } else if (mark == kThink) {
        if (cur_ == Channel::kInvoke) close_invoke();
        cur_ = Channel::kThinking;
      } else if (mark == kText || mark == kToolError || mark == kInvokeText) {
        if (cur_ == Channel::kInvoke) close_invoke();
        cur_ = Channel::kDisplay;
      } else if (mark == kInvokeJson) {
        if (cur_ == Channel::kInvoke) close_invoke();
        cur_ = Channel::kInvoke;
        invoke_buf_.clear();
      } else if (mark == kEndMessage) {
        if (cur_ == Channel::kInvoke) close_invoke();
        cur_ = Channel::kOutside;
      } else if (mark == kEndSampling || mark == kEndOfText) {
        if (cur_ == Channel::kInvoke) close_invoke();
        stopped_ = true;
        eot_seen_ = true;
        eot_at_ = acc_d_.size();
        hold_.clear();
        return;
      }
      // Prompt-only markers (user/system/tool headers, xml/image bodies,
      // BOS) drop silently.
      break;
    }
    if (consumed) continue;
    // Fillers: <|unused...|> through '>' drop silently (prompt image
    // placeholder and reserved tokens never carry text).
    if (Impl::inkling_starts_with(hold_, 0, "<|unused", 8)) {
      const size_t gt = hold_.find('>', 8);
      const size_t bad = hold_.find('<', 8);
      if (gt != std::string::npos && (bad == std::string::npos || gt < bad) &&
          gt <= 64) {
        if (awaiting_name_) {
          route_text(name_buf_);
          name_buf_.clear();
          awaiting_name_ = false;
        }
        hold_.erase(0, gt + 1);
        continue;
      }
      if (end_of_input) {
        hold_.clear();  // length-cap fragment: stripped
        if (awaiting_name_) {
          name_buf_.clear();
          awaiting_name_ = false;
        }
        continue;
      }
      if (hold_.size() > 64 ||
          (bad != std::string::npos && bad < 64 && gt == std::string::npos)) {
        if (awaiting_name_) {
          route_text(name_buf_);
          name_buf_.clear();
          awaiting_name_ = false;
        }
        route_text(hold_.substr(0, 1));
        hold_.erase(0, 1);
        continue;
      }
      return;  // split filler: wait for more
    }
    // Prefix hold-back: a strict marker/stop prefix waits for more input
    // (dropped at end_of_input as a length-cap fragment).
    {
      bool prefix = false;
      for (size_t m = 0; m < kNumFixed && !prefix; ++m) {
        const size_t n = std::strlen(kFixed[m]);
        if (n > hold_.size() &&
            std::string(kFixed[m]).compare(0, hold_.size(), hold_) == 0)
          prefix = true;
      }
      if (hold_.size() < 8 &&
          std::string("<|unused").compare(0, hold_.size(), hold_) == 0)
        prefix = true;
      for (const auto& s : stops_) {
        if (s.size() > hold_.size() && s.compare(0, hold_.size(), hold_) == 0)
          prefix = true;
      }
      if (prefix) {
        if (end_of_input) {
          hold_.clear();  // length-cap fragment: stripped
          continue;
        }
        return;
      }
    }
    // Not a marker: literal text. A pending NAME resolves first (the '<'
    // was not a content opener, so the NAME was display text).
    if (awaiting_name_) {
      route_text(name_buf_);
      name_buf_.clear();
      awaiting_name_ = false;
    }
    if (hold_.size() > kHoldMax) {
      // Unbounded non-marker '<' run: stream ahead instead of stalling.
      route_text(hold_.substr(0, 1));
      hold_.erase(0, 1);
      continue;
    }
    route_text(hold_.substr(0, 1));
    hold_.erase(0, 1);
  }
}

inline StreamDeltas ToolCallStreamParser::feed(const std::string& raw_piece) {
  hold_ += raw_piece;
  drive(false);
  return emit_deltas();
}

inline std::string ToolCallStreamParser::push(const std::string& raw_piece) {
  hold_ += raw_piece;
  drive(false);
  release_display();
  release_reasoning();
  std::string out;
  if (rel_d_ > emitted_d_) {
    out = acc_d_.substr(emitted_d_, rel_d_ - emitted_d_);
    emitted_d_ = rel_d_;
  }
  if (rel_r_ > emitted_r_) {
    stash_r_ += acc_r_.substr(emitted_r_, rel_r_ - emitted_r_);
    emitted_r_ = rel_r_;
  }
  return out;
}

inline std::string ToolCallStreamParser::poll_reasoning() {
  std::string out = stash_r_;
  stash_r_.clear();
  return out;
}

inline ToolParseResult ToolCallStreamParser::finish() {
  if (!finished_) {
    drive(true);
    if (awaiting_name_) {
      name_buf_.clear();
      awaiting_name_ = false;
    }
    if (cur_ == Channel::kInvoke) close_invoke();
    cur_ = Channel::kOutside;
    if (!stopped_) {
      // Length-cap strip: longest trailing strict marker/stop prefix goes.
      static const char* kStripFixed[] = {
          "<|message_model|>",   "<|message_user|>",   "<|message_system|>",
          "<|message_tool|>",    "<|content_thinking|>", "<|content_text|>",
          "<|content_tool_error|>", "<|content_invoke_tool_text|>",
          "<|content_invoke_tool_json|>", "<|content_xml|>",
          "<|content_image|>",   "<|content_model_end_sampling|>",
          "<|begin_of_text|>",   "<|end_message|>",     "<|endoftext|>",
          "<|unused",            "<|content_"};
      std::vector<std::string> dtoks(kStripFixed, kStripFixed + 17);
      dtoks.insert(dtoks.end(), stops_.begin(), stops_.end());
      std::vector<std::string> rtoks(kStripFixed, kStripFixed + 17);
      const size_t dd =
          Impl::inkling_suffix_overlap(acc_d_, dtoks, max_stop_len_ + 64);
      if (dd > 0) acc_d_.resize(acc_d_.size() - dd);
      const size_t rd = Impl::inkling_suffix_overlap(acc_r_, rtoks, 64);
      if (rd > 0) acc_r_.resize(acc_r_.size() - rd);
      release_display();
      release_reasoning();
    }
    rel_d_ = acc_d_.size();
    rel_r_ = acc_r_.size();
    ToolParseResult t;
    t.text = acc_d_.substr(emitted_d_);
    t.reasoning = acc_r_.substr(emitted_r_);
    t.calls = calls_;
    emitted_d_ = rel_d_;
    emitted_r_ = rel_r_;
    tail_ = t;
    finished_ = true;
  }
  return tail_;
}

// Whole-text parse: shares the incremental machine (single feed + finish),
// so streaming deltas agree with it by construction.
inline ToolParseResult parse_tool_calls(
    const std::string& raw, const std::vector<std::string>& stops = {}) {
  ToolCallStreamParser p(stops);
  StreamDeltas d = p.feed(raw);
  ToolParseResult t = p.finish();
  t.text = d.display + t.text;
  t.reasoning = d.reasoning + t.reasoning;
  return t;
}

}  // namespace Inkling
}  // namespace KokkosDNN

#endif
