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

#include <charconv>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
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

}  // namespace Inkling
}  // namespace KokkosDNN

#endif
