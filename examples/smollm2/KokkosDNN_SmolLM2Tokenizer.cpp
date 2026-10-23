// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#include "KokkosDNN_SmolLM2Tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace KokkosDNN {
namespace SmolLM2 {
namespace {

// ---------------------------------------------------------------------------
// File + JSON helpers (schema-targeted, no third-party JSON dependency).
// ---------------------------------------------------------------------------

std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer: cannot open " + p);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

size_t skip_ws(const std::string& s, size_t p) {
  while (p < s.size() &&
         (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r'))
    ++p;
  return p;
}

void utf8_encode_cp(uint32_t cp, std::string& out) {
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
}

// Decode a JSON string; s[q] must be the opening quote. Returns the decoded
// value and advances q past the closing quote. Handles all standard escapes
// including \uXXXX with UTF-16 surrogate pairs.
std::string json_parse_string(const std::string& s, size_t& q) {
  ++q;  // opening quote
  std::string out;
  while (q < s.size()) {
    char c = s[q++];
    if (c == '"') break;
    if (c != '\\') {
      out += c;
      continue;
    }
    if (q >= s.size()) break;
    char e = s[q++];
    switch (e) {
      case 'n': out += '\n'; break;
      case 't': out += '\t'; break;
      case 'r': out += '\r'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case '/': out += '/'; break;
      case '"': out += '"'; break;
      case '\\': out += '\\'; break;
      case 'u': {
        if (q + 4 > s.size())
          throw std::runtime_error("tokenizer: bad \\u escape");
        uint32_t cp = std::stoul(s.substr(q, 4), nullptr, 16);
        q += 4;
        // surrogate pair?
        if (cp >= 0xD800 && cp <= 0xDBFF && q + 6 <= s.size() &&
            s[q] == '\\' && s[q + 1] == 'u') {
          uint32_t lo = std::stoul(s.substr(q + 2, 4), nullptr, 16);
          if (lo >= 0xDC00 && lo <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            q += 6;
          }
        }
        utf8_encode_cp(cp, out);
        break;
      }
      default: out += e; break;
    }
  }
  return out;
}

// Find `"key"` used as an object key at/after `from`: returns the position
// just past the ':' or npos. A match only counts when followed by ':'.
size_t find_key_value(const std::string& js, const char* key, size_t from) {
  std::string pat = std::string("\"") + key + "\"";
  size_t p = from;
  while ((p = js.find(pat, p)) != std::string::npos) {
    size_t c = skip_ws(js, p + pat.size());
    if (c < js.size() && js[c] == ':') return c + 1;
    p += pat.size();
  }
  return std::string::npos;
}

// Given the position of an opening bracket, find the matching closer,
// respecting JSON strings. Returns npos on unbalanced input.
size_t match_bracket(const std::string& js, size_t open, char oc, char cc) {
  bool in_str = false, esc = false;
  int depth = 0;
  for (size_t i = open; i < js.size(); ++i) {
    char c = js[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
    } else {
      if (c == '"') in_str = true;
      else if (c == oc) ++depth;
      else if (c == cc && --depth == 0) return i;
    }
  }
  return std::string::npos;
}

// Extract the body (without outer braces) of the object value of `key`
// searched from `from`. Throws when absent/unbalanced.
std::string object_body(const std::string& js, const char* key, size_t from,
                        const char* what) {
  size_t v = find_key_value(js, key, from);
  if (v == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: no ") + what);
  v = skip_ws(js, v);
  if (v >= js.size() || js[v] != '{')
    throw std::runtime_error(std::string("tokenizer: ") + what +
                             " is not an object");
  size_t close = match_bracket(js, v, '{', '}');
  if (close == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: unbalanced ") + what);
  return js.substr(v + 1, close - v - 1);
}

// Same for an array value.
std::string array_body(const std::string& js, const char* key, size_t from,
                       const char* what) {
  size_t v = find_key_value(js, key, from);
  if (v == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: no ") + what);
  v = skip_ws(js, v);
  if (v >= js.size() || js[v] != '[')
    throw std::runtime_error(std::string("tokenizer: ") + what +
                             " is not an array");
  size_t close = match_bracket(js, v, '[', ']');
  if (close == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: unbalanced ") + what);
  return js.substr(v + 1, close - v - 1);
}

long parse_number(const std::string& s, size_t& p) {
  p = skip_ws(s, p);
  size_t e = p;
  if (e < s.size() && (s[e] == '-' || s[e] == '+')) ++e;
  while (e < s.size() && std::isdigit((unsigned char)s[e])) ++e;
  if (e == p) throw std::runtime_error("tokenizer: expected number");
  long v = std::stol(s.substr(p, e - p));
  p = e;
  return v;
}

// ---------------------------------------------------------------------------
// UTF-8 <-> codepoints.
// ---------------------------------------------------------------------------

// Lenient UTF-8 decode: invalid sequences become U+FFFD.
std::vector<uint32_t> utf8_decode(const std::string& s) {
  std::vector<uint32_t> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    unsigned char c = s[i];
    uint32_t cp = 0xFFFD;
    size_t len = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c >> 5) == 0x6 && i + 1 < n) {
      unsigned char c1 = s[i + 1];
      if ((c1 & 0xC0) == 0x80) {
        cp = ((c & 0x1F) << 6) | (c1 & 0x3F);
        len = 2;
        if (cp < 0x80) cp = 0xFFFD, len = 1;
      }
    } else if ((c >> 4) == 0xE && i + 2 < n) {
      unsigned char c1 = s[i + 1], c2 = s[i + 2];
      if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80) {
        cp = ((c & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        len = 3;
        if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD, len = 1;
      }
    } else if ((c >> 3) == 0x1E && i + 3 < n) {
      unsigned char c1 = s[i + 1], c2 = s[i + 2], c3 = s[i + 3];
      if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80 &&
          (c3 & 0xC0) == 0x80) {
        cp = ((c & 0x07) << 18) | ((c1 & 0x3F) << 12) |
             ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        len = 4;
        if (cp < 0x10000 || cp > 0x10FFFF) cp = 0xFFFD, len = 1;
      }
    }
    out.push_back(cp);
    i += len;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Unicode classes for the GPT-2 ByteLevel regex.
// \p{L} / \p{N} come from generated General_Category tables below;
// \s follows the Unicode White_Space property.
// ---------------------------------------------------------------------------

// Unicode General_Category L* (letters), UCD 15.1.0.
static const uint32_t kLetterRanges[][2] = {
  {65,90},{97,122},{170,170},{181,181},{186,186},{192,214},
  {216,246},{248,705},{710,721},{736,740},{748,748},{750,750},
  {880,884},{886,887},{890,893},{895,895},{902,902},{904,906},
  {908,908},{910,929},{931,1013},{1015,1153},{1162,1327},{1329,1366},
  {1369,1369},{1376,1416},{1488,1514},{1519,1522},{1568,1610},{1646,1647},
  {1649,1747},{1749,1749},{1765,1766},{1774,1775},{1786,1788},{1791,1791},
  {1808,1808},{1810,1839},{1869,1957},{1969,1969},{1994,2026},{2036,2037},
  {2042,2042},{2048,2069},{2074,2074},{2084,2084},{2088,2088},{2112,2136},
  {2144,2154},{2160,2183},{2185,2190},{2208,2249},{2308,2361},{2365,2365},
  {2384,2384},{2392,2401},{2417,2432},{2437,2444},{2447,2448},{2451,2472},
  {2474,2480},{2482,2482},{2486,2489},{2493,2493},{2510,2510},{2524,2525},
  {2527,2529},{2544,2545},{2556,2556},{2565,2570},{2575,2576},{2579,2600},
  {2602,2608},{2610,2611},{2613,2614},{2616,2617},{2649,2652},{2654,2654},
  {2674,2676},{2693,2701},{2703,2705},{2707,2728},{2730,2736},{2738,2739},
  {2741,2745},{2749,2749},{2768,2768},{2784,2785},{2809,2809},{2821,2828},
  {2831,2832},{2835,2856},{2858,2864},{2866,2867},{2869,2873},{2877,2877},
  {2908,2909},{2911,2913},{2929,2929},{2947,2947},{2949,2954},{2958,2960},
  {2962,2965},{2969,2970},{2972,2972},{2974,2975},{2979,2980},{2984,2986},
  {2990,3001},{3024,3024},{3077,3084},{3086,3088},{3090,3112},{3114,3129},
  {3133,3133},{3160,3162},{3165,3165},{3168,3169},{3200,3200},{3205,3212},
  {3214,3216},{3218,3240},{3242,3251},{3253,3257},{3261,3261},{3293,3294},
  {3296,3297},{3313,3314},{3332,3340},{3342,3344},{3346,3386},{3389,3389},
  {3406,3406},{3412,3414},{3423,3425},{3450,3455},{3461,3478},{3482,3505},
  {3507,3515},{3517,3517},{3520,3526},{3585,3632},{3634,3635},{3648,3654},
  {3713,3714},{3716,3716},{3718,3722},{3724,3747},{3749,3749},{3751,3760},
  {3762,3763},{3773,3773},{3776,3780},{3782,3782},{3804,3807},{3840,3840},
  {3904,3911},{3913,3948},{3976,3980},{4096,4138},{4159,4159},{4176,4181},
  {4186,4189},{4193,4193},{4197,4198},{4206,4208},{4213,4225},{4238,4238},
  {4256,4293},{4295,4295},{4301,4301},{4304,4346},{4348,4680},{4682,4685},
  {4688,4694},{4696,4696},{4698,4701},{4704,4744},{4746,4749},{4752,4784},
  {4786,4789},{4792,4798},{4800,4800},{4802,4805},{4808,4822},{4824,4880},
  {4882,4885},{4888,4954},{4992,5007},{5024,5109},{5112,5117},{5121,5740},
  {5743,5759},{5761,5786},{5792,5866},{5873,5880},{5888,5905},{5919,5937},
  {5952,5969},{5984,5996},{5998,6000},{6016,6067},{6103,6103},{6108,6108},
  {6176,6264},{6272,6276},{6279,6312},{6314,6314},{6320,6389},{6400,6430},
  {6480,6509},{6512,6516},{6528,6571},{6576,6601},{6656,6678},{6688,6740},
  {6823,6823},{6917,6963},{6981,6988},{7043,7072},{7086,7087},{7098,7141},
  {7168,7203},{7245,7247},{7258,7293},{7296,7304},{7312,7354},{7357,7359},
  {7401,7404},{7406,7411},{7413,7414},{7418,7418},{7424,7615},{7680,7957},
  {7960,7965},{7968,8005},{8008,8013},{8016,8023},{8025,8025},{8027,8027},
  {8029,8029},{8031,8061},{8064,8116},{8118,8124},{8126,8126},{8130,8132},
  {8134,8140},{8144,8147},{8150,8155},{8160,8172},{8178,8180},{8182,8188},
  {8305,8305},{8319,8319},{8336,8348},{8450,8450},{8455,8455},{8458,8467},
  {8469,8469},{8473,8477},{8484,8484},{8486,8486},{8488,8488},{8490,8493},
  {8495,8505},{8508,8511},{8517,8521},{8526,8526},{8579,8580},{11264,11492},
  {11499,11502},{11506,11507},{11520,11557},{11559,11559},{11565,11565},{11568,11623},
  {11631,11631},{11648,11670},{11680,11686},{11688,11694},{11696,11702},{11704,11710},
  {11712,11718},{11720,11726},{11728,11734},{11736,11742},{11823,11823},{12293,12294},
  {12337,12341},{12347,12348},{12353,12438},{12445,12447},{12449,12538},{12540,12543},
  {12549,12591},{12593,12686},{12704,12735},{12784,12799},{13312,19903},{19968,42124},
  {42192,42237},{42240,42508},{42512,42527},{42538,42539},{42560,42606},{42623,42653},
  {42656,42725},{42775,42783},{42786,42888},{42891,42954},{42960,42961},{42963,42963},
  {42965,42969},{42994,43009},{43011,43013},{43015,43018},{43020,43042},{43072,43123},
  {43138,43187},{43250,43255},{43259,43259},{43261,43262},{43274,43301},{43312,43334},
  {43360,43388},{43396,43442},{43471,43471},{43488,43492},{43494,43503},{43514,43518},
  {43520,43560},{43584,43586},{43588,43595},{43616,43638},{43642,43642},{43646,43695},
  {43697,43697},{43701,43702},{43705,43709},{43712,43712},{43714,43714},{43739,43741},
  {43744,43754},{43762,43764},{43777,43782},{43785,43790},{43793,43798},{43808,43814},
  {43816,43822},{43824,43866},{43868,43881},{43888,44002},{44032,55203},{55216,55238},
  {55243,55291},{63744,64109},{64112,64217},{64256,64262},{64275,64279},{64285,64285},
  {64287,64296},{64298,64310},{64312,64316},{64318,64318},{64320,64321},{64323,64324},
  {64326,64433},{64467,64829},{64848,64911},{64914,64967},{65008,65019},{65136,65140},
  {65142,65276},{65313,65338},{65345,65370},{65382,65470},{65474,65479},{65482,65487},
  {65490,65495},{65498,65500},{65536,65547},{65549,65574},{65576,65594},{65596,65597},
  {65599,65613},{65616,65629},{65664,65786},{66176,66204},{66208,66256},{66304,66335},
  {66349,66368},{66370,66377},{66384,66421},{66432,66461},{66464,66499},{66504,66511},
  {66560,66717},{66736,66771},{66776,66811},{66816,66855},{66864,66915},{66928,66938},
  {66940,66954},{66956,66962},{66964,66965},{66967,66977},{66979,66993},{66995,67001},
  {67003,67004},{67072,67382},{67392,67413},{67424,67431},{67456,67461},{67463,67504},
  {67506,67514},{67584,67589},{67592,67592},{67594,67637},{67639,67640},{67644,67644},
  {67647,67669},{67680,67702},{67712,67742},{67808,67826},{67828,67829},{67840,67861},
  {67872,67897},{67968,68023},{68030,68031},{68096,68096},{68112,68115},{68117,68119},
  {68121,68149},{68192,68220},{68224,68252},{68288,68295},{68297,68324},{68352,68405},
  {68416,68437},{68448,68466},{68480,68497},{68608,68680},{68736,68786},{68800,68850},
  {68864,68899},{69248,69289},{69296,69297},{69376,69404},{69415,69415},{69424,69445},
  {69488,69505},{69552,69572},{69600,69622},{69635,69687},{69745,69746},{69749,69749},
  {69763,69807},{69840,69864},{69891,69926},{69956,69956},{69959,69959},{69968,70002},
  {70006,70006},{70019,70066},{70081,70084},{70106,70106},{70108,70108},{70144,70161},
  {70163,70187},{70207,70208},{70272,70278},{70280,70280},{70282,70285},{70287,70301},
  {70303,70312},{70320,70366},{70405,70412},{70415,70416},{70419,70440},{70442,70448},
  {70450,70451},{70453,70457},{70461,70461},{70480,70480},{70493,70497},{70656,70708},
  {70727,70730},{70751,70753},{70784,70831},{70852,70853},{70855,70855},{71040,71086},
  {71128,71131},{71168,71215},{71236,71236},{71296,71338},{71352,71352},{71424,71450},
  {71488,71494},{71680,71723},{71840,71903},{71935,71942},{71945,71945},{71948,71955},
  {71957,71958},{71960,71983},{71999,71999},{72001,72001},{72096,72103},{72106,72144},
  {72161,72161},{72163,72163},{72192,72192},{72203,72242},{72250,72250},{72272,72272},
  {72284,72329},{72349,72349},{72368,72440},{72704,72712},{72714,72750},{72768,72768},
  {72818,72847},{72960,72966},{72968,72969},{72971,73008},{73030,73030},{73056,73061},
  {73063,73064},{73066,73097},{73112,73112},{73440,73458},{73474,73474},{73476,73488},
  {73490,73523},{73648,73648},{73728,74649},{74880,75075},{77712,77808},{77824,78895},
  {78913,78918},{82944,83526},{92160,92728},{92736,92766},{92784,92862},{92880,92909},
  {92928,92975},{92992,92995},{93027,93047},{93053,93071},{93760,93823},{93952,94026},
  {94032,94032},{94099,94111},{94176,94177},{94179,94179},{94208,100343},{100352,101589},
  {101632,101640},{110576,110579},{110581,110587},{110589,110590},{110592,110882},{110898,110898},
  {110928,110930},{110933,110933},{110948,110951},{110960,111355},{113664,113770},{113776,113788},
  {113792,113800},{113808,113817},{119808,119892},{119894,119964},{119966,119967},{119970,119970},
  {119973,119974},{119977,119980},{119982,119993},{119995,119995},{119997,120003},{120005,120069},
  {120071,120074},{120077,120084},{120086,120092},{120094,120121},{120123,120126},{120128,120132},
  {120134,120134},{120138,120144},{120146,120485},{120488,120512},{120514,120538},{120540,120570},
  {120572,120596},{120598,120628},{120630,120654},{120656,120686},{120688,120712},{120714,120744},
  {120746,120770},{120772,120779},{122624,122654},{122661,122666},{122928,122989},{123136,123180},
  {123191,123197},{123214,123214},{123536,123565},{123584,123627},{124112,124139},{124896,124902},
  {124904,124907},{124909,124910},{124912,124926},{124928,125124},{125184,125251},{125259,125259},
  {126464,126467},{126469,126495},{126497,126498},{126500,126500},{126503,126503},{126505,126514},
  {126516,126519},{126521,126521},{126523,126523},{126530,126530},{126535,126535},{126537,126537},
  {126539,126539},{126541,126543},{126545,126546},{126548,126548},{126551,126551},{126553,126553},
  {126555,126555},{126557,126557},{126559,126559},{126561,126562},{126564,126564},{126567,126570},
  {126572,126578},{126580,126583},{126585,126588},{126590,126590},{126592,126601},{126603,126619},
  {126625,126627},{126629,126633},{126635,126651},{131072,173791},{173824,177977},{177984,178205},
  {178208,183969},{183984,191456},{191472,192093},{194560,195101},{196608,201546},{201552,205743},
};
// Unicode General_Category N* (numbers), UCD 15.1.0.
static const uint32_t kNumberRanges[][2] = {
  {48,57},{178,179},{185,185},{188,190},{1632,1641},{1776,1785},
  {1984,1993},{2406,2415},{2534,2543},{2548,2553},{2662,2671},{2790,2799},
  {2918,2927},{2930,2935},{3046,3058},{3174,3183},{3192,3198},{3302,3311},
  {3416,3422},{3430,3448},{3558,3567},{3664,3673},{3792,3801},{3872,3891},
  {4160,4169},{4240,4249},{4969,4988},{5870,5872},{6112,6121},{6128,6137},
  {6160,6169},{6470,6479},{6608,6618},{6784,6793},{6800,6809},{6992,7001},
  {7088,7097},{7232,7241},{7248,7257},{8304,8304},{8308,8313},{8320,8329},
  {8528,8578},{8581,8585},{9312,9371},{9450,9471},{10102,10131},{11517,11517},
  {12295,12295},{12321,12329},{12344,12346},{12690,12693},{12832,12841},{12872,12879},
  {12881,12895},{12928,12937},{12977,12991},{42528,42537},{42726,42735},{43056,43061},
  {43216,43225},{43264,43273},{43472,43481},{43504,43513},{43600,43609},{44016,44025},
  {65296,65305},{65799,65843},{65856,65912},{65930,65931},{66273,66299},{66336,66339},
  {66369,66369},{66378,66378},{66513,66517},{66720,66729},{67672,67679},{67705,67711},
  {67751,67759},{67835,67839},{67862,67867},{68028,68029},{68032,68047},{68050,68095},
  {68160,68168},{68221,68222},{68253,68255},{68331,68335},{68440,68447},{68472,68479},
  {68521,68527},{68858,68863},{68912,68921},{69216,69246},{69405,69414},{69457,69460},
  {69573,69579},{69714,69743},{69872,69881},{69942,69951},{70096,70105},{70113,70132},
  {70384,70393},{70736,70745},{70864,70873},{71248,71257},{71360,71369},{71472,71483},
  {71904,71922},{72016,72025},{72784,72812},{73040,73049},{73120,73129},{73552,73561},
  {73664,73684},{74752,74862},{92768,92777},{92864,92873},{93008,93017},{93019,93025},
  {93824,93846},{119488,119507},{119520,119539},{119648,119672},{120782,120831},{123200,123209},
  {123632,123641},{124144,124153},{125127,125135},{125264,125273},{126065,126123},{126125,126127},
  {126129,126132},{126209,126253},{126255,126269},{127232,127244},{130032,130041},
};

bool in_ranges(const uint32_t (*t)[2], size_t n, uint32_t cp) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (cp < t[mid][0]) hi = mid;
    else if (cp > t[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}

bool is_letter(uint32_t cp) {
  return in_ranges(kLetterRanges, sizeof(kLetterRanges) / sizeof(kLetterRanges[0]), cp);
}
bool is_number(uint32_t cp) {
  return in_ranges(kNumberRanges, sizeof(kNumberRanges) / sizeof(kNumberRanges[0]), cp);
}

bool is_space(uint32_t cp) {
  // Unicode White_Space property (15.1): 0009-000D, 0020, 0085, 00A0,
  // 1680, 2000-200A, 2028, 2029, 202F, 205F, 3000.
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 ||
         cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
         cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
         cp == 0x3000;
}

// ---------------------------------------------------------------------------
// GPT-2 ByteLevel pre-tokenizer (use_regex=true) + Digits splitting.
//
// SmolLM2's pre_tokenizer is Sequence[Digits(individual_digits=true),
// ByteLevel(add_prefix_space=false, use_regex=true)]. HF applies them in
// order: Digits first isolates every ASCII digit into its own piece
// (boundaries before and after each digit; all other bytes, spaces
// included, stay with the neighboring non-digit run), then ByteLevel
// applies the GPT-2 regex to each piece:
//
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|
//    \s+(?!\S)|\s+
//
// Contractions are case-SENSITIVE, the optional prefix on letter/number/
// symbol runs is a single literal U+0020 space (not \s), numbers group
// greedily (\p{N}+, although Digits has already isolated ASCII digits by
// the time ByteLevel runs), and there is no special CR/LF handling
// (\r,\n are plain \s). Leftmost-first alternation. Non-ASCII numbers
// are NOT pre-isolated (HF Digits keys on ASCII digits); they fall
// through to the ` ?\p{N}+` alternative.
// ---------------------------------------------------------------------------

// Alternative 1 (case-sensitive): 's|'t|'re|'ve|'m|'ll|'d.
size_t match_contraction_gpt2(const std::vector<uint32_t>& cp, size_t i) {
  size_t n = cp.size();
  if (i >= n || cp[i] != '\'') return 0;
  auto at = [&](size_t k) -> uint32_t {
    return i + k < n ? cp[i + k] : 0;
  };
  if (at(1) == 's' || at(1) == 't' || at(1) == 'm' || at(1) == 'd') return 2;
  if ((at(1) == 'r' && at(2) == 'e') || (at(1) == 'v' && at(2) == 'e') ||
      (at(1) == 'l' && at(2) == 'l'))
    return 3;
  return 0;
}

// One GPT-2 regex match at codepoint position i; length in codepoints
// (always >= 1, so scanning terminates).
size_t gpt2_match_at(const std::vector<uint32_t>& cp, size_t i) {
  const size_t n = cp.size();
  size_t len = 0;
  if ((len = match_contraction_gpt2(cp, i)) != 0) return len;
  // Alts 2-4 share one optional literal-' ' prefix.
  const size_t j = (cp[i] == ' ') ? i + 1 : i;
  if (j < n && is_letter(cp[j])) {
    size_t k = j;
    while (k < n && is_letter(cp[k])) ++k;
    return k - i;
  }
  if (j < n && is_number(cp[j])) {
    size_t k = j;
    while (k < n && is_number(cp[k])) ++k;
    return k - i;
  }
  if (j < n && !is_space(cp[j]) && !is_letter(cp[j]) &&
      !is_number(cp[j])) {
    size_t k = j;
    while (k < n && !is_space(cp[k]) && !is_letter(cp[k]) &&
           !is_number(cp[k]))
      ++k;
    return k - i;
  }
  if (is_space(cp[i])) {
    // Alts 5/6: \s+(?!\S)|\s+. A maximal ws run matches fully at end of
    // input, otherwise all but one char (the (?!\S) lookahead backtracks
    // so the next match may start with a literal-' ' prefix or \s+).
    size_t r = i;
    while (r < n && is_space(cp[r])) ++r;
    if (r >= n) return r - i;
    return (r - i >= 2) ? r - i - 1 : r - i;
  }
  return 1;  // unreachable for well-formed input; guarantees termination
}

// HF Digits(individual_digits=true): cut before and after every ASCII
// digit. Digits 0-9 are single bytes that never occur inside multibyte
// UTF-8 sequences, so a byte scan is exact. Returns byte-offset segments.
std::vector<std::string> split_ascii_digits(const std::string& text) {
  std::vector<std::string> segs;
  size_t start = 0;
  for (size_t b = 0; b < text.size(); ++b) {
    if (text[b] >= '0' && text[b] <= '9') {
      if (b > start) segs.push_back(text.substr(start, b - start));
      segs.push_back(text.substr(b, 1));
      start = b + 1;
    }
  }
  if (text.size() > start) segs.push_back(text.substr(start));
  return segs;
}

std::vector<std::string> split_pretokens_gpt2(const std::string& text) {
  std::vector<std::string> pieces;
  for (const std::string& seg : split_ascii_digits(text)) {
    std::vector<uint32_t> cp = utf8_decode(seg);
    std::vector<size_t> off;
    off.reserve(cp.size() + 1);
    for (size_t b = 0, k = 0; k < cp.size(); ++k) {
      off.push_back(b);
      uint32_t c = cp[k];
      b += (c < 0x80) ? 1 : (c < 0x800) ? 2 : (c < 0x10000) ? 3 : 4;
    }
    off.push_back(seg.size());
    size_t i = 0;
    while (i < cp.size()) {
      size_t len = gpt2_match_at(cp, i);
      pieces.push_back(seg.substr(off[i], off[i + len] - off[i]));
      i += len;
    }
  }
  return pieces;
}

// ---------------------------------------------------------------------------
// GPT-2 style byte mapping: every byte <-> one unicode char.
// ---------------------------------------------------------------------------

struct ByteMaps {
  std::string enc[256];  // byte -> UTF-8 of mapped char
  std::unordered_map<std::string, unsigned char> dec;  // UTF-8 -> byte
  ByteMaps() {
    // bs = printable ASCII + Latin-1 supplement ranges, then the rest.
    bool in_bs[256] = {};
    for (int b = 0x21; b <= 0x7E; ++b) in_bs[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) in_bs[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) in_bs[b] = true;
    std::vector<int> bs, cs;
    for (int b = 0; b < 256; ++b)
      if (in_bs[b]) {
        bs.push_back(b);
        cs.push_back(b);
      }
    int m = 0;
    for (int b = 0; b < 256; ++b)
      if (!in_bs[b]) {
        bs.push_back(b);
        cs.push_back(0x100 + m++);
      }
    for (size_t k = 0; k < bs.size(); ++k) {
      utf8_encode_cp((uint32_t)cs[k], enc[bs[k]]);
      dec[enc[bs[k]]] = (unsigned char)bs[k];
    }
  }
};

const ByteMaps& byte_maps() {
  static ByteMaps m;
  return m;
}

std::string bytes_to_unicode(const std::string& bytes) {
  const ByteMaps& m = byte_maps();
  std::string out;
  for (unsigned char b : bytes) out += m.enc[b];
  return out;
}

// Inverse: mapped unicode chars -> original bytes. Chars outside the table
// (should not occur) pass through as UTF-8.
std::string unicode_to_bytes(const std::string& s) {
  const ByteMaps& m = byte_maps();
  std::string out;
  std::vector<uint32_t> cp = utf8_decode(s);
  for (uint32_t c : cp) {
    std::string u;
    utf8_encode_cp(c, u);
    auto it = m.dec.find(u);
    if (it != m.dec.end()) {
      out += (char)it->second;
    } else {
      out += u;  // lenient
    }
  }
  return out;
}

}  // namespace

void Tokenizer::load(const std::string& path) {
  path_ = path;
  std::string js = read_file(path);

  token_to_id_.clear();
  merge_rank_.clear();
  added_tokens_.clear();
  special_ids_.clear();
  special_content_.clear();

  // model.{vocab,merges}: scope the search inside "model" so same-named
  // keys elsewhere (there are none, but be strict) cannot interfere.
  size_t model_pos = find_key_value(js, "model", 0);
  if (model_pos == std::string::npos)
    throw std::runtime_error("tokenizer: no model object");
  model_pos = skip_ws(js, model_pos);
  if (model_pos >= js.size() || js[model_pos] != '{')
    throw std::runtime_error("tokenizer: model is not an object");
  size_t model_close = match_bracket(js, model_pos, '{', '}');
  if (model_close == std::string::npos)
    throw std::runtime_error("tokenizer: unbalanced model object");

  std::string vocab = object_body(js, "vocab", model_pos, "model.vocab");
  {
    std::unordered_map<int, std::string> idmap;
    int max_id = -1;
    size_t i = 0;
    while (i < vocab.size()) {
      size_t q = vocab.find('"', i);
      if (q == std::string::npos) break;
      std::string tok = json_parse_string(vocab, q);
      size_t colon = vocab.find(':', q);
      if (colon == std::string::npos) break;
      size_t p = colon + 1;
      long id = parse_number(vocab, p);
      token_to_id_[tok] = (int)id;
      idmap[(int)id] = tok;
      if (id > max_id) max_id = (int)id;
      i = p;
    }
    if (max_id < 0) throw std::runtime_error("tokenizer: empty vocab");
    vocab_size_ = max_id + 1;
    id_to_token_.assign(vocab_size_, "");
    for (auto& kv : idmap) id_to_token_[kv.first] = kv.second;
  }

  std::string merges = array_body(js, "merges", model_pos, "model.merges");
  {
    size_t j = 0;
    int rank = 0;
    while (j < merges.size()) {
      size_t q = merges.find('"', j);
      if (q == std::string::npos) break;
      std::string m = json_parse_string(merges, q);
      merge_rank_[m] = rank++;
      j = q;
    }
  }

  // added_tokens: [{id, content, special, ...}] in any field order.
  {
    std::string added = array_body(js, "added_tokens", 0, "added_tokens");
    size_t j = 0;
    while (j < added.size()) {
      size_t o = added.find('{', j);
      if (o == std::string::npos) break;
      size_t c = match_bracket(added, o, '{', '}');
      if (c == std::string::npos)
        throw std::runtime_error("tokenizer: unbalanced added_token");
      std::string obj = added.substr(o, c - o + 1);
      size_t pv = find_key_value(obj, "id", 0);
      size_t cv = find_key_value(obj, "content", 0);
      if (pv == std::string::npos || cv == std::string::npos) {
        j = c + 1;
        continue;
      }
      size_t p = pv;
      long id = parse_number(obj, p);
      size_t q = skip_ws(obj, cv);
      std::string content;
      if (q < obj.size() && obj[q] == '"')
        content = json_parse_string(obj, q);
      else {
        j = c + 1;
        continue;
      }
      bool special = true;
      size_t sv = find_key_value(obj, "special", 0);
      if (sv != std::string::npos) {
        size_t t = skip_ws(obj, sv);
        if (obj.compare(t, 4, "true") == 0) special = true;
        else if (obj.compare(t, 5, "false") == 0) special = false;
      }
      if (id >= 0 && !content.empty()) {
        // added_tokens extend the vocab (0..16 for SmolLM2).
        auto it = token_to_id_.find(content);
        if (it == token_to_id_.end()) {
          token_to_id_[content] = (int)id;
          if (id >= (int)id_to_token_.size()) id_to_token_.resize(id + 1);
          id_to_token_[id] = content;
          if (id + 1 > vocab_size_) vocab_size_ = (int)id + 1;
        }
        added_tokens_.push_back({(int)id, content, special});
        if (special) {
          special_ids_.insert((int)id);
          special_content_[(int)id] = content;
        }
      }
      j = c + 1;
    }
    // longest content first for greedy longest-match splitting
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const AddedToken& a, const AddedToken& b) {
                return a.content.size() > b.content.size();
              });
  }

  // Resolve well-known specials by content, else keep in-vocab fallbacks
  // so bos/eos/eot are always usable as model input ids (vocab-49152
  // range safety). Base-model convention bos=eos=<|endoftext|> (id 0);
  // the Instruct stop <|im_end|> becomes eot; eom aliases eos.
  auto resolve = [&](const char* content, int fallback) {
    for (auto& t : added_tokens_)
      if (t.content == content && t.special) return t.id;
    return fallback;
  };
  bos_id_ = resolve("<|endoftext|>", 0);
  eos_id_ = resolve("<|endoftext|>", 0);
  eom_id_ = resolve("<|eom_id|>", eos_id_);
  eot_id_ = resolve("<|im_end|>", eos_id_);

  // Sibling tokenizer_config.json override (best-effort, never throws):
  // content resolution alone cannot tell SmolLM2-Base (bos=eos=
  // <|endoftext|>) from SmolLM2-Instruct (bos=<|im_start|>,
  // eos=pad=<|im_end|>); the config file records that choice. Only
  // attempted when the tokenizer path contains a directory.
  {
    size_t slash = path_.find_last_of('/');
    if (slash != std::string::npos) {
      try {
        std::string cfg =
            read_file(path_.substr(0, slash + 1) + "tokenizer_config.json");
        auto override_from_config = [&](const char* key, int& target) {
          size_t kv = find_key_value(cfg, key, 0);
          if (kv == std::string::npos) return;
          size_t q = skip_ws(cfg, kv);
          if (q >= cfg.size() || cfg[q] != '"') return;  // null/non-string
          std::string tok = json_parse_string(cfg, q);
          auto it = token_to_id_.find(tok);
          if (it != token_to_id_.end()) target = it->second;
        };
        override_from_config("bos_token", bos_id_);
        override_from_config("eos_token", eos_id_);
      } catch (...) {
      }
    }
  }
}

void Tokenizer::bpe_piece(const std::string& piece,
                          std::vector<int>& out) const {
  if (piece.empty()) return;
  // split into single unicode chars (UTF-8 aware)
  std::vector<std::string> parts;
  for (size_t i = 0; i < piece.size();) {
    unsigned char c = piece[i];
    size_t len = 1;
    if (c >= 0x80) {
      if ((c >> 5) == 0x6) len = 2;
      else if ((c >> 4) == 0xE) len = 3;
      else if ((c >> 3) == 0x1E) len = 4;
    }
    parts.push_back(piece.substr(i, len));
    i += len;
  }
  // greedily merge the lowest-rank adjacent pair
  while (parts.size() > 1) {
    int best_rank = -1;
    size_t best = 0;
    for (size_t k = 0; k + 1 < parts.size(); ++k) {
      auto it = merge_rank_.find(parts[k] + " " + parts[k + 1]);
      if (it != merge_rank_.end() &&
          (best_rank < 0 || it->second < best_rank)) {
        best_rank = it->second;
        best = k;
      }
    }
    if (best_rank < 0) break;
    parts[best] += parts[best + 1];
    parts.erase(parts.begin() + best + 1);
  }
  // byte-fallback encode path: with no applicable merge each single
  // byte-mapped char must resolve to a base vocab id (the 256 byte-level
  // entries); anything else is a corrupt/foreign file.
  for (auto& s : parts) {
    auto it = token_to_id_.find(s);
    if (it == token_to_id_.end())
      throw std::runtime_error("tokenizer: BPE token missing from vocab: " +
                               unicode_to_bytes(s));
    out.push_back(it->second);
  }
}

std::vector<int> Tokenizer::encode(const std::string& text,
                                   bool add_bos) const {
  std::vector<int> ids;
  if (add_bos) ids.push_back(bos_id_);
  if (text.empty() || !loaded()) return ids;

  // Split out literal special tokens first (longest match wins). The '<'
  // fast path holds for the ChatML specials (<|endoftext|>, <|im_start|>,
  // <|im_end|>) alike.
  auto special_at = [&](size_t pos) -> const AddedToken* {
    if (text[pos] != '<') return nullptr;  // all known specials start here
    for (auto& t : added_tokens_) {
      if (t.special && text.compare(pos, t.content.size(), t.content) == 0)
        return &t;
    }
    return nullptr;
  };

  std::string buf;
  auto flush = [&]() {
    if (buf.empty()) return;
    for (const auto& piece : split_pretokens_gpt2(buf)) {
      std::string mapped = bytes_to_unicode(piece);
      bpe_piece(mapped, ids);
    }
    buf.clear();
  };

  size_t i = 0;
  while (i < text.size()) {
    if (const AddedToken* t = special_at(i)) {
      flush();
      ids.push_back(t->id);
      i += t->content.size();
    } else {
      buf += text[i++];
    }
  }
  flush();
  return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids,
                              bool skip_special_tokens) const {
  std::string out, acc;
  auto flush = [&]() {
    if (!acc.empty()) {
      out += unicode_to_bytes(acc);
      acc.clear();
    }
  };
  for (int id : ids) {
    if (is_special(id)) {
      flush();
      if (!skip_special_tokens) {
        auto it = special_content_.find(id);
        if (it != special_content_.end()) out += it->second;
      }
      continue;
    }
    if (id < 0 || id >= (int)id_to_token_.size() ||
        id_to_token_[id].empty())
      throw std::runtime_error("tokenizer: invalid id in decode");
    acc += id_to_token_[id];
  }
  flush();
  return out;
}

std::vector<ChatMessage> make_chat_messages(const std::string& prompt,
                                            const std::string& system,
                                            const std::string& role) {
  std::vector<ChatMessage> msgs;
  if (!system.empty()) msgs.push_back({"system", system});
  msgs.push_back({role.empty() ? "user" : role, prompt});
  return msgs;
}

std::vector<int> ensure_bos(const std::vector<int>& ids, int bos_id,
                            bool want_bos) {
  size_t k = 0;
  while (k < ids.size() && ids[k] == bos_id) ++k;
  if (want_bos) {
    std::vector<int> out;
    out.reserve(ids.size() - k + 1);
    out.push_back(bos_id);
    out.insert(out.end(), ids.begin() + (long)k, ids.end());
    return out;
  }
  return std::vector<int>(ids.begin() + (long)k, ids.end());
}

std::vector<int> encode_rendered_chat(const Tokenizer& tok,
                                      const std::string& rendered,
                                      bool add_bos) {
  // add_bos=false here: the literal BOS in `rendered` (if any) is picked
  // up by the special-token split; ensure_bos() then enforces the
  // exactly-one (or zero) invariant on top.
  std::vector<int> body = tok.encode(rendered, false);
  return ensure_bos(body, tok.bos(), add_bos);
}

std::string render_chatml_template(const std::vector<ChatMessage>& messages,
                                   bool add_generation_prompt,
                                   const std::string& default_system) {
  std::string out;
  bool has_system = false;
  for (const auto& m : messages)
    if (m.role == "system") {
      has_system = true;
      break;
    }
  if (!has_system && !default_system.empty())
    out += "<|im_start|>system\n" + default_system + "<|im_end|>\n";
  for (const auto& m : messages) {
    out += "<|im_start|>" + m.role + "\n";
    out += m.content;
    out += "<|im_end|>\n";
  }
  if (add_generation_prompt) out += "<|im_start|>assistant\n";
  return out;
}

std::vector<int> encode_chatml(const Tokenizer& tok,
                               const std::vector<ChatMessage>& messages,
                               bool add_generation_prompt, bool add_bos) {
  return encode_rendered_chat(
      tok, render_chatml_template(messages, add_generation_prompt), add_bos);
}

size_t find_stop_string(const std::string& text,
                        const std::vector<std::string>& stop_strings,
                        std::string* matched) {
  size_t best = std::string::npos;
  const std::string* which = nullptr;
  for (const auto& s : stop_strings) {
    if (s.empty()) continue;
    size_t p = text.find(s);
    if (p != std::string::npos && (best == std::string::npos || p < best)) {
      best  = p;
      which = &s;
    }
  }
  if (matched) *matched = (which ? *which : std::string());
  return best;
}

std::string truncate_at_stop_strings(
    const std::string& text, const std::vector<std::string>& stop_strings) {
  size_t p = find_stop_string(text, stop_strings);
  if (p == std::string::npos) return text;
  return text.substr(0, p);
}

}  // namespace SmolLM2
}  // namespace KokkosDNN
