// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#include "KokkosDNN_LFM25Tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace KokkosDNN {
namespace LFM25 {
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
// ByteLevel pre-tokenizer with the LFM2 regex (tokenizer.json
// pre_tokenizer = Sequence[Split(regex, Isolated), ByteLevel(no prefix)]):
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// This is NOT the GPT-2 pattern: contractions are case-INsensitive, the
// optional prefix on a letter run is any single non-CR/LF/letter/number
// character, numbers group in runs of at most three, symbol runs absorb
// trailing CR/LF, and a whitespace run that contains a newline breaks at
// the last newline (llama.cpp's LLAMA_VOCAB_PRE_TYPE_LLAMA3, which it also
// selects for `tokenizer.ggml.pre == "lfm2"`). Leftmost-first
// alternation; lengths are in codepoints.
// ---------------------------------------------------------------------------

bool is_crlf(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// Alternative 1 (case-insensitive): 's|'t|'re|'ve|'m|'ll|'d.
size_t match_contraction_lfm2(const std::vector<uint32_t>& cp, size_t i) {
  size_t n = cp.size();
  if (i >= n || cp[i] != '\'') return 0;
  auto at = [&](size_t k) -> uint32_t {
    uint32_t c = i + k < n ? cp[i + k] : 0;
    return (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
  };
  if (at(1) == 's' || at(1) == 't' || at(1) == 'm' || at(1) == 'd') return 2;
  if ((at(1) == 'r' && at(2) == 'e') || (at(1) == 'v' && at(2) == 'e') ||
      (at(1) == 'l' && at(2) == 'l'))
    return 3;
  return 0;
}

// One pre-token match at codepoint position i; length in codepoints
// (always >= 1, so scanning terminates).
size_t lfm2_match_at(const std::vector<uint32_t>& cp, size_t i) {
  const size_t n = cp.size();
  size_t len = 0;
  if ((len = match_contraction_lfm2(cp, i)) != 0) return len;

  // Alt 2: [^\r\n\p{L}\p{N}]?\p{L}+
  {
    size_t j = i;
    if (!is_crlf(cp[i]) && !is_letter(cp[i]) && !is_number(cp[i])) j = i + 1;
    if (j < n && is_letter(cp[j])) {
      size_t k = j;
      while (k < n && is_letter(cp[k])) ++k;
      return k - i;
    }
  }

  // Alt 3: \p{N}{1,3}
  if (is_number(cp[i])) {
    size_t k = i;
    while (k < n && k - i < 3 && is_number(cp[k])) ++k;
    return k - i;
  }

  // Alt 4: " ?[^\s\p{L}\p{N}]+[\r\n]*"
  {
    size_t j = (cp[i] == ' ') ? i + 1 : i;
    if (j < n && !is_space(cp[j]) && !is_letter(cp[j]) && !is_number(cp[j])) {
      size_t k = j;
      while (k < n && !is_space(cp[k]) && !is_letter(cp[k]) &&
             !is_number(cp[k]))
        ++k;
      while (k < n && is_crlf(cp[k])) ++k;
      return k - i;
    }
  }

  if (is_space(cp[i])) {
    size_t r = i;
    while (r < n && is_space(cp[r])) ++r;
    // Alt 5: \s*[\r\n]+ -- the greedy \s* backs off to the last newline of
    // the run, so the match ends right after it (consecutive newlines
    // there are part of [\r\n]+).
    size_t last_nl = 0;
    bool has_nl = false;
    for (size_t t = i; t < r; ++t)
      if (is_crlf(cp[t])) {
        last_nl = t;
        has_nl = true;
      }
    if (has_nl) return last_nl + 1 - i;
    // Alt 6/7: \s+(?!\S)|\s+. A newline-free run matches fully at end of
    // input, otherwise all but one char (the (?!\S) lookahead backtracks
    // so the next match may start with a literal-' ' prefix or \s+).
    if (r >= n) return r - i;
    return (r - i >= 2) ? r - i - 1 : r - i;
  }
  return 1;  // unreachable for well-formed input; guarantees termination
}

std::vector<std::string> split_pretokens_lfm2(const std::string& text) {
  std::vector<std::string> pieces;
  const std::string& seg = text;
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
    size_t len = lfm2_match_at(cp, i);
    pieces.push_back(seg.substr(off[i], off[i + len] - off[i]));
    i += len;
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

  // merges: two on-disk spellings share this key.
  //   legacy (transformers <= 4.x): ["a b", ...]
  //   current (transformers 5.x):   [["a", "b"], ...]
  // BPE looks pairs up as "left right", so both forms land in the same
  // table. Parsing only the string form silently drops every merge on a
  // 5.x file, which degrades encoding to pure byte fallback.
  std::string merges = array_body(js, "merges", model_pos, "model.merges");
  {
    size_t j = 0;
    int rank = 0;
    while (j < merges.size()) {
      size_t q = merges.find_first_of("\"[", j);
      if (q == std::string::npos) break;
      if (merges[q] == '[') {
        size_t k = merges.find('"', q + 1);
        if (k == std::string::npos) break;
        std::string left = json_parse_string(merges, k);
        size_t k2 = merges.find('"', k);
        if (k2 == std::string::npos) break;
        std::string right = json_parse_string(merges, k2);
        merge_rank_[left + " " + right] = rank++;
        j = k2;
      } else {
        std::string m = json_parse_string(merges, q);
        merge_rank_[m] = rank++;
        j = q;
      }
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
        // added_tokens extend the vocab (LFM2.5 specials live at the top of
        // the 128k range, e.g. 124893..124902).
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

  // Resolve well-known specials by content, else fall back to the numeric
  // LFM2.5 ids (kPadId/kBosId/... in the header) so bos/eos/pad are always
  // usable as model input ids. LFM2.5 convention: bos=<|startoftext|>
  // (124894), eos/stop=<|im_end|> (124900).
  auto resolve = [&](const char* content, int fallback) {
    for (auto& t : added_tokens_)
      if (t.content == content && t.special) return t.id;
    return fallback;
  };
  bos_id_         = resolve(kBosToken, kBosId);
  eos_id_         = resolve(kEosToken, kEosId);
  pad_id_         = resolve(kPadToken, kPadId);
  im_start_id_    = resolve(kImStartToken, kImStartId);
  think_start_id_ = resolve(kThinkStartToken, kThinkStartId);
  think_end_id_   = resolve(kThinkEndToken, kThinkEndId);

  // LFM2.5 tool-call markers are control tokens even when a tokenizer.json
  // files them as non-special (the stock file does): force them special so
  // they encode atomically to single ids and stay out of default
  // skip-specials decodes, where the tool parser consumes them
  // structurally instead.
  for (auto& tok : added_tokens_) {
    if (tok.content == kToolCallStartToken ||
        tok.content == kToolCallEndToken) {
      tok.special = true;
      special_ids_.insert(tok.id);
      special_content_[tok.id] = tok.content;
    }
  }

  // Sibling tokenizer_config.json override (best-effort, never throws):
  // the config file records the bos/eos choice when content resolution is
  // ambiguous. Only attempted when the tokenizer path contains a
  // directory.
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
  // fast path holds for the ChatML/think specials (<|startoftext|>,
  // <|im_start|>, <|im_end|>, <think>, ...) alike.
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
    for (const auto& piece : split_pretokens_lfm2(buf)) {
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
  if (!system.empty()) msgs.push_back({"system", system, "", {}, ""});
  msgs.push_back({role.empty() ? "user" : role, prompt, "", {}, ""});
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

std::string render_lfm25_chatml(const std::vector<ChatMessage>& messages,
                                bool add_generation_prompt,
                                bool preserve_thinking,
                                const std::string& default_system) {
  std::string out;
  bool has_system = false;
  for (const auto& m : messages)
    if (m.role == "system") {
      has_system = true;
      break;
    }
  if (!has_system && !default_system.empty())
    out += std::string(kImStartToken) + "system\n" + default_system +
           kEosToken + "\n";
  for (const auto& m : messages) {
    out += std::string(kImStartToken) + m.role + "\n";
    if (preserve_thinking && m.role == "assistant" && !m.thinking.empty())
      out += std::string(kThinkStartToken) + m.thinking + kThinkEndToken;
    out += m.content;
    out += std::string(kEosToken) + "\n";
  }
  if (add_generation_prompt)
    out += std::string(kImStartToken) + "assistant\n" + kThinkStartToken;
  return out;
}

std::vector<int> encode_chatml(const Tokenizer& tok,
                               const std::vector<ChatMessage>& messages,
                               bool add_generation_prompt, bool add_bos,
                               bool preserve_thinking) {
  return encode_rendered_chat(
      tok,
      render_lfm25_chatml(messages, add_generation_prompt, preserve_thinking),
      add_bos);
}

// ---------------------------------------------------------------------------
// Native tool-use support (render + parse).
// ---------------------------------------------------------------------------

namespace {

// Trim ASCII whitespace (space, \t, \n, \r, \f, \v) from both ends.
std::string tool_trim(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// Append a JSON string literal for `v` (double-quoted, minimal escapes).
void tool_append_json_string(const std::string& v, std::string& out) {
  static const char* kHex = "0123456789abcdef";
  out += '"';
  for (char c : v) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += "\\u00";
          out += kHex[(c >> 4) & 0xF];
          out += kHex[c & 0xF];
        } else {
          out += c;
        }
    }
  }
  out += '"';
}

// Split on `sep` at nesting depth 0, respecting single/double quotes
// (with backslash escapes) and (), [], {} nesting.
std::vector<std::string> tool_split_top(const std::string& s, char sep) {
  std::vector<std::string> parts;
  std::string cur;
  int depth = 0;
  char quote = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (quote) {
      cur += c;
      if (c == '\\' && i + 1 < s.size()) cur += s[++i];
      else if (c == quote) quote = 0;
      continue;
    }
    if (c == '"' || c == '\'') {
      quote = c;
      cur += c;
    } else if (c == '(' || c == '[' || c == '{') {
      ++depth;
      cur += c;
    } else if (c == ')' || c == ']' || c == '}') {
      if (depth > 0) --depth;
      cur += c;
    } else if (c == sep && depth == 0) {
      parts.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  parts.push_back(cur);
  return parts;
}

bool tool_is_ident(const std::string& s) {
  if (s.empty()) return false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    const bool ok = (c == '_') || (c >= 'A' && c <= 'Z') ||
                    (c >= 'a' && c <= 'z') || (i > 0 && c >= '0' && c <= '9');
    if (!ok) return false;
  }
  return true;
}

bool tool_looks_number(const std::string& s) {
  if (s.empty()) return false;
  size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
  bool digits = false, dot = false;
  for (; i < s.size(); ++i) {
    const char c = s[i];
    if (c >= '0' && c <= '9') {
      digits = true;
    } else if (c == '.' && !dot) {
      dot = true;
    } else if ((c == 'e' || c == 'E') && digits) {
      ++i;
      if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
      bool ed = false;
      for (; i < s.size(); ++i) {
        if (s[i] < '0' || s[i] > '9') return false;
        ed = true;
      }
      return ed;
    } else {
      return false;
    }
  }
  return digits;
}

// Decode standard escapes inside a quoted pythonic string body.
std::string tool_unescape_py(const std::string& body) {
  std::string out;
  for (size_t i = 0; i < body.size(); ++i) {
    const char c = body[i];
    if (c != '\\' || i + 1 >= body.size()) {
      out += c;
      continue;
    }
    const char e = body[++i];
    switch (e) {
      case 'n': out += '\n'; break;
      case 't': out += '\t'; break;
      case 'r': out += '\r'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case '0': out += '\0'; break;
      default: out += e; break;  // \\, \', \", and lenient passthrough
    }
  }
  return out;
}

// Recursive-descent native pythonic value -> JSON value. Lenient: bare
// words become JSON strings, garbage becomes null; entries always survive.
struct ToolPyValueParser {
  const std::string& s;
  size_t p = 0;

  void ws() {
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
  }
  std::string parse_value() {
    ws();
    if (p >= s.size()) return "null";
    const char c = s[p];
    if (c == '"' || c == '\'') {
      const char q = s[p++];
      std::string body;
      while (p < s.size() && s[p] != q) {
        if (s[p] == '\\' && p + 1 < s.size()) {
          body += s[p++];
          body += s[p++];
        } else {
          body += s[p++];
        }
      }
      if (p < s.size()) ++p;  // closing quote
      std::string out;
      tool_append_json_string(tool_unescape_py(body), out);
      return out;
    }
    if (c == '[' || c == '{') {
      const char open = c;
      const char close = (open == '[') ? ']' : '}';
      ++p;
      std::string out;
      out += open;
      bool first = true;
      while (true) {
        ws();
        if (p >= s.size()) break;  // unterminated: close leniently
        if (s[p] == close) {
          ++p;
          break;
        }
        std::string key_prefix;
        if (open == '{') {
          // Key: quoted string or bare word.
          if (s[p] == '"' || s[p] == '\'') {
            key_prefix = parse_value();
          } else {
            size_t b = p;
            while (p < s.size() && s[p] != ':' && s[p] != ',' && s[p] != close &&
                   !std::isspace(static_cast<unsigned char>(s[p])))
              ++p;
            tool_append_json_string(tool_trim(s.substr(b, p - b)),
                                    key_prefix);
          }
          ws();
          if (p < s.size() && s[p] == ':') {
            ++p;
          } else {
            // Not a pair: treat the key text as a positional value.
            if (!first) out += ",";
            first = false;
            out += key_prefix;
            ws();
            if (p < s.size() && s[p] == ',') {
              ++p;
              continue;
            }
            continue;
          }
        }
        const std::string val = parse_value();
        if (!first) out += ",";
        first = false;
        out += key_prefix;
        if (open == '{') out += ":";
        out += val;
        ws();
        if (p < s.size() && s[p] == ',') {
          ++p;
          continue;
        }
        if (p < s.size() && s[p] == close) continue;
        break;  // garbage: stop, close below
      }
      ws();
      if (p < s.size() && s[p] == close) ++p;
      return out + close;
    }
    if (s.compare(p, 4, "True") == 0) {
      p += 4;
      return "true";
    }
    if (s.compare(p, 5, "False") == 0) {
      p += 5;
      return "false";
    }
    if (s.compare(p, 4, "None") == 0) {
      p += 4;
      return "null";
    }
    // Number or bare word: consume to the next structural char.
    size_t b = p;
    int depth = 0;
    while (p < s.size()) {
      const char d = s[p];
      if (d == '(' || d == '[' || d == '{') ++depth;
      if (d == ')' || d == ']' || d == '}') {
        if (depth == 0) break;
        --depth;
      }
      if (depth == 0 && d == ',') break;
      ++p;
    }
    const std::string tok = tool_trim(s.substr(b, p - b));
    if (tool_looks_number(tok)) return tok;
    std::string out;
    tool_append_json_string(tok, out);
    return out;
  }
};

// Minimal JSON value -> pythonic converter for history rendering.
// Returns "" when the top level is not a JSON object; sets ok=false on
// any malformed input.
struct ToolJsonArgParser {
  const std::string& s;
  size_t p = 0;
  bool ok = true;

  void ws() {
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
  }
  bool expect(char c) {
    ws();
    if (p < s.size() && s[p] == c) {
      ++p;
      return true;
    }
    ok = false;
    return false;
  }
  // Parses any JSON value into its pythonic spelling.
  std::string parse_value() {
    ws();
    if (p >= s.size()) {
      ok = false;
      return "None";
    }
    const char c = s[p];
    if (c == '"') {
      ++p;
      std::string body;
      while (p < s.size() && s[p] != '"') {
        if (s[p] == '\\' && p + 1 < s.size()) {
          const char e = s[++p];
          switch (e) {
            case 'n': body += '\n'; break;
            case 't': body += '\t'; break;
            case 'r': body += '\r'; break;
            case 'b': body += '\b'; break;
            case 'f': body += '\f'; break;
            case 'u': {
              // Best-effort \uXXXX: BMP values that fit one UTF-8 sequence.
              unsigned cp = 0;
              bool good = (p + 4 < s.size());
              for (int k = 1; good && k <= 4; ++k) {
                const char h = s[p + k];
                cp <<= 4;
                if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                else good = false;
              }
              if (good) {
                if (cp < 0x80) {
                  body += (char)cp;
                } else if (cp < 0x800) {
                  body += (char)(0xC0 | (cp >> 6));
                  body += (char)(0x80 | (cp & 0x3F));
                } else {
                  body += (char)(0xE0 | (cp >> 12));
                  body += (char)(0x80 | ((cp >> 6) & 0x3F));
                  body += (char)(0x80 | (cp & 0x3F));
                }
                p += 4;
              } else {
                body += 'u';
              }
              break;
            }
            default: body += e; break;
          }
          ++p;
        } else {
          body += s[p++];
        }
      }
      if (p < s.size()) ++p;  // closing quote
      std::string out = "\"";
      for (char b : body) {
        if (b == '"') out += "\\\"";
        else if (b == '\\') out += "\\\\";
        else if (b == '\n') out += "\\n";
        else out += b;
      }
      return out + "\"";
    }
    if (c == '{' || c == '[') {
      const char open = c;
      const char close = (open == '{') ? '}' : ']';
      ++p;
      std::string out;
      out += open;
      bool first = true;
      bool closed = false;
      while (p < s.size()) {
        ws();
        if (p < s.size() && s[p] == close) {
          ++p;
          closed = true;
          break;
        }
        if (open == '{') {
          if (p >= s.size() || s[p] != '"') {
            ok = false;
            break;
          }
          const std::string key = parse_value();
          if (!expect(':')) break;
          const std::string val = parse_value();
          if (!first) out += ", ";
          first = false;
          out += key + ": " + val;
        } else {
          const std::string v = parse_value();
          if (!first) out += ", ";
          first = false;
          out += v;
        }
        ws();
        if (p < s.size() && s[p] == ',') {
          ++p;
          continue;
        }
        if (p < s.size() && s[p] == close) {
          ++p;
          closed = true;
          break;
        }
        ok = false;
        break;
      }
      if (!closed) ok = false;
      return out + close;
    }
    if (s.compare(p, 4, "true") == 0) {
      p += 4;
      return "True";
    }
    if (s.compare(p, 5, "false") == 0) {
      p += 5;
      return "False";
    }
    if (s.compare(p, 4, "null") == 0) {
      p += 4;
      return "None";
    }
    size_t b = p;
    while (p < s.size() && s[p] != ',' && s[p] != ']' && s[p] != '}' &&
           !std::isspace(static_cast<unsigned char>(s[p])))
      ++p;
    const std::string tok = s.substr(b, p - b);
    if (tok.empty() || !tool_looks_number(tok)) ok = false;
    return tok;
  }
};

// Format one parsed call back into native `name(args)` form.
std::string tool_format_call(const ToolCall& c) {
  return c.name + "(" + args_json_to_pythonic(c.arguments) + ")";
}

// Markers whose trailing partial heads must never leak into streamed or
// whole-text output (a length-capped turn can end mid-marker, and a chunk
// boundary can split one). Kept in one place so the whole-text parse and
// the streaming hold-back agree byte-for-byte.
const char* const kHoldMarkers[] = {
    kThinkStartToken,       kThinkEndToken,
    kToolCallStartToken,    kToolCallEndToken,
    kThinkStartAltToken,    kThinkEndAltToken,
    kToolCallStartAltToken, kToolCallEndAltToken,
    kEosToken};

// Earliest occurrence of any of the alternative markers at/after `from`.
// Returns npos when none match; `which` (optional) points at the matched
// literal so callers can advance by its length. Ties resolve in argument
// order, which keeps the native marker preferred over an alias at the same
// position (they never actually collide).
size_t find_earliest(const std::string& b, size_t from,
                     std::initializer_list<const char*> markers,
                     const char** which = nullptr) {
  size_t best = std::string::npos;
  for (const char* m : markers) {
    const size_t p = b.find(m, from);
    if (p != std::string::npos && (best == std::string::npos || p < best)) {
      best = p;
      if (which) *which = m;
    }
  }
  return best;
}

// Length of the think opener (`<think>` or `<thinking>`) at `p`, else 0.
size_t think_opener_at(const std::string& b, size_t p) {
  const char* m = nullptr;
  const size_t s =
      find_earliest(b, p, {kThinkStartToken, kThinkStartAltToken}, &m);
  return (s == p && m) ? std::string(m).size() : 0;
}

// Earliest think terminator (closing tag, tool-call start, or turn end).
// kind 0 = an explicit closing tag (consumed as part of the trace), kind 1 =
// an implicit terminator left in the display for the call/stop stages.
size_t find_think_terminator(const std::string& b, size_t from, size_t& len,
                             int& kind) {
  const char* m = nullptr;
  const size_t p =
      find_earliest(b, from,
                    {kThinkEndToken, kThinkEndAltToken, kToolCallStartToken,
                     kToolCallStartAltToken, kEosToken},
                    &m);
  if (p == std::string::npos) {
    len = 0;
    kind = 0;
    return p;
  }
  len = std::string(m).size();
  kind = (m == kThinkEndToken || m == kThinkEndAltToken) ? 0 : 1;
  return p;
}

// Length of the trailing partial marker head at the end of b (0 when the
// end is marker-clean). Only proper prefixes count: a complete marker is
// structural, not a fragment.
size_t trailing_partial_marker(const std::string& b) {
  size_t hold = 0;
  for (const char* m : kHoldMarkers) {
    const std::string s(m);
    if (s.size() <= 1) continue;
    const size_t max_k = std::min(b.size(), s.size() - 1);
    for (size_t k = max_k; k >= 1; --k) {
      if (b.compare(b.size() - k, k, s, 0, k) == 0) {
        hold = std::max(hold, k);
        break;
      }
      if (k == 1) break;  // size_t underflow guard
    }
  }
  return hold;
}

// Only the single longest trailing partial control-marker head is ever
// stripped/held: any earlier '<' is followed by '<' and therefore cannot
// begin a marker, so it is unambiguous content. Stripping to a fixpoint
// would drop those content bytes ("hello <<" -> "hello " instead of
// "hello <") and, in the streaming layers, hold an unbounded run while
// rescanning it on every push (O(N^2)). l2/l3 already use this single-head
// form; the whole-text parse, l1 and l4 must agree with it byte-for-byte.

// Think-block split: extracts the leading LFM2.5 <think>...</think>
// reasoning trace from raw completion text (specials visible). display is
// the input with think blocks removed (markers included); reasoning
// concatenates the block contents in order. The generation prompt ends
// with an open <think>, so raw completions normally open implicitly: when
// start_in_reasoning is true, the leading segment up to the first
// </think> / tool-call start / turn terminator is reasoning even without
// an explicit opener (a redundant leading opener is stripped). An
// explicit opener later in the turn only counts while no non-whitespace
// display content precedes it (tool-call blocks excluded, so thinking
// after a tool call still counts); a later literal `<think>`
// (e.g. the model discussing its own special tokens) stays display
// content instead of swallowing the answer into reasoning. A block closes
// at the first </think>; a tool call start or the turn terminator also
// ends the block implicitly (the terminator itself is left in the display
// for the tool/stop stages), so a turn cut or generated without a close
// tag still separates reasoning from content. An unclosed trailing
// explicit <think> treats the rest as reasoning; a lone </think> without
// an opener passes through into the display unless start_in_reasoning
// covers the leading segment.
struct SplitThink {
  std::string display;
  std::string reasoning;
};

// True when b[pos, end) holds no non-whitespace outside complete tool-call
// blocks. Complete blocks are excised downstream, so they must not count as
// display content (otherwise thinking after a tool call would look mid-turn
// and leak into `content`). An unclosed block start counts as content, as
// does a block whose end marker lies past `end`: from this window's point of
// view it may still be plain text. Shared by the whole-text split and the
// streaming layer so both classify a gap identically.
bool blank_ignoring_tool_blocks(const std::string& b, size_t pos,
                                size_t end) {
  while (pos < end) {
    const char* sm = nullptr;
    const size_t st =
        find_earliest(b, pos, {kToolCallStartToken, kToolCallStartAltToken},
                      &sm);
    if (st == std::string::npos || st >= end) break;
    for (size_t i = pos; i < st; ++i)
      if (!std::isspace(static_cast<unsigned char>(b[i]))) return false;
    const char* em = nullptr;
    const size_t en =
        find_earliest(b, st + std::string(sm).size(),
                      {kToolCallEndToken, kToolCallEndAltToken}, &em);
    if (en == std::string::npos || en + std::string(em).size() > end)
      return false;
    pos = en + std::string(em).size();
  }
  for (size_t i = pos; i < end; ++i)
    if (!std::isspace(static_cast<unsigned char>(b[i]))) return false;
  return true;
}

SplitThink split_think_blocks(const std::string& raw,
                              bool start_in_reasoning) {
  SplitThink out;
  // An explicit opener only counts before any non-whitespace display
  // content (tool blocks excluded); anything already emitted plus the gap
  // before the opener must be blank. Tracks incrementally: prior_out_ws is
  // whether out.display so far is all whitespace ignoring tool blocks.
  bool prior_out_ws = true;
  size_t pos = 0;
  // Prompt-provided open thinker: the leading segment is reasoning up to
  // the first decided closer / tool-call start / turn terminator, even
  // with no explicit opener in the raw text. A turn that never reaches a
  // terminator is a completion cut mid-thought (length cap), so the whole
  // remaining text is the trace — the streaming view cannot know at
  // token time whether a terminator is still coming, so it commits to
  // reasoning and the whole-text parse has to agree (otherwise the two
  // views diverge on truncated turns). Leading whitespace belongs to the
  // trace in every case: it precedes the (implicit or redundant) opener,
  // so it must never reach `content` (a whitespace-only `content` would
  // also suppress the null-content tool-call shape).
  if (start_in_reasoning) {
    size_t ws_end = 0;
    while (ws_end < raw.size() &&
           std::isspace(static_cast<unsigned char>(raw[ws_end])))
      ++ws_end;
    const size_t opener = think_opener_at(raw, ws_end);
    const size_t seg = ws_end + opener;
    size_t term_len = 0;
    int term_kind = 0;  // 0 = close tag, 1 = implicit terminator
    const size_t term = find_think_terminator(raw, seg, term_len, term_kind);
    out.reasoning += raw.substr(0, ws_end);
    if (term == std::string::npos) {
      // No terminator: cut mid-thought (or a degenerate turn that never
      // closed the prompt-opened trace). Everything after the opener is
      // the trace.
      out.reasoning += raw.substr(seg);
      return out;
    }
    out.reasoning += raw.substr(seg, term - seg);
    pos = (term_kind == 0) ? term + term_len : term;
  }
  while (true) {
    const char* open_m = nullptr;
    const size_t s = find_earliest(
        raw, pos, {kThinkStartToken, kThinkStartAltToken}, &open_m);
    if (s == std::string::npos) {
      out.display += raw.substr(pos);
      break;
    }
    const size_t open_len = std::string(open_m).size();
    // Gap blank (and nothing non-blank emitted yet), ignoring excised
    // tool blocks?
    bool gap_ws =
        prior_out_ws && blank_ignoring_tool_blocks(raw, pos, s);
    if (!gap_ws) {
      // Literal `<think>` inside content (or a tool-call argument): keep
      // the marker and keep scanning after it.
      out.display += raw.substr(pos, s - pos + open_len);
      prior_out_ws = false;
      pos = s + open_len;
      continue;
    }
    out.display += raw.substr(pos, s - pos);
    const size_t inner = s + open_len;
    size_t e_len = 0;
    int e_kind = 0;  // 0 = close tag, 1 = implicit terminator
    const size_t e = find_think_terminator(raw, inner, e_len, e_kind);
    if (e == std::string::npos) {
      out.reasoning += raw.substr(inner);
      break;
    }
    out.reasoning += raw.substr(inner, e - inner);
    pos = (e_kind == 0) ? e + e_len : e;
  }
  return out;
}

// The turn terminator ends every turn, independent of caller stops:
// without it a custom stop list would leak a complete <|im_end|> into
// content (the parser decodes with specials visible) and the flushed
// tail could re-emit text past a custom stop cut. Union, don't replace,
// so caller stops keep working.
std::vector<std::string> with_turn_terminator(
    std::vector<std::string> stops) {
  for (const auto& s : stops)
    if (s == kEosToken) return stops;
  stops.emplace_back(kEosToken);
  return stops;
}

}  // namespace

std::string args_pythonic_to_json(const std::string& args) {
  try {
    const std::string t = tool_trim(args);
    if (t.empty()) return "{}";
    std::string out = "{";
    bool first = true;
    for (const std::string& entry : tool_split_top(t, ',')) {
      const size_t eq = entry.find('=');
      if (eq == std::string::npos) continue;
      const std::string key = tool_trim(entry.substr(0, eq));
      if (!tool_is_ident(key)) continue;
      ToolPyValueParser pv{tool_trim(entry.substr(eq + 1)), 0};
      const std::string val = pv.parse_value();
      std::string jk;
      tool_append_json_string(key, jk);
      if (!first) out += ",";
      first = false;
      out += jk + ":" + val;
    }
    return out + "}";
  } catch (...) {
    return "{}";
  }
}

std::string args_json_to_pythonic(const std::string& args_json) {
  try {
    const std::string t = tool_trim(args_json);
    if (t.empty() || t == "{}") return "";
    ToolJsonArgParser jp{t, 0, true};
    if (!jp.expect('{')) return "";
    std::string out;
    bool first = true;
    jp.ws();
    if (jp.p < jp.s.size() && jp.s[jp.p] == '}') return "";
    while (jp.p < jp.s.size()) {
      jp.ws();
      if (jp.p < jp.s.size() && jp.s[jp.p] == '}') {
        ++jp.p;
        break;
      }
      if (jp.p >= jp.s.size() || jp.s[jp.p] != '"') return "";
      ++jp.p;  // opening quote of the key
      std::string key;
      while (jp.p < jp.s.size() && jp.s[jp.p] != '"') {
        if (jp.s[jp.p] == '\\' && jp.p + 1 < jp.s.size()) {
          key += jp.s[jp.p++];
          key += jp.s[jp.p++];
        } else {
          key += jp.s[jp.p++];
        }
      }
      if (jp.p >= jp.s.size()) return "";
      ++jp.p;  // closing quote
      if (!tool_is_ident(key)) return "";
      if (!jp.expect(':')) return "";
      const std::string val = jp.parse_value();
      if (!jp.ok) return "";
      if (!first) out += ", ";
      first = false;
      out += key + "=" + val;
      jp.ws();
      if (jp.p < jp.s.size() && jp.s[jp.p] == ',') {
        ++jp.p;
        continue;
      }
      if (jp.p < jp.s.size() && jp.s[jp.p] == '}') continue;
      break;
    }
    if (!jp.ok) return "";
    jp.ws();
    if (jp.p < jp.s.size()) return "";  // closing '}' already consumed above
    return out;
  } catch (...) {
    return "";
  }
}

namespace {

// Index of the ')' matching the '(' at `open` (quotes respected), or npos
// when the list is truncated.
size_t tool_matching_paren(const std::string& t, size_t open) {
  int depth = 0;
  char quote = 0;
  bool esc = false;
  for (size_t i = open; i < t.size(); ++i) {
    const char c = t[i];
    if (quote) {
      if (esc) {
        esc = false;
      } else if (c == '\\') {
        esc = true;
      } else if (c == quote) {
        quote = 0;
      }
      continue;
    }
    if (c == '"' || c == '\'') {
      quote = c;
    } else if (c == '(') {
      ++depth;
    } else if (c == ')' && --depth == 0) {
      return i;
    }
  }
  return std::string::npos;
}

// `<parameter=k>value</parameter>` pairs (LFM3 / qwen3-coder grammar) to a
// JSON object string. Values are raw text (newlines trimmed).
std::string tool_xml_params_to_json(const std::string& body) {
  std::string out = "{";
  bool first = true;
  size_t pos = 0;
  while ((pos = body.find("<parameter", pos)) != std::string::npos) {
    const size_t tag_end = body.find('>', pos);
    const size_t eq = body.find('=', pos);
    if (tag_end == std::string::npos || eq == std::string::npos ||
        eq > tag_end) {
      break;
    }
    const std::string key = tool_trim(body.substr(eq + 1, tag_end - eq - 1));
    if (!tool_is_ident(key)) {
      pos = tag_end + 1;
      continue;
    }
    const size_t vs = tag_end + 1;
    size_t ve = body.find("</parameter>", vs);
    const bool closed = ve != std::string::npos;
    if (!closed) ve = body.size();
    const std::string val = tool_trim(body.substr(vs, ve - vs));
    std::string jk, jv;
    tool_append_json_string(key, jk);
    tool_append_json_string(val, jv);
    if (!first) out += ",";
    first = false;
    out += jk + ":" + jv;
    pos = closed ? ve + 12 : body.size();
  }
  return out + "}";
}

// Lenient scanner for the LFM3 / qwen3-coder grammar and hybrids of it with
// the native pythonic form:
//   <function=name><parameter=k>v</parameter></function>
//   <function>name</function><parameter=k>v</parameter></function>
//   <function=name(args)>
//   name(args)                     (only when reached after a `function=`)
// Each call is consumed whole, so commas inside a parameter value or an
// argument list cannot split it.
void tool_scan_marked_calls(const std::string& t,
                            std::vector<ToolCall>& calls) {
  const std::string head = "function";
  const std::string fn_close = "</function>";
  size_t pos = 0;
  while ((pos = t.find(head, pos)) != std::string::npos) {
    size_t q = pos + head.size();
    while (q < t.size() && std::isspace(static_cast<unsigned char>(t[q])))
      ++q;
    if (q < t.size() && t[q] == '>') {
      // Degenerate `<function>NAME</function>` form: the name is element
      // text and any parameter tags follow its own closing tag.
      const size_t ne = t.find(fn_close, q + 1);
      if (ne == std::string::npos) break;
      const std::string name = tool_trim(t.substr(q + 1, ne - q - 1));
      if (!tool_is_ident(name)) {
        pos = ne + fn_close.size();
        continue;
      }
      const size_t ps = ne + fn_close.size();
      const size_t pe = t.find(fn_close, ps);
      ToolCall c;
      c.name = name;
      c.arguments = tool_xml_params_to_json(
          t.substr(ps, pe == std::string::npos ? std::string::npos : pe - ps));
      calls.push_back(std::move(c));
      pos = (pe == std::string::npos) ? t.size() : pe + fn_close.size();
      continue;
    }
    if (q >= t.size() || t[q] != '=') {
      pos = std::max(q, pos + 1);  // unrelated prose mentioning "function"
      continue;
    }
    ++q;
    while (q < t.size() && std::isspace(static_cast<unsigned char>(t[q])))
      ++q;
    const size_t ns = q;
    while (q < t.size() &&
           (std::isalnum(static_cast<unsigned char>(t[q])) || t[q] == '_' ||
            t[q] == '-' || t[q] == '.'))
      ++q;
    const std::string name = t.substr(ns, q - ns);
    if (!tool_is_ident(name)) {
      pos = q;
      continue;
    }
    size_t r = q;
    while (r < t.size() && std::isspace(static_cast<unsigned char>(t[r])))
      ++r;
    if (r < t.size() && t[r] == '(') {
      const size_t close = tool_matching_paren(t, r);
      if (close == std::string::npos) break;  // truncated argument list
      ToolCall c;
      c.name = name;
      c.arguments = args_pythonic_to_json(t.substr(r + 1, close - r - 1));
      calls.push_back(std::move(c));
      pos = close + 1;
      continue;
    }
    if (r < t.size() && t[r] == '>') {
      const size_t fs = r + 1;
      const size_t fe = t.find(fn_close, fs);
      ToolCall c;
      c.name = name;
      c.arguments = tool_xml_params_to_json(
          t.substr(fs, fe == std::string::npos ? std::string::npos : fe - fs));
      calls.push_back(std::move(c));
      if (fe == std::string::npos) break;
      pos = fe + fn_close.size();
      continue;
    }
    pos = std::max(q, pos + 1);
  }
}

}  // namespace

std::vector<ToolCall> parse_tool_call_block(const std::string& block) {
  std::vector<ToolCall> calls;
  try {
    std::string t = tool_trim(block);
    // Native blocks wrap calls in [...]; tolerate either bracket missing
    // (a max_tokens cut usually drops the trailing ']' but keeps '[').
    // Strip each end independently so "[f(a=1), g(b=2)" still recovers f.
    if (!t.empty() && t.front() == '[') t = tool_trim(t.substr(1));
    if (!t.empty() && t.back() == ']') t = tool_trim(t.substr(0, t.size() - 1));
    if (t.empty()) return calls;
    // Marked (LFM3 / qwen3-coder or `function=`-prefixed hybrid) bodies use
    // the structural scanner; plain native bodies use the pythonic list.
    // Gate on the body *starting* with a marked head (or carrying parameter
    // tags) so a native argument string that merely contains "function=" is
    // not mistaken for a call.
    const bool marked = t.compare(0, 9, "<function") == 0 ||
                        t.compare(0, 9, "function=") == 0 ||
                        t.find("<parameter") != std::string::npos;
    if (marked) {
      tool_scan_marked_calls(t, calls);
      if (!calls.empty()) return calls;
    }
    for (const std::string& item : tool_split_top(t, ',')) {
      const std::string e = tool_trim(item);
      if (e.empty()) continue;
      const size_t open = e.find('(');
      if (open == std::string::npos || e.back() != ')') continue;
      const std::string name = tool_trim(e.substr(0, open));
      if (!tool_is_ident(name)) continue;
      const std::string args = e.substr(open + 1, e.size() - open - 2);
      ToolCall c;
      c.name = name;
      c.arguments = args_pythonic_to_json(args);
      calls.push_back(std::move(c));
    }
  } catch (...) {
  }
  return calls;
}

ToolParseResult parse_tool_calls(const std::string& raw,
                                 const std::vector<std::string>& stops,
                                 bool start_in_reasoning) {
  ToolParseResult r;
  try {
    // Length-capped turns can end mid-marker; strip the single trailing
    // partial control-marker head up front so neither display nor
    // reasoning leaks a fragment the streaming view deliberately held.
    // Same single-head helper as the streaming settled region: both views
    // drop exactly the same bytes, so an already-streamed delta is always
    // a prefix of the whole-text result and finish() never retracts.
    std::string input = raw;
    if (const size_t hold = trailing_partial_marker(input); hold > 0)
      input.erase(input.size() - hold);
    const SplitThink th = split_think_blocks(input, start_in_reasoning);
    r.reasoning = th.reasoning;
    const std::string& text_in = th.display;
    std::string text;
    size_t pos = 0;
    while (true) {
      const char* sm = nullptr;
      const size_t s = find_earliest(
          text_in, pos, {kToolCallStartToken, kToolCallStartAltToken}, &sm);
      if (s == std::string::npos) {
        text += text_in.substr(pos);
        break;
      }
      const size_t start_len = std::string(sm).size();
      text += text_in.substr(pos, s - pos);
      const char* em = nullptr;
      const size_t e =
          find_earliest(text_in, s + start_len,
                        {kToolCallEndToken, kToolCallEndAltToken}, &em);
      if (e == std::string::npos) {
        // Truncated tail (length-capped inside a call block): recover
        // best-effort so a cut-off call still executes instead of leaking
        // call markup into the display text. Unparseable tails stay text.
        const std::string inner = text_in.substr(s + start_len);
        bool recovered = false;
        for (auto& c : parse_tool_call_block(inner)) {
          r.calls.push_back(std::move(c));
          recovered = true;
        }
        if (!recovered) text += inner;
        break;
      }
      // The opener and closer may be different spellings (an XML opener
      // with the native end token, a hybrid observed in the wild), so the
      // body is just the bytes between the matched pair.
      for (auto& c : parse_tool_call_block(
               text_in.substr(s + start_len, e - s - start_len)))
        r.calls.push_back(std::move(c));
      pos = e + std::string(em).size();
    }
    // The turn ends at the first stop: anything from it onward is not
    // content. The turn terminator is unioned in (see with_turn_terminator)
    // so the flushed streaming tail agrees with the streamed deltas even
    // when the caller passes custom stops; callers without stops get the
    // plain turn-terminator cut.
    const std::vector<std::string> eff = with_turn_terminator(stops);
    const size_t stop = find_stop_string(text, eff);
    if (stop != std::string::npos) text.erase(stop);
    r.text = std::move(text);
  } catch (...) {
    r.calls.clear();
  }
  return r;
}

std::string render_lfm25_chatml_with_tools(
    const std::vector<ChatMessage>& messages, const std::string& tools_json,
    const std::string& system_suffix, bool add_generation_prompt,
    bool preserve_thinking, const std::string& default_system) {
  // Effective system block: first system message (if any) + default +
  // tools catalogue + caller directives.
  std::string sys;
  bool has_system = false;
  for (const auto& m : messages)
    if (m.role == "system") {
      has_system = true;
      if (!m.content.empty()) sys += (sys.empty() ? "" : "\n") + m.content;
    }
  if (!has_system && !default_system.empty()) sys = default_system;
  if (!tools_json.empty())
    sys += (sys.empty() ? "" : "\n") + std::string("List of tools: ") +
           tools_json;
  if (!system_suffix.empty())
    sys += (sys.empty() ? "" : "\n") + system_suffix;

  std::string out;
  if (has_system || !sys.empty())
    out += std::string(kImStartToken) + "system\n" + sys + kEosToken + "\n";
  for (const auto& m : messages) {
    if (m.role == "system") continue;  // folded into the block above
    const std::string role = (m.role == "tool") ? "tool" : m.role;
    out += std::string(kImStartToken) + role + "\n";
    if (preserve_thinking && m.role == "assistant" && !m.thinking.empty())
      out += std::string(kThinkStartToken) + m.thinking + kThinkEndToken;
    out += m.content;
    if (m.role == "assistant" && !m.tool_calls.empty()) {
      out += kToolCallStartToken;
      out += "[";
      bool first = true;
      for (const auto& c : m.tool_calls) {
        if (c.name.empty() || !tool_is_ident(c.name)) continue;
        if (!first) out += ", ";
        first = false;
        out += tool_format_call(c);
      }
      out += "]";
      out += kToolCallEndToken;
    }
    out += std::string(kEosToken) + "\n";
  }
  if (add_generation_prompt)
    out += std::string(kImStartToken) + "assistant\n" + kThinkStartToken;
  return out;
}

std::vector<int> encode_chatml_with_tools(
    const Tokenizer& tok, const std::vector<ChatMessage>& messages,
    const std::string& tools_json, const std::string& system_suffix,
    bool add_generation_prompt, bool add_bos, bool preserve_thinking) {
  return encode_rendered_chat(
      tok,
      render_lfm25_chatml_with_tools(messages, tools_json, system_suffix,
                                     add_generation_prompt,
                                     preserve_thinking),
      add_bos);
}

ToolCallStreamParser::ToolCallStreamParser(
    const std::vector<std::string>& stop_strings, bool start_in_reasoning)
    : l2_lead_(start_in_reasoning),
      stops_(with_turn_terminator(stop_strings)),
      start_in_reasoning_(start_in_reasoning) {
  // (The turn terminator is unioned in: it ends every turn even when the
  // caller passes custom stops, so a complete <|im_end|> can never leak
  // into content. Empty input therefore needs no special case.)
  max_stop_len_ = 1;
  for (const auto& s : stops_)
    max_stop_len_ = std::max(max_stop_len_, s.size());
}

// Length of the trailing incomplete UTF-8 sequence in the prefix ending at
// end (end <= b.size(); 0 when character-complete). Mirrors
// IncrementalDetokenizer policy; the _at form exists because the emission
// frontier may lag the buffer end once a stop cut applies.
size_t stream_trailing_partial_utf8(const std::string& b, size_t end) {
  if (end == 0 || end > b.size()) return 0;
  size_t cont = 0;
  while (cont < end && cont < 4 &&
         (static_cast<unsigned char>(b[end - 1 - cont]) & 0xC0u) == 0x80u)
    ++cont;
  if (cont >= 4 || cont == end) return 0;
  const unsigned char lead = static_cast<unsigned char>(b[end - 1 - cont]);
  size_t want = 1;
  if (lead >= 0xC0u && lead < 0xE0u) want = 2;
  else if (lead >= 0xE0u && lead < 0xF0u) want = 3;
  else if (lead >= 0xF0u && lead < 0xF8u) want = 4;
  else return 0;
  return (1 + cont < want) ? 1 + cont : 0;
}

// Layer 1: settled delivery. raw_ is append-only; l1_delivered_ only moves
// forward over fully scanned text and complete call blocks, so each byte
// is scanned O(1) times. An unclosed <think> is deliberately NOT held:
// thinking streams live as reasoning. Undecided trailing partial marker
// heads stay in raw_ (l1_delivered_ lags) until they complete or are
// disproven, so a chunk boundary never splits a marker across the layer.
void ToolCallStreamParser::l1_drive() {
  const size_t max_end = std::string(kToolCallEndToken).size();
  const size_t n = raw_.size();
  while (true) {
    if (l1_in_block_) {
      const char* em = nullptr;
      const size_t e = find_earliest(
          raw_, l1_end_search_, {kToolCallEndToken, kToolCallEndAltToken}, &em);
      if (e == std::string::npos) {
        // Keep a bounded overlap so an end marker split across the chunk
        // boundary is still found next time; each byte is scanned O(1).
        const size_t floor = (n >= max_end - 1) ? n - (max_end - 1) : 0;
        l1_end_search_ = std::max(l1_end_search_, floor);
        return;
      }
      const size_t end_len = std::string(em).size();
      // The block closed: deliver the held head bytes (if any) plus the
      // whole block, in order, then keep scanning for what follows.
      std::string piece = l1_prehold_;
      l1_prehold_.clear();
      piece += raw_.substr(l1_block_start_, e + end_len - l1_block_start_);
      l1_delivered_ = e + end_len;
      l1_in_block_ = false;
      l2_consume(piece);
      continue;
    }
    const char* sm = nullptr;
    const size_t s = find_earliest(
        raw_, l1_delivered_, {kToolCallStartToken, kToolCallStartAltToken},
        &sm);
    if (s == std::string::npos) {
      // Block-free tail: deliver everything but the single trailing
      // partial control-marker head (same single-head helper as the
      // whole-text parse, so both views drop exactly the same bytes at a
      // length-capped cut). Holding only the longest head keeps the
      // undelivered tail bounded, so each push costs O(1).
      std::string cand = raw_.substr(l1_delivered_);
      if (const size_t hold = trailing_partial_marker(cand); hold > 0)
        cand.erase(cand.size() - hold);
      l1_delivered_ += cand.size();
      l2_consume(cand);
      return;
    }
    std::string text = raw_.substr(l1_delivered_, s - l1_delivered_);
    if (const size_t hold = trailing_partial_marker(text); hold > 0)
      text.erase(text.size() - hold);
    // Heads adjacent to the block start are held (they may yet complete
    // with bytes from inside or past the block) and delivered first when
    // the block closes, preserving input order.
    l1_prehold_ = raw_.substr(l1_delivered_ + text.size(),
                              s - l1_delivered_ - text.size());
    l2_consume(text);
    l1_delivered_ = s;
    l1_in_block_ = true;
    l1_block_start_ = s;
    l1_end_search_ = s + std::string(sm).size();
  }
}

// Layer 2: think split over the settled stream, mirroring
// split_think_blocks stage-for-stage on the same bytes (complete blocks
// included, so an opener inside call arguments behaves identically). Only
// the newly delivered piece plus the bounded undecided tail is examined;
// a marker starting before the firm frontier F is always complete, so
// firm bytes are routed exactly once. Display bytes flow to layer 3,
// reasoning bytes accumulate in out_r_ (both append-only).
void ToolCallStreamParser::l2_consume(const std::string& piece) {
  if (piece.empty()) return;
  std::string buf = l2_pend_ + piece;
  l2_pend_.clear();
  // Firm frontier: everything before a trailing partial control-marker
  // head is decided exactly as the whole-text split would (a marker
  // starting before the frontier is always complete, so firm bytes are
  // routed once and never revisited). The head itself is held until it
  // completes or is disproven, so a chunk boundary never splits a marker
  // across the layer. Marker-clean input therefore streams with no lag.
  const size_t f = buf.size() - trailing_partial_marker(buf);
  size_t pos = 0;
  std::string disp;
  disp.reserve(piece.size());
  // Prompt-provided open thinker (l2_lead_): the turn opens inside
  // reasoning, so settled bytes stream live as reasoning until the first
  // decided closer / tool-call start / turn terminator. A redundant
  // leading opener is stripped. This mirrors the start_in_reasoning leg
  // of split_think_blocks; the flag disambiguates immediately, so no
  // holding is needed to keep the views monotone.
  if (l2_lead_) {
    size_t seg = 0;
    // Fresh: nothing but whitespace has been routed to the trace so far,
    // so one redundant opener may still be stripped (mirroring the
    // whole-text leg, which only strips an opener preceded by whitespace).
    // Once a non-whitespace byte or the opener itself is consumed, a later
    // `<think>` is literal trace text exactly as in the whole-text pass.
    if (l2_lead_fresh_) {
      size_t ws_end = 0;
      while (ws_end < f &&
             std::isspace(static_cast<unsigned char>(buf[ws_end])))
        ++ws_end;
      out_r_ += buf.substr(0, ws_end);
      seg = ws_end;
      const size_t opener = think_opener_at(buf, ws_end);
      if (opener > 0 && ws_end + opener <= f) {
        seg = ws_end + opener;
        l2_lead_fresh_ = false;
      } else if (ws_end < f) {
        l2_lead_fresh_ = false;  // a non-blank byte: no opener can follow
      } else {
        // Only whitespace settled so far: hold the undecided head (which
        // may be the opener) and wait for more bytes.
        l2_pend_ = buf.substr(seg);
        l3_consume(disp);
        return;
      }
    }
    size_t e_len = 0;
    int e_kind = 0;
    const size_t e = find_think_terminator(buf, seg, e_len, e_kind);
    if (e == std::string::npos || e >= f) {
      if (f > seg) out_r_ += buf.substr(seg, f - seg);
      // Undecided tail (possibly the terminator itself) is held.
      l2_pend_ = buf.substr(std::max(seg, f));
      l3_consume(disp);
      return;
    }
    out_r_ += buf.substr(seg, e - seg);
    pos = (e_kind == 0) ? e + e_len : e;
    l2_lead_ = false;
  }
  while (true) {
    if (l2_in_think_) {
      // Earliest of close / implicit terminators (same order and
      // tie-break as the whole-text split).
      size_t e_len = 0;
      int e_kind = 0;
      const size_t e = find_think_terminator(buf, pos, e_len, e_kind);
      if (e == std::string::npos || e >= f) {
        // No terminator decided yet: everything before the frontier stays
        // reasoning whatever follows (a later close only appends). pos may
        // already lie past the frontier (opener just consumed); only the
        // undecided tail is held.
        if (f > pos) out_r_ += buf.substr(pos, f - pos);
        pos = std::max(pos, f);
        break;
      }
      out_r_ += buf.substr(pos, e - pos);
      // A close tag is consumed; an implicit terminator is left in the
      // display path for the tool/stop stages.
      pos = (e_kind == 0) ? e + e_len : e;
      l2_in_think_ = false;
      continue;
    }
    const char* open_m = nullptr;
    const size_t s =
        find_earliest(buf, pos, {kThinkStartToken, kThinkStartAltToken},
                      &open_m);
    if (s == std::string::npos || s >= f) {
      if (f > pos) {
        const std::string seg = buf.substr(pos, f - pos);
        disp += seg;
        // Tool-call blocks are excised downstream, so they must not mark
        // the display as non-blank (thinking after a tool call still
        // counts as leading).
        if (l2_ws_) l2_ws_ = blank_ignoring_tool_blocks(seg, 0, seg.size());
      }
      pos = std::max(pos, f);
      break;
    }
    const size_t open_len = std::string(open_m).size();
    if (l2_ws_ && blank_ignoring_tool_blocks(buf, pos, s)) {
      disp += buf.substr(pos, s - pos);
      pos = s + open_len;
      l2_in_think_ = true;
      continue;
    }
    // Literal `<think>` inside content (or a tool-call argument): kept in
    // the display, and thinking is over for this turn.
    disp += buf.substr(pos, s - pos + open_len);
    l2_ws_ = false;
    pos = s + open_len;
  }
  l2_pend_ = buf.substr(pos);
  l3_consume(disp);
}

// Layer 3: tool-call block excision over the display stream (same order as
// the whole-text parse: excise first, stop-cut later in layer 4, so a stop
// inside a call block cannot desync the streaming view from finish()).
// In-block bytes are dropped firmly — whether the block later closes
// (excised, as in the whole-text parse) or stays cut (recovered as calls
// or backfilled as text by finish()) they never reach the display view.
void ToolCallStreamParser::l3_consume(const std::string& piece) {
  if (piece.empty()) return;
  std::string buf = l3_pend_ + piece;
  l3_pend_.clear();
  // Firm frontier (same rule as layer 2): a block marker starting before
  // it is always complete. In-block bytes are dropped firmly; the
  // undecided head is held so a marker split across pieces still excises.
  const size_t f = buf.size() - trailing_partial_marker(buf);
  size_t pos = 0;
  while (true) {
    if (l3_in_call_) {
      const char* em = nullptr;
      const size_t e = find_earliest(
          buf, pos, {kToolCallEndToken, kToolCallEndAltToken}, &em);
      if (e == std::string::npos || e >= f) {
        pos = std::max(pos, f);
        break;
      }
      pos = e + std::string(em).size();
      l3_in_call_ = false;
      continue;
    }
    const char* sm = nullptr;
    const size_t s = find_earliest(
        buf, pos, {kToolCallStartToken, kToolCallStartAltToken}, &sm);
    if (s == std::string::npos || s >= f) {
      if (f > pos) out_e_ += buf.substr(pos, f - pos);
      pos = std::max(pos, f);
      break;
    }
    out_e_ += buf.substr(pos, s - pos);
    pos = s + std::string(sm).size();
    l3_in_call_ = true;
  }
  l3_pend_ = buf.substr(pos);
}

StreamDeltas ToolCallStreamParser::feed(const std::string& raw_piece) {
  raw_ += raw_piece;
  l1_drive();
  StreamDeltas d = emit_deltas();
  // Robust to mixed use: anything stashed by earlier push() calls is older
  // and goes first. (Server code uses either feed() or push()/poll()
  // exclusively, so the stash is normally empty here.)
  if (!stash_r_.empty()) {
    d.reasoning = stash_r_ + d.reasoning;
    stash_r_.clear();
  }
  return d;
}

std::string ToolCallStreamParser::push(const std::string& raw_piece) {
  raw_ += raw_piece;
  l1_drive();
  StreamDeltas d = emit_deltas();
  stash_r_ += d.reasoning;
  return d.display;
}

// Layer 4: the stop search advances a cursor over the append-only outside
// stream (a stop starting before the cursor minus the longest stop is
// already decided, so only the new tail plus a bounded overlap is
// searched); emission then holds back trailing partial UTF-8,
// control-marker heads, and partial custom stop heads, exactly as the
// previous whole-region implementation did, but computed at the frontier
// in bounded work instead of rescanning the turn.
StreamDeltas ToolCallStreamParser::emit_deltas() {
  StreamDeltas out;
  if (!l4_stopped_) {
    const size_t c =
        find_stop_string(out_e_.substr(l4_stop_search_), stops_);
    if (c != std::string::npos) {
      l4_stopped_ = true;
      l4_cut_ = l4_stop_search_ + c;
    } else {
      const size_t n = out_e_.size();
      l4_stop_search_ = (max_stop_len_ > 0 && n >= max_stop_len_ - 1)
                            ? n - (max_stop_len_ - 1)
                            : 0;
    }
  }
  // Display delta: cut at the earliest complete stop, then hold back what
  // could still complete into something retractable. A completed stop only
  // ever cuts held bytes, so the emitted view stays a prefix of the
  // whole-text result.
  {
    const size_t firm = l4_stopped_ ? l4_cut_ : out_e_.size();
    size_t safe = firm;
    const size_t hold_utf8 = stream_trailing_partial_utf8(out_e_, safe);
    if (hold_utf8 > 0 && hold_utf8 <= safe) safe -= hold_utf8;
    // Single longest hold-back (not a fixpoint): only the suffix ending at
    // safe can still complete with future bytes. An earlier '<' followed
    // by '<' is unambiguous content, so holding it would both drop
    // content ("hello <<" -> "hello ") and rescan an unbounded run on
    // every push (O(N^2)). Bounded by the longest marker/stop prefix.
    size_t hold = 0;
    if (safe > emitted_) {
      for (const char* s : kHoldMarkers) {
        const size_t slen = std::char_traits<char>::length(s);
        if (slen <= 1) continue;
        if (safe <= emitted_) break;
        const size_t max_k = std::min({safe - emitted_, slen - 1, safe});
        for (size_t k = max_k; k >= 1; --k) {
          if (out_e_.compare(safe - k, k, s, 0, k) == 0) {
            hold = std::max(hold, k);
            break;
          }
          if (k == 1) break;
        }
      }
      // Same hold-back for caller stop strings (proper prefixes only; a
      // complete stop was already cut above).
      for (const auto& stop : stops_) {
        if (stop.size() <= 1) continue;
        if (safe <= emitted_) break;
        const size_t max_k =
            std::min({safe - emitted_, stop.size() - 1, safe});
        for (size_t k = max_k; k >= 1; --k) {
          if (out_e_.compare(safe - k, k, stop, 0, k) == 0) {
            hold = std::max(hold, k);
            break;
          }
          if (k == 1) break;
        }
      }
      if (hold > 0 && hold <= safe - emitted_) safe -= hold;
    }
    // Defensive rebase: never substr past the end, never emit backwards.
    if (emitted_ > out_e_.size()) emitted_ = out_e_.size();
    if (emitted_ > firm) emitted_ = firm;
    if (safe < emitted_) safe = emitted_;
    out.display = out_e_.substr(emitted_, safe - emitted_);
    emitted_ = safe;
  }
  // Reasoning delta: stops do not apply to the trace (as in the
  // whole-text parse); only partial UTF-8 and control-marker heads are
  // held, so an already-streamed prefix can never retract.
  {
    size_t safe = out_r_.size();
    const size_t hold_utf8 = stream_trailing_partial_utf8(out_r_, safe);
    if (hold_utf8 > 0 && hold_utf8 <= safe) safe -= hold_utf8;
    // Single longest marker hold (same rationale as display): only the
    // suffix ending at safe can still complete.
    size_t hold = 0;
    if (safe > emitted_r_) {
      for (const char* m : kHoldMarkers) {
        const std::string s(m);
        if (s.size() <= 1) continue;
        if (safe <= emitted_r_) break;
        const size_t max_k = std::min({safe - emitted_r_, s.size() - 1, safe});
        for (size_t k = max_k; k >= 1; --k) {
          if (out_r_.compare(safe - k, k, s, 0, k) == 0) {
            hold = std::max(hold, k);
            break;
          }
          if (k == 1) break;
        }
      }
      if (hold > 0 && hold <= safe - emitted_r_) safe -= hold;
    }
    // Defensive rebase (same rationale as display): never substr past size.
    if (emitted_r_ > out_r_.size()) emitted_r_ = out_r_.size();
    if (safe < emitted_r_) safe = emitted_r_;
    out.reasoning = out_r_.substr(emitted_r_, safe - emitted_r_);
    emitted_r_ = safe;
  }
  return out;
}

std::string ToolCallStreamParser::poll_reasoning() {
  std::string d = std::move(stash_r_);
  stash_r_.clear();
  return d;
}

ToolParseResult ToolCallStreamParser::finish() {
  // Whole-text parse on the retained raw input (O(n) once); the streamed
  // deltas are prefixes of it by construction, so the tail is whatever was
  // held back. Same stop truncation as the streaming view.
  ToolParseResult full = parse_tool_calls(raw_, stops_, start_in_reasoning_);
  ToolParseResult tail;
  tail.calls = std::move(full.calls);
  // emitted_ counts chars of the outside text; the parsed text may
  // additionally end at a stop marker, so clamp defensively. Same for
  // reasoning: streamed live, the tail is whatever was held back.
  tail.text =
      (emitted_ <= full.text.size()) ? full.text.substr(emitted_) : "";
  tail.reasoning = (emitted_r_ <= full.reasoning.size())
                       ? full.reasoning.substr(emitted_r_)
                       : "";
  return tail;
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

std::vector<std::string> default_stop_strings() { return {kEosToken}; }

}  // namespace LFM25
}  // namespace KokkosDNN
