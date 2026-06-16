#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <fmt/core.h>

namespace {

struct Args {
    std::filesystem::path input;
    std::filesystem::path output;
    std::uint64_t limit = 0;
    int max_abs_cp = 1600;
    bool enyo_runtime_target = false;
};

struct Stats {
    std::uint64_t read = 0;
    std::uint64_t written = 0;
    std::uint64_t skipped_cp = 0;
    std::uint64_t skipped_bad = 0;
};

std::string_view trim(std::string_view text)
{
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t' ||
                             text.front() == '\r' || text.front() == '\n'))
        text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' ||
                             text.back() == '\r' || text.back() == '\n'))
        text.remove_suffix(1);
    return text;
}

[[noreturn]] void usage(std::string_view argv0)
{
    fmt::print(stderr,
               "usage: {} --input rows.jsonl --output bullet.txt "
               "[--limit N] [--max-abs-cp N] [--enyo-runtime-target]\n",
               argv0);
    std::exit(2);
}

std::uint64_t parse_u64(std::string_view text, std::string_view name)
{
    std::uint64_t value = 0;
    auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size())
        throw std::runtime_error(fmt::format("invalid {}: {}", name, text));
    return value;
}

int parse_i32(std::string_view text, std::string_view name)
{
    int value = 0;
    auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size())
        throw std::runtime_error(fmt::format("invalid {}: {}", name, text));
    return value;
}

std::filesystem::path expand_user(std::string_view text)
{
    std::string path(text);
    if (!path.empty() && path[0] == '~') {
        const char *home = std::getenv("HOME");
        if (!home)
            throw std::runtime_error("HOME is not set");
        if (path.size() == 1)
            path = home;
        else if (path[1] == '/')
            path = std::string(home) + path.substr(1);
    }
    return std::filesystem::path(path);
}

Args parse_args(int argc, char **argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        auto value = [&]() -> std::string_view {
            if (++i >= argc)
                usage(argv[0]);
            return argv[i];
        };

        if (key == "--input")
            args.input = expand_user(value());
        else if (key == "--output")
            args.output = expand_user(value());
        else if (key == "--limit")
            args.limit = parse_u64(value(), key);
        else if (key == "--max-abs-cp")
            args.max_abs_cp = parse_i32(value(), key);
        else if (key == "--enyo-runtime-target")
            args.enyo_runtime_target = true;
        else if (key == "--help" || key == "-h")
            usage(argv[0]);
        else
            throw std::runtime_error(fmt::format("unknown argument: {}", key));
    }

    if (args.input.empty() || args.output.empty())
        usage(argv[0]);
    if (args.max_abs_cp < 0)
        throw std::runtime_error("--max-abs-cp must be >= 0");
    return args;
}

std::string unescape_json_string(std::string_view text)
{
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }
        if (++i >= text.size())
            throw std::runtime_error("bad JSON string escape");
        switch (text[i]) {
        case '"':
        case '\\':
        case '/':
            out.push_back(text[i]);
            break;
        case 'b':
            out.push_back('\b');
            break;
        case 'f':
            out.push_back('\f');
            break;
        case 'n':
            out.push_back('\n');
            break;
        case 'r':
            out.push_back('\r');
            break;
        case 't':
            out.push_back('\t');
            break;
        default:
            throw std::runtime_error("unsupported JSON string escape");
        }
    }
    return out;
}

std::size_t field_value_start(std::string_view line, std::string_view key)
{
    const auto key_pos = line.find(key);
    if (key_pos == std::string_view::npos)
        throw std::runtime_error(fmt::format("missing {}", key));
    const auto colon = line.find(':', key_pos + key.size());
    if (colon == std::string_view::npos)
        throw std::runtime_error(fmt::format("missing colon after {}", key));
    auto pos = colon + 1;
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t'))
        ++pos;
    return pos;
}

std::string extract_string(std::string_view line, std::string_view key)
{
    auto pos = field_value_start(line, key);
    if (pos >= line.size() || line[pos] != '"')
        throw std::runtime_error(fmt::format("{} is not a string", key));
    ++pos;
    std::string raw;
    bool escaped = false;
    for (; pos < line.size(); ++pos) {
        const char ch = line[pos];
        if (!escaped && ch == '"')
            return unescape_json_string(raw);
        raw.push_back(ch);
        escaped = !escaped && ch == '\\';
        if (ch != '\\')
            escaped = false;
    }
    throw std::runtime_error(fmt::format("unterminated {}", key));
}

bool try_extract_string(std::string_view line, std::string_view key, std::string &value)
{
    try {
        value = extract_string(line, key);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool try_extract_number(std::string_view line, std::string_view key, double &value)
{
    try {
        const auto pos = field_value_start(line, key);
        const char *begin = line.data() + pos;
        const char *end = line.data() + line.size();
        char *parse_end = nullptr;
        value = std::strtod(begin, &parse_end);
        return parse_end != begin && parse_end <= end;
    } catch (const std::exception &) {
        return false;
    }
}

char side_to_move(std::string_view fen)
{
    const auto first = fen.find(' ');
    if (first == std::string_view::npos || first + 2 >= fen.size())
        throw std::runtime_error("bad FEN");
    return fen[first + 1];
}

double phase_scale_from_fen(std::string_view fen)
{
    const auto end = fen.find(' ');
    std::string_view board = end == std::string_view::npos ? fen : fen.substr(0, end);
    int minors = 0;
    int rooks = 0;
    int queens = 0;
    for (const char ch : board) {
        if (ch == 'N' || ch == 'n' || ch == 'B' || ch == 'b')
            ++minors;
        else if (ch == 'R' || ch == 'r')
            ++rooks;
        else if (ch == 'Q' || ch == 'q')
            ++queens;
    }
    const int phase = 3 * minors + 5 * rooks + 10 * queens;
    return (128.0 + static_cast<double>(phase)) / 128.0;
}

double categorical_result(double value)
{
    if (value > 0.5)
        return 1.0;
    if (value < 0.5)
        return 0.0;
    return 0.5;
}

double white_result_from_row(std::string_view line, std::string_view fen)
{
    std::string result;
    if (try_extract_string(line, "\"result\"", result)) {
        if (result == "1-0")
            return 1.0;
        if (result == "0-1")
            return 0.0;
        if (result == "1/2-1/2")
            return 0.5;
    }

    double wdl = 0.5;
    try_extract_number(line, "\"wdl\"", wdl);
    const char stm = side_to_move(fen);
    const double white_wdl = stm == 'w' ? wdl : 1.0 - wdl;
    return categorical_result(white_wdl);
}

int white_score_from_row(std::string_view line, std::string_view fen, bool enyo_runtime_target)
{
    double score_raw = 0.0;
    if (!try_extract_number(line, "\"score\"", score_raw))
        throw std::runtime_error("missing score");
    int score = static_cast<int>(std::llround(score_raw));
    if (enyo_runtime_target)
        score = static_cast<int>(std::llround(static_cast<double>(score) / phase_scale_from_fen(fen)));
    return side_to_move(fen) == 'w' ? score : -score;
}

void print_stats(const Args &args, const Stats &stats)
{
    fmt::print("{{\n");
    fmt::print("  \"input\": \"{}\",\n", args.input.string());
    fmt::print("  \"output\": \"{}\",\n", args.output.string());
    fmt::print("  \"read\": {},\n", stats.read);
    fmt::print("  \"written\": {},\n", stats.written);
    fmt::print("  \"skipped_cp\": {},\n", stats.skipped_cp);
    fmt::print("  \"skipped_bad\": {},\n", stats.skipped_bad);
    fmt::print("  \"target\": \"{}\"\n", args.enyo_runtime_target ? "enyo-runtime" : "raw-cp");
    fmt::print("}}\n");
}

}

int main(int argc, char **argv)
{
    try {
        const Args args = parse_args(argc, argv);
        std::ifstream input(args.input);
        if (!input)
            throw std::runtime_error(fmt::format("cannot open {}", args.input.string()));

        const auto parent = args.output.parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        std::ofstream output(args.output);
        if (!output)
            throw std::runtime_error(fmt::format("cannot open {}", args.output.string()));

        Stats stats;
        std::string line;
        while (std::getline(input, line)) {
            ++stats.read;
            try {
                const auto text = trim(line);
                if (text.empty())
                    continue;
                const std::string fen = extract_string(text, "\"fen\"");
                const int score = white_score_from_row(text, fen, args.enyo_runtime_target);
                if (args.max_abs_cp > 0 && std::abs(score) > args.max_abs_cp) {
                    ++stats.skipped_cp;
                    continue;
                }
                const double result = white_result_from_row(text, fen);
                output << fen << " | " << score << " | " << fmt::format("{:.1f}", result) << '\n';
                ++stats.written;
                if (args.limit > 0 && stats.written >= args.limit)
                    break;
            } catch (const std::exception &) {
                ++stats.skipped_bad;
            }
        }

        print_stats(args, stats);
        return 0;
    } catch (const std::exception &exc) {
        fmt::print(stderr, "error: {}\n", exc.what());
        return 1;
    }
}
