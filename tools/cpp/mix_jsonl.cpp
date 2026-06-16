#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/core.h>

namespace {

struct Source {
    std::filesystem::path path;
    std::uint64_t requested = 0;
    std::uint64_t total_rows = 0;
    std::uint64_t eligible_rows = 0;
    std::uint64_t selected_rows = 0;
    std::uint64_t written = 0;
    int max_abs_cp = 0;
};

struct Args {
    std::filesystem::path output;
    std::vector<Source> sources;
    std::uint64_t seed = 1;
    std::uint64_t progress = 250'000;
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
               "usage: {} --output rows.jsonl --source PATH:ROWS[:MAX_ABS_CP] "
               "[--seed N] [--progress N]\n",
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

Source parse_source(std::string_view spec)
{
    const auto last = spec.rfind(':');
    if (last == std::string_view::npos)
        throw std::runtime_error(fmt::format("invalid source spec: {}", spec));

    const auto maybe_second = spec.substr(0, last).rfind(':');
    Source source;
    std::string_view path_text;
    if (maybe_second == std::string_view::npos) {
        path_text = spec.substr(0, last);
        source.requested = parse_u64(spec.substr(last + 1), "ROWS");
    } else {
        path_text = spec.substr(0, maybe_second);
        source.requested = parse_u64(spec.substr(maybe_second + 1, last - maybe_second - 1), "ROWS");
        source.max_abs_cp = parse_i32(spec.substr(last + 1), "MAX_ABS_CP");
        if (source.max_abs_cp < 0)
            throw std::runtime_error(fmt::format("{}: MAX_ABS_CP must be >= 0", spec));
    }
    source.path = expand_user(path_text);
    return source;
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

        if (key == "--output")
            args.output = expand_user(value());
        else if (key == "--source")
            args.sources.push_back(parse_source(value()));
        else if (key == "--seed")
            args.seed = parse_u64(value(), key);
        else if (key == "--progress")
            args.progress = parse_u64(value(), key);
        else if (key == "--unique-fen")
            throw std::runtime_error("--unique-fen is not supported by the streaming mixer");
        else if (key == "--help" || key == "-h")
            usage(argv[0]);
        else
            throw std::runtime_error(fmt::format("unknown argument: {}", key));
    }

    if (args.output.empty() || args.sources.empty())
        usage(argv[0]);
    return args;
}

bool extract_score(std::string_view line, double &score)
{
    const std::string_view key = "\"score\"";
    const auto key_pos = line.find(key);
    if (key_pos == std::string_view::npos)
        return false;
    const auto colon = line.find(':', key_pos + key.size());
    if (colon == std::string_view::npos)
        return false;

    const char *begin = line.data() + colon + 1;
    const char *end = line.data() + line.size();
    while (begin != end && (*begin == ' ' || *begin == '\t'))
        ++begin;

    char *parse_end = nullptr;
    score = std::strtod(begin, &parse_end);
    return parse_end != begin;
}

bool keep_line(std::string_view line, int max_abs_cp)
{
    if (max_abs_cp <= 0)
        return true;
    double score = 0.0;
    if (!extract_score(line, score))
        return false;
    return std::abs(score) <= static_cast<double>(max_abs_cp);
}

void count_source(Source &source)
{
    std::ifstream input(source.path);
    if (!input)
        throw std::runtime_error(fmt::format("cannot open {}", source.path.string()));

    std::string line;
    while (std::getline(input, line)) {
        const auto text = trim(line);
        if (text.empty())
            continue;
        ++source.total_rows;
        if (keep_line(text, source.max_abs_cp))
            ++source.eligible_rows;
    }
    source.selected_rows = std::min(source.requested, source.eligible_rows);
}

std::string json_string(std::string_view text)
{
    std::string out;
    out.reserve(text.size() + 2);
    out.push_back('"');
    for (const char ch : text) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(ch);
            break;
        }
    }
    out.push_back('"');
    return out;
}

class SelectedLineReader {
public:
    SelectedLineReader(const Source &source, std::uint64_t seed)
        : source_(source),
          input_(source.path),
          rng_(seed),
          remaining_available_(source.eligible_rows),
          remaining_needed_(source.selected_rows)
    {
        if (!input_)
            throw std::runtime_error(fmt::format("cannot open {}", source.path.string()));
    }

    std::string next()
    {
        std::string line;
        while (std::getline(input_, line)) {
            const auto text = trim(line);
            if (text.empty() || !keep_line(text, source_.max_abs_cp))
                continue;
            if (remaining_needed_ == 0)
                break;

            std::uniform_int_distribution<std::uint64_t> dist(0, remaining_available_ - 1);
            const bool take = dist(rng_) < remaining_needed_;
            --remaining_available_;
            if (!take)
                continue;

            --remaining_needed_;
            return std::string(text);
        }
        throw std::runtime_error(fmt::format("source exhausted unexpectedly: {}", source_.path.string()));
    }

private:
    const Source &source_;
    std::ifstream input_;
    std::mt19937_64 rng_;
    std::uint64_t remaining_available_;
    std::uint64_t remaining_needed_;
};

void print_stats(const Args &args, const std::vector<Source> &sources, std::uint64_t written)
{
    fmt::print("{{\n");
    fmt::print("  \"output\": {},\n", json_string(args.output.string()));
    fmt::print("  \"seed\": {},\n", args.seed);
    fmt::print("  \"streaming\": true,\n");
    fmt::print("  \"tool\": \"nnue-mix-jsonl\",\n");
    fmt::print("  \"sources\": [\n");
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const auto &source = sources[i];
        fmt::print("    {{\"path\": {}, \"requested\": {}, \"total_rows\": {}, \"eligible_rows\": {}, "
                   "\"filtered_rows\": {}, \"selected_rows\": {}, \"max_abs_cp\": {}, "
                   "\"written\": {}}}{}",
                   json_string(source.path.string()),
                   source.requested,
                   source.total_rows,
                   source.eligible_rows,
                   source.total_rows - source.eligible_rows,
                   source.selected_rows,
                   source.max_abs_cp,
                   source.written,
                   i + 1 == sources.size() ? "\n" : ",\n");
    }
    fmt::print("  ],\n");
    fmt::print("  \"written_rows\": {}\n", written);
    fmt::print("}}\n");
}

}

int main(int argc, char **argv)
{
    try {
        Args args = parse_args(argc, argv);
        for (auto &source : args.sources)
            count_source(source);

        const auto parent = args.output.parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        std::ofstream output(args.output);
        if (!output)
            throw std::runtime_error(fmt::format("cannot open {}", args.output.string()));

        std::mt19937_64 master(args.seed);
        std::vector<SelectedLineReader> readers;
        readers.reserve(args.sources.size());
        for (const auto &source : args.sources)
            readers.emplace_back(source, master());

        std::vector<std::uint64_t> remaining;
        remaining.reserve(args.sources.size());
        std::uint64_t total_remaining = 0;
        for (const auto &source : args.sources) {
            remaining.push_back(source.selected_rows);
            total_remaining += source.selected_rows;
        }

        std::uint64_t written = 0;
        while (total_remaining > 0) {
            std::uniform_int_distribution<std::uint64_t> dist(0, total_remaining - 1);
            const std::uint64_t pick = dist(master);
            std::uint64_t cumulative = 0;
            std::size_t source_index = 0;
            for (std::size_t i = 0; i < remaining.size(); ++i) {
                cumulative += remaining[i];
                if (pick < cumulative) {
                    source_index = i;
                    break;
                }
            }

            output << readers[source_index].next() << '\n';
            --remaining[source_index];
            --total_remaining;
            ++args.sources[source_index].written;
            ++written;
            if (args.progress > 0 && written % args.progress == 0)
                fmt::print("mixed {}\n", written);
        }

        print_stats(args, args.sources, written);
        return 0;
    } catch (const std::exception &exc) {
        fmt::print(stderr, "error: {}\n", exc.what());
        return 1;
    }
}
