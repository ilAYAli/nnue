#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace {

constexpr auto NoneSquare = uint8_t{64};

enum class Color : uint8_t { White = 0, Black = 1 };
enum class PieceType : uint8_t { Pawn = 0, Knight = 1, Bishop = 2, Rook = 3, Queen = 4, King = 5, None = 6 };
enum class MoveType : uint8_t { Normal = 0, Promotion = 1, Castle = 2, EnPassant = 3 };
enum class CastleType : uint8_t { Short, Long };

constexpr auto operator!(Color color) -> Color
{
    return color == Color::White ? Color::Black : Color::White;
}

constexpr auto color_index(Color color) -> size_t
{
    return static_cast<size_t>(color);
}

constexpr auto piece_index(PieceType type) -> size_t
{
    return static_cast<size_t>(type);
}

struct Piece {
    uint8_t id = (static_cast<uint8_t>(PieceType::None) << 1) | static_cast<uint8_t>(Color::White);

    static constexpr auto none() -> Piece
    {
        return {};
    }

    static constexpr auto from_id(uint8_t id) -> Piece
    {
        return Piece{id};
    }

    static constexpr auto make(PieceType type, Color color) -> Piece
    {
        return Piece{static_cast<uint8_t>((static_cast<uint8_t>(type) << 1) | static_cast<uint8_t>(color))};
    }

    constexpr auto type() const -> PieceType
    {
        return static_cast<PieceType>(id >> 1);
    }

    constexpr auto color() const -> Color
    {
        return static_cast<Color>(id & 1);
    }

    friend constexpr auto operator==(Piece lhs, Piece rhs) -> bool = default;
};

struct Move {
    uint8_t from = NoneSquare;
    uint8_t to = NoneSquare;
    MoveType type = MoveType::Normal;
    Piece promoted = Piece::none();

    static constexpr auto normal(uint8_t from, uint8_t to) -> Move
    {
        return {from, to, MoveType::Normal, Piece::none()};
    }

    static constexpr auto en_passant(uint8_t from, uint8_t to) -> Move
    {
        return {from, to, MoveType::EnPassant, Piece::none()};
    }

    static constexpr auto promotion(uint8_t from, uint8_t to, Piece promoted) -> Move
    {
        return {from, to, MoveType::Promotion, promoted};
    }

    static constexpr auto castle(uint8_t from, uint8_t to) -> Move
    {
        return {from, to, MoveType::Castle, Piece::none()};
    }

    static constexpr auto from_castle(CastleType type, Color side) -> Move
    {
        if (type == CastleType::Short)
            return side == Color::White ? castle(4, 7) : castle(60, 63);
        return side == Color::White ? castle(4, 0) : castle(60, 56);
    }

    constexpr auto castle_type() const -> CastleType
    {
        return (to & 7) == 7 ? CastleType::Short : CastleType::Long;
    }
};

constexpr uint8_t WhiteKingSide = 0x1;
constexpr uint8_t WhiteQueenSide = 0x2;
constexpr uint8_t BlackKingSide = 0x4;
constexpr uint8_t BlackQueenSide = 0x8;
constexpr uint8_t WhiteCastling = WhiteKingSide | WhiteQueenSide;
constexpr uint8_t BlackCastling = BlackKingSide | BlackQueenSide;

constexpr auto rank_of(uint8_t square) -> uint8_t
{
    return square >> 3;
}

constexpr auto file_of(uint8_t square) -> uint8_t
{
    return square & 7;
}

constexpr auto bit(uint8_t square) -> uint64_t
{
    return uint64_t{1} << square;
}

auto read_u16_be(const uint8_t* data) -> uint16_t
{
    return static_cast<uint16_t>((uint16_t{data[0]} << 8) | uint16_t{data[1]});
}

auto read_u32_le(const uint8_t* data) -> uint32_t
{
    return uint32_t{data[0]} | (uint32_t{data[1]} << 8) | (uint32_t{data[2]} << 16) | (uint32_t{data[3]} << 24);
}

auto read_u64_be(const uint8_t* data) -> uint64_t
{
    auto value = uint64_t{0};
    for (auto i = 0; i < 8; ++i)
        value = (value << 8) | data[i];
    return value;
}

auto unsigned_to_signed(uint16_t value) -> int16_t
{
    auto rotated = static_cast<uint16_t>((value >> 1) | (value << 15));
    if ((rotated & 0x8000) != 0)
        rotated ^= 0x7fff;
    return static_cast<int16_t>(rotated);
}

auto wrap_add_i16(int16_t lhs, int16_t rhs) -> int16_t
{
    return static_cast<int16_t>(static_cast<uint16_t>(lhs) + static_cast<uint16_t>(rhs));
}

auto wrap_neg_i16(int16_t value) -> int16_t
{
    return static_cast<int16_t>(uint16_t{0} - static_cast<uint16_t>(value));
}

auto abs_i16(int16_t value) -> uint32_t
{
    return value < 0 ? static_cast<uint32_t>(-static_cast<int32_t>(value)) : static_cast<uint32_t>(value);
}

auto used_bits_safe(uint64_t value) -> size_t
{
    if (value == 0)
        return 0;
    return 64 - static_cast<size_t>(std::countl_zero(value - 1));
}

auto nth_set_bit(uint64_t value, uint64_t index) -> uint8_t
{
    while (index-- > 0)
        value &= value - 1;
    return static_cast<uint8_t>(std::countr_zero(value));
}

auto pop_lsb(uint64_t& value) -> uint8_t
{
    auto square = static_cast<uint8_t>(std::countr_zero(value));
    value &= value - 1;
    return square;
}

auto pawn_attacks(Color color, uint8_t square) -> uint64_t
{
    auto attacks = uint64_t{0};
    const auto file = file_of(square);
    const auto rank = rank_of(square);

    if (color == Color::White) {
        if (rank < 7 && file > 0)
            attacks |= bit(static_cast<uint8_t>(square + 7));
        if (rank < 7 && file < 7)
            attacks |= bit(static_cast<uint8_t>(square + 9));
    } else {
        if (rank > 0 && file > 0)
            attacks |= bit(static_cast<uint8_t>(square - 9));
        if (rank > 0 && file < 7)
            attacks |= bit(static_cast<uint8_t>(square - 7));
    }

    return attacks;
}

auto step_attacks(uint8_t square, const std::array<std::pair<int, int>, 8>& offsets) -> uint64_t
{
    auto attacks = uint64_t{0};
    const auto file = static_cast<int>(file_of(square));
    const auto rank = static_cast<int>(rank_of(square));

    for (const auto& [df, dr] : offsets) {
        const auto next_file = file + df;
        const auto next_rank = rank + dr;
        if (next_file >= 0 && next_file < 8 && next_rank >= 0 && next_rank < 8)
            attacks |= bit(static_cast<uint8_t>(next_rank * 8 + next_file));
    }

    return attacks;
}

auto knight_attacks(uint8_t square) -> uint64_t
{
    static constexpr auto offsets = std::array{
        std::pair{-2, -1}, std::pair{-1, -2}, std::pair{1, -2}, std::pair{2, -1},
        std::pair{2, 1}, std::pair{1, 2}, std::pair{-1, 2}, std::pair{-2, 1},
    };
    return step_attacks(square, offsets);
}

auto king_attacks(uint8_t square) -> uint64_t
{
    static constexpr auto offsets = std::array{
        std::pair{-1, -1}, std::pair{0, -1}, std::pair{1, -1}, std::pair{-1, 0},
        std::pair{1, 0}, std::pair{-1, 1}, std::pair{0, 1}, std::pair{1, 1},
    };
    return step_attacks(square, offsets);
}

auto ray_attacks(uint8_t square, uint64_t occupied, int df, int dr) -> uint64_t
{
    auto attacks = uint64_t{0};
    auto file = static_cast<int>(file_of(square)) + df;
    auto rank = static_cast<int>(rank_of(square)) + dr;

    while (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
        const auto target = static_cast<uint8_t>(rank * 8 + file);
        attacks |= bit(target);
        if ((occupied & bit(target)) != 0)
            break;
        file += df;
        rank += dr;
    }

    return attacks;
}

auto bishop_attacks(uint8_t square, uint64_t occupied) -> uint64_t
{
    return ray_attacks(square, occupied, 1, 1) | ray_attacks(square, occupied, 1, -1)
        | ray_attacks(square, occupied, -1, 1) | ray_attacks(square, occupied, -1, -1);
}

auto rook_attacks(uint8_t square, uint64_t occupied) -> uint64_t
{
    return ray_attacks(square, occupied, 1, 0) | ray_attacks(square, occupied, -1, 0)
        | ray_attacks(square, occupied, 0, 1) | ray_attacks(square, occupied, 0, -1);
}

auto piece_attacks(PieceType type, uint8_t square, uint64_t occupied) -> uint64_t
{
    switch (type) {
    case PieceType::Knight:
        return knight_attacks(square);
    case PieceType::Bishop:
        return bishop_attacks(square, occupied);
    case PieceType::Rook:
        return rook_attacks(square, occupied);
    case PieceType::Queen:
        return bishop_attacks(square, occupied) | rook_attacks(square, occupied);
    case PieceType::King:
        return king_attacks(square);
    default:
        throw std::runtime_error("invalid piece type for attacks");
    }
}

struct Position {
    std::array<uint64_t, 6> bb{};
    std::array<uint64_t, 2> bb_color{};
    std::array<Piece, 64> pieces{};
    Color stm = Color::White;
    uint8_t castling_rights = 0;
    uint8_t halfmove = 0;
    uint16_t fullmove = 1;
    uint8_t ep_square = NoneSquare;

    Position()
    {
        pieces.fill(Piece::none());
    }

    auto occupied() const -> uint64_t
    {
        return bb_color[0] | bb_color[1];
    }

    auto pieces_of(Color color) const -> uint64_t
    {
        return bb_color[color_index(color)];
    }

    auto pieces_of(Color color, PieceType type) const -> uint64_t
    {
        return bb_color[color_index(color)] & bb[piece_index(type)];
    }

    auto piece_at(uint8_t square) const -> Piece
    {
        if (square >= 64)
            return Piece::none();
        return pieces[square];
    }

    void place_piece(Piece piece, uint8_t square)
    {
        const auto mask = bit(square);
        bb_color[color_index(piece.color())] |= mask;
        bb[piece_index(piece.type())] |= mask;
        pieces[square] = piece;
    }

    void remove_piece_type(Color color, PieceType type, uint8_t square)
    {
        const auto mask = bit(square);
        bb_color[color_index(color)] ^= mask;
        bb[piece_index(type)] ^= mask;
        pieces[square] = Piece::none();
    }

    auto king_square(Color color) const -> uint8_t
    {
        const auto kings = pieces_of(color, PieceType::King);
        if (kings == 0)
            return NoneSquare;
        return static_cast<uint8_t>(std::countr_zero(kings));
    }

    auto is_attacked(uint8_t square, Color attacker) const -> bool
    {
        if (square >= 64)
            return false;

        const auto occ = occupied();
        return ((pawn_attacks(!attacker, square) & pieces_of(attacker, PieceType::Pawn))
                   | (knight_attacks(square) & pieces_of(attacker, PieceType::Knight))
                   | (king_attacks(square) & pieces_of(attacker, PieceType::King))
                   | (bishop_attacks(square, occ) & (pieces_of(attacker, PieceType::Bishop) | pieces_of(attacker, PieceType::Queen)))
                   | (rook_attacks(square, occ) & (pieces_of(attacker, PieceType::Rook) | pieces_of(attacker, PieceType::Queen))))
            != 0;
    }

    auto is_checked(Color color) const -> bool
    {
        return is_attacked(king_square(color), !color);
    }

    void update_castling_rights(Color color, uint8_t from, uint8_t to)
    {
        if (color == Color::White) {
            if (from == 4 || to == 4)
                castling_rights &= static_cast<uint8_t>(~WhiteCastling);
            if (from == 0 || to == 0)
                castling_rights &= static_cast<uint8_t>(~WhiteQueenSide);
            if (from == 7 || to == 7)
                castling_rights &= static_cast<uint8_t>(~WhiteKingSide);
        } else {
            if (from == 60 || to == 60)
                castling_rights &= static_cast<uint8_t>(~BlackCastling);
            if (from == 56 || to == 56)
                castling_rights &= static_cast<uint8_t>(~BlackQueenSide);
            if (from == 63 || to == 63)
                castling_rights &= static_cast<uint8_t>(~BlackKingSide);
        }
    }

    void do_move(Move move)
    {
        const auto from = move.from;
        const auto to = move.to;
        const auto piece = piece_at(from);
        const auto type = piece.type();

        remove_piece_type(stm, type, from);

        if (move.type != MoveType::Castle) {
            const auto captured = piece_at(to);
            if (captured != Piece::none()) {
                const auto captured_type = captured.type();
                remove_piece_type(!stm, captured_type, to);
                if (captured_type == PieceType::Rook)
                    update_castling_rights(!stm, from, to);
                halfmove = 0;
            }
        }

        if (type == PieceType::King || type == PieceType::Rook)
            update_castling_rights(stm, from, to);

        if (move.type == MoveType::Promotion) {
            place_piece(move.promoted, to);
        } else if (move.type == MoveType::EnPassant) {
            const auto captured_square = static_cast<uint8_t>(to ^ 8);
            remove_piece_type(!stm, PieceType::Pawn, captured_square);
            place_piece(piece, to);
        } else if (move.type == MoveType::Normal) {
            place_piece(piece, to);
        } else if (move.type == MoveType::Castle) {
            if (move.castle_type() == CastleType::Short) {
                const auto rook_to = stm == Color::White ? uint8_t{5} : uint8_t{61};
                const auto king_to = stm == Color::White ? uint8_t{6} : uint8_t{62};
                const auto rook = piece_at(to);
                remove_piece_type(stm, PieceType::Rook, to);
                place_piece(rook, rook_to);
                place_piece(piece, king_to);
            } else {
                const auto rook_to = stm == Color::White ? uint8_t{3} : uint8_t{59};
                const auto king_to = stm == Color::White ? uint8_t{2} : uint8_t{58};
                const auto rook = piece_at(to);
                remove_piece_type(stm, PieceType::Rook, to);
                place_piece(rook, rook_to);
                place_piece(piece, king_to);
            }
        }

        if (type == PieceType::Pawn)
            halfmove = 0;
        else
            ++halfmove;

        if (stm == Color::Black)
            ++fullmove;

        ep_square = NoneSquare;

        if (type == PieceType::Pawn && std::abs(static_cast<int>(to) - static_cast<int>(from)) == 16) {
            const auto ep = static_cast<uint8_t>(to ^ 8);
            auto enemy_pawns = pawn_attacks(stm, ep) & pieces_of(!stm, PieceType::Pawn);

            while (enemy_pawns != 0) {
                const auto enemy_square = pop_lsb(enemy_pawns);
                const auto enemy_pawn = piece_at(enemy_square);

                remove_piece_type(!stm, PieceType::Pawn, enemy_square);
                place_piece(enemy_pawn, ep);
                remove_piece_type(stm, PieceType::Pawn, to);

                const auto checked = is_checked(!stm);

                place_piece(enemy_pawn, enemy_square);
                remove_piece_type(!stm, PieceType::Pawn, ep);
                place_piece(piece, to);

                if (!checked) {
                    ep_square = ep;
                    break;
                }
            }
        }

        stm = !stm;
    }
};

auto decompress_position(const uint8_t* data) -> Position
{
    auto pos = Position{};
    const auto occupied = read_u64_be(data);
    const auto* packed = data + 8;
    auto remaining = occupied;
    auto nibble_index = size_t{0};

    while (remaining != 0) {
        const auto square = pop_lsb(remaining);
        const auto packed_byte = packed[nibble_index / 2];
        const auto nibble = static_cast<uint8_t>((nibble_index % 2) == 0 ? packed_byte & 0xf : packed_byte >> 4);

        if (nibble <= 11) {
            pos.place_piece(Piece::from_id(nibble), square);
        } else if (nibble == 12) {
            if (rank_of(square) == 3) {
                pos.place_piece(Piece::make(PieceType::Pawn, Color::White), square);
                pos.ep_square = static_cast<uint8_t>(square - 8);
            } else {
                pos.place_piece(Piece::make(PieceType::Pawn, Color::Black), square);
                pos.ep_square = static_cast<uint8_t>(square + 8);
            }
        } else if (nibble == 13) {
            pos.place_piece(Piece::make(PieceType::Rook, Color::White), square);
            pos.castling_rights |= square == 0 ? WhiteQueenSide : WhiteKingSide;
        } else if (nibble == 14) {
            pos.place_piece(Piece::make(PieceType::Rook, Color::Black), square);
            pos.castling_rights |= square == 56 ? BlackQueenSide : BlackKingSide;
        } else {
            pos.place_piece(Piece::make(PieceType::King, Color::Black), square);
            pos.stm = Color::Black;
        }

        ++nibble_index;
    }

    return pos;
}

auto decompress_move(const uint8_t* data) -> Move
{
    const auto packed = read_u16_be(data);
    if (packed == 0)
        return {};

    const auto type = static_cast<MoveType>(packed >> 14);
    const auto from = static_cast<uint8_t>((packed >> 8) & 0x3f);
    const auto to = static_cast<uint8_t>((packed >> 2) & 0x3f);
    auto promoted = Piece::none();

    if (type == MoveType::Promotion) {
        const auto color = rank_of(to) == 0 ? Color::Black : Color::White;
        const auto promoted_type = static_cast<PieceType>(static_cast<uint8_t>(PieceType::Knight) + (packed & 0x3));
        promoted = Piece::make(promoted_type, color);
    }

    return {from, to, type, promoted};
}

struct Entry {
    Position pos;
    Move move;
    int16_t score = 0;
    uint16_t ply = 0;
    int16_t result = 0;
};

auto unpack_entry(const uint8_t* data) -> Entry
{
    auto entry = Entry{};
    entry.pos = decompress_position(data);
    entry.move = decompress_move(data + 24);
    entry.score = unsigned_to_signed(read_u16_be(data + 26));

    const auto ply_result = read_u16_be(data + 28);
    entry.ply = static_cast<uint16_t>(ply_result & 0x3fff);
    entry.result = unsigned_to_signed(static_cast<uint16_t>(ply_result >> 14));
    entry.pos.fullmove = static_cast<uint16_t>((entry.ply / 2) + 1);
    entry.pos.halfmove = static_cast<uint8_t>(read_u16_be(data + 30));

    return entry;
}

struct BitReader {
    size_t read_bits_left = 8;
    size_t read_offset = 0;

    auto extract_bits_le8(const uint8_t* data, size_t size, size_t count) -> uint8_t
    {
        if (count == 0)
            return 0;
        if (count > 8)
            throw std::runtime_error("bit count larger than 8");

        if (read_bits_left == 0) {
            ++read_offset;
            read_bits_left = 8;
        }
        if (read_offset >= size)
            throw std::runtime_error("movetext overread");

        const auto byte = static_cast<uint8_t>(data[read_offset] << (8 - read_bits_left));
        auto bits = static_cast<uint8_t>(byte >> (8 - count));

        if (count > read_bits_left) {
            const auto spill_count = count - read_bits_left;
            if (read_offset + 1 >= size)
                throw std::runtime_error("movetext spill overread");
            bits |= static_cast<uint8_t>(data[read_offset + 1] >> (8 - spill_count));
            read_bits_left += 8;
            ++read_offset;
        }

        read_bits_left -= count;
        return bits;
    }

    auto extract_vle16(const uint8_t* data, size_t size, size_t block_size) -> uint16_t
    {
        const auto mask = static_cast<uint16_t>((1u << block_size) - 1u);
        auto value = uint16_t{0};
        auto offset = size_t{0};

        while (true) {
            const auto block = static_cast<uint16_t>(extract_bits_le8(data, size, block_size + 1));
            value |= static_cast<uint16_t>((block & mask) << offset);
            if ((block >> block_size) == 0)
                break;
            offset += block_size;
            if (offset >= 16)
                throw std::runtime_error("vle16 overread");
        }

        return value;
    }

    auto num_read_bytes() const -> size_t
    {
        return read_offset + (read_bits_left != 8 ? 1 : 0);
    }
};

struct MoveScoreReader {
    BitReader reader;
    int16_t last_score = 0;
    Entry entry;

    explicit MoveScoreReader(Entry stem)
        : last_score(wrap_neg_i16(stem.score)), entry(std::move(stem))
    {
    }

    auto next(const uint8_t* movetext, size_t size) -> Entry
    {
        entry.pos.do_move(entry.move);
        auto [move, score] = next_move_score(movetext, size);
        entry.move = move;
        entry.score = score;
        ++entry.ply;
        entry.result = wrap_neg_i16(entry.result);
        return entry;
    }

    auto next_move_score(const uint8_t* movetext, size_t size) -> std::pair<Move, int16_t>
    {
        const auto side = entry.pos.stm;
        const auto our_pieces = entry.pos.pieces_of(side);
        const auto their_pieces = entry.pos.pieces_of(!side);
        const auto occupied = our_pieces | their_pieces;
        const auto piece_id = reader.extract_bits_le8(movetext, size, used_bits_safe(std::popcount(our_pieces)));
        const auto from = nth_set_bit(our_pieces, piece_id);
        const auto type = entry.pos.piece_at(from).type();
        const auto move = decode_move(movetext, size, type, from, occupied);
        const auto score = decode_score(movetext, size);
        last_score = wrap_neg_i16(score);
        return {move, score};
    }

    auto decode_score(const uint8_t* movetext, size_t size) -> int16_t
    {
        const auto delta = unsigned_to_signed(reader.extract_vle16(movetext, size, 4));
        return wrap_add_i16(last_score, delta);
    }

    auto decode_move(const uint8_t* movetext, size_t size, PieceType type, uint8_t from, uint64_t occupied) -> Move
    {
        const auto side = entry.pos.stm;
        const auto our_pieces = entry.pos.pieces_of(side);

        if (type == PieceType::Pawn) {
            const auto promotion_rank = side == Color::White ? uint8_t{6} : uint8_t{1};
            const auto start_rank = side == Color::White ? uint8_t{1} : uint8_t{6};
            const auto forward = side == Color::White ? 8 : -8;
            const auto ep = entry.pos.ep_square;
            auto attack_targets = entry.pos.pieces_of(!side);
            if (ep != NoneSquare)
                attack_targets |= bit(ep);

            auto destinations = pawn_attacks(side, from) & attack_targets;
            const auto one_forward = static_cast<uint8_t>(static_cast<int>(from) + forward);
            if ((occupied & bit(one_forward)) == 0) {
                destinations |= bit(one_forward);
                if (rank_of(from) == start_rank) {
                    const auto two_forward = static_cast<uint8_t>(static_cast<int>(one_forward) + forward);
                    if ((occupied & bit(two_forward)) == 0)
                        destinations |= bit(two_forward);
                }
            }

            const auto destination_count = std::popcount(destinations);
            if (rank_of(from) == promotion_rank) {
                const auto move_id = reader.extract_bits_le8(movetext, size, used_bits_safe(destination_count * 4u));
                const auto promoted_type = static_cast<PieceType>(static_cast<uint8_t>(PieceType::Knight) + (move_id % 4));
                const auto to = nth_set_bit(destinations, move_id / 4);
                return Move::promotion(from, to, Piece::make(promoted_type, side));
            }

            const auto move_id = reader.extract_bits_le8(movetext, size, used_bits_safe(destination_count));
            const auto to = nth_set_bit(destinations, move_id);
            return to == ep ? Move::en_passant(from, to) : Move::normal(from, to);
        }

        if (type == PieceType::King) {
            const auto castling_mask = side == Color::White ? WhiteCastling : BlackCastling;
            const auto attacks = king_attacks(from) & ~our_pieces;
            const auto attacks_size = std::popcount(attacks);
            const auto num_castlings = std::popcount(static_cast<uint8_t>(entry.pos.castling_rights & castling_mask));
            const auto move_count = attacks_size + num_castlings;
            const auto move_id = reader.extract_bits_le8(movetext, size, used_bits_safe(move_count));

            if (move_id >= attacks_size) {
                const auto castle_index = move_id - attacks_size;
                const auto has_long = (entry.pos.castling_rights & (side == Color::White ? WhiteQueenSide : BlackQueenSide)) != 0;
                const auto castle_type = castle_index == 0 && has_long ? CastleType::Long : CastleType::Short;
                return Move::from_castle(castle_type, side);
            }

            return Move::normal(from, nth_set_bit(attacks, move_id));
        }

        const auto attacks = piece_attacks(type, from, occupied) & ~our_pieces;
        const auto move_id = reader.extract_bits_le8(movetext, size, used_bits_safe(std::popcount(attacks)));
        return Move::normal(from, nth_set_bit(attacks, move_id));
    }
};

struct Options {
    std::string input;
    uint16_t min_ply = 16;
    uint32_t max_abs_cp = 10000;
    bool quiet_only = true;
    uint64_t max_seen = 0;
    uint64_t progress = 0;
};

struct Counts {
    uint64_t seen = 0;
    uint64_t kept = 0;
    uint64_t skipped_min_ply = 0;
    uint64_t skipped_score = 0;
    uint64_t skipped_non_quiet = 0;
    uint64_t skipped_in_check = 0;
    uint64_t chains = 0;
    uint64_t chunks = 0;
    uint64_t bytes_read = 0;
    int64_t normalized_score_sum = 0;
    uint64_t normalized_abs_score_sum = 0;
    long double normalized_score_sq_sum = 0.0;
    std::array<uint64_t, 6> normalized_abs_score_buckets{};
};

auto phase_scale(const Position& position) -> double
{
    auto phase = uint32_t{0};
    auto occupied = position.occupied();
    while (occupied != 0) {
        const auto type = position.piece_at(pop_lsb(occupied)).type();
        if (type == PieceType::Knight || type == PieceType::Bishop)
            phase += 3;
        else if (type == PieceType::Rook)
            phase += 5;
        else if (type == PieceType::Queen)
            phase += 10;
    }
    return (128.0 + static_cast<double>(phase)) / 128.0;
}

auto normalized_score(const Entry& entry) -> int16_t
{
    constexpr auto EvalClamp = int16_t{2045};
    const auto clamped = std::clamp(entry.score, static_cast<int16_t>(-EvalClamp), EvalClamp);
    return static_cast<int16_t>(std::round(static_cast<double>(clamped) / phase_scale(entry.pos)));
}

auto score_bucket(uint32_t absolute_score) -> size_t
{
    if (absolute_score < 50)
        return 0;
    if (absolute_score < 100)
        return 1;
    if (absolute_score < 300)
        return 2;
    if (absolute_score < 800)
        return 3;
    if (absolute_score < 1600)
        return 4;
    return 5;
}

auto parse_u64(std::string_view text, std::string_view name) -> uint64_t
{
    auto value = uint64_t{0};
    const auto* begin = text.data();
    const auto* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end)
        throw std::runtime_error(fmt::format("invalid {}: {}", name, text));
    return value;
}

auto parse_bool(std::string_view text, std::string_view name) -> bool
{
    if (text == "1" || text == "true" || text == "yes")
        return true;
    if (text == "0" || text == "false" || text == "no")
        return false;
    throw std::runtime_error(fmt::format("invalid {}: {}", name, text));
}

void print_usage()
{
    fmt::print("usage: count_binpack [options] <file.binpack>\n"
               "\n"
               "options:\n"
               "  --min-ply N        default 16\n"
               "  --max-abs-cp N     default 10000\n"
               "  --quiet-only 0|1   default 1\n"
               "  --max-seen N       stop after N decoded positions; 0 means no cap\n"
               "  --progress N       print progress every N decoded positions; 0 disables it\n");
}

auto parse_options(int argc, char** argv) -> Options
{
    auto options = Options{};

    for (auto i = 1; i < argc; ++i) {
        const auto arg = std::string_view{argv[i]};
        const auto next = [&](std::string_view name) -> std::string_view {
            if (++i >= argc)
                throw std::runtime_error(fmt::format("missing value for {}", name));
            return argv[i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg == "--min-ply") {
            options.min_ply = static_cast<uint16_t>(parse_u64(next(arg), arg));
        } else if (arg == "--max-abs-cp") {
            options.max_abs_cp = static_cast<uint32_t>(parse_u64(next(arg), arg));
        } else if (arg == "--quiet-only") {
            options.quiet_only = parse_bool(next(arg), arg);
        } else if (arg == "--max-seen") {
            options.max_seen = parse_u64(next(arg), arg);
        } else if (arg == "--progress") {
            options.progress = parse_u64(next(arg), arg);
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error(fmt::format("unknown option: {}", arg));
        } else if (options.input.empty()) {
            options.input = std::string{arg};
        } else {
            throw std::runtime_error("only one input file is supported");
        }
    }

    if (options.input.empty())
        throw std::runtime_error("missing input file");
    return options;
}

auto keep_entry(const Entry& entry, const Options& options, Counts& counts) -> bool
{
    if (entry.ply < options.min_ply) {
        ++counts.skipped_min_ply;
        return false;
    }

    if (abs_i16(entry.score) > options.max_abs_cp) {
        ++counts.skipped_score;
        return false;
    }

    if (options.quiet_only) {
        if (entry.move.type != MoveType::Normal || entry.move.to >= 64 || entry.pos.piece_at(entry.move.to).type() != PieceType::None) {
            ++counts.skipped_non_quiet;
            return false;
        }
    }

    if (entry.pos.is_checked(entry.pos.stm)) {
        ++counts.skipped_in_check;
        return false;
    }

    ++counts.kept;
    const auto normalized = normalized_score(entry);
    const auto absolute = abs_i16(normalized);
    counts.normalized_score_sum += normalized;
    counts.normalized_abs_score_sum += absolute;
    counts.normalized_score_sq_sum += static_cast<long double>(normalized) * normalized;
    ++counts.normalized_abs_score_buckets[score_bucket(absolute)];
    return true;
}

auto should_stop(const Options& options, const Counts& counts) -> bool
{
    return options.max_seen != 0 && counts.seen >= options.max_seen;
}

void count_entry(const Entry& entry, const Options& options, Counts& counts)
{
    ++counts.seen;
    keep_entry(entry, options, counts);

    if (options.progress != 0 && counts.seen % options.progress == 0)
        fmt::print(stderr, "seen={} kept={}\n", counts.seen, counts.kept);
}

auto count_file(const Options& options) -> Counts
{
    constexpr auto HeaderSize = size_t{8};
    constexpr auto MaxChunkSize = uint32_t{100 * 1024 * 1024};

    auto input = std::ifstream{options.input, std::ios::binary};
    if (!input)
        throw std::runtime_error(fmt::format("failed to open {}", options.input));

    auto counts = Counts{};
    auto header = std::array<uint8_t, HeaderSize>{};
    auto chunk = std::vector<uint8_t>{};

    while (input.read(reinterpret_cast<char*>(header.data()), header.size())) {
        counts.bytes_read += header.size();
        if (!std::equal(header.begin(), header.begin() + 4, "BINP"))
            throw std::runtime_error("invalid BINP magic");

        const auto chunk_size = read_u32_le(header.data() + 4);
        if (chunk_size > MaxChunkSize)
            throw std::runtime_error("chunk size larger than 100 MiB");

        chunk.resize(chunk_size);
        input.read(reinterpret_cast<char*>(chunk.data()), chunk.size());
        if (input.gcount() != static_cast<std::streamsize>(chunk.size()))
            throw std::runtime_error("truncated chunk");
        counts.bytes_read += chunk.size();
        ++counts.chunks;

        auto offset = size_t{0};
        while (offset + 34 <= chunk.size()) {
            auto entry = unpack_entry(chunk.data() + offset);
            offset += 32;
            const auto plies = read_u16_be(chunk.data() + offset);
            offset += 2;
            ++counts.chains;

            count_entry(entry, options, counts);
            if (should_stop(options, counts))
                return counts;

            auto reader = MoveScoreReader{entry};
            const auto* movetext = chunk.data() + offset;
            const auto movetext_size = chunk.size() - offset;

            for (auto ply = uint16_t{0}; ply < plies; ++ply) {
                const auto next_entry = reader.next(movetext, movetext_size);
                count_entry(next_entry, options, counts);
                if (should_stop(options, counts))
                    return counts;
            }

            offset += reader.reader.num_read_bytes();
        }
    }

    if (!input.eof())
        throw std::runtime_error("failed while reading input");

    return counts;
}

void print_counts(const Options& options, const Counts& counts, double seconds)
{
    const auto file_bytes = std::filesystem::file_size(options.input);
    const auto rate = seconds > 0 ? static_cast<double>(counts.seen) / seconds : 0.0;

    fmt::print("input: {}\n", options.input);
    fmt::print("file_bytes: {}\n", file_bytes);
    fmt::print("bytes_read: {}\n", counts.bytes_read);
    fmt::print("chunks: {}\n", counts.chunks);
    fmt::print("chains: {}\n", counts.chains);
    fmt::print("seen: {}\n", counts.seen);
    fmt::print("kept: {}\n", counts.kept);
    fmt::print("skipped_min_ply: {}\n", counts.skipped_min_ply);
    fmt::print("skipped_score: {}\n", counts.skipped_score);
    fmt::print("skipped_non_quiet: {}\n", counts.skipped_non_quiet);
    fmt::print("skipped_in_check: {}\n", counts.skipped_in_check);
    if (counts.kept != 0) {
        const auto n = static_cast<long double>(counts.kept);
        const auto mean = static_cast<long double>(counts.normalized_score_sum) / n;
        const auto variance = std::max(0.0L, counts.normalized_score_sq_sum / n - mean * mean);
        fmt::print("normalized_score_mean: {:.3f}\n", static_cast<double>(mean));
        fmt::print("normalized_abs_score_mean: {:.3f}\n", static_cast<double>(static_cast<long double>(counts.normalized_abs_score_sum) / n));
        fmt::print("normalized_score_stdev: {:.3f}\n", static_cast<double>(std::sqrt(variance)));
        fmt::print("normalized_abs_score_buckets: {} {} {} {} {} {}\n",
            counts.normalized_abs_score_buckets[0], counts.normalized_abs_score_buckets[1],
            counts.normalized_abs_score_buckets[2], counts.normalized_abs_score_buckets[3],
            counts.normalized_abs_score_buckets[4], counts.normalized_abs_score_buckets[5]);
    }
    fmt::print("elapsed_seconds: {:.3f}\n", seconds);
    fmt::print("entries_per_second: {:.0f}\n", rate);
    if (options.max_seen != 0 && counts.seen >= options.max_seen)
        fmt::print("stopped_at_max_seen: {}\n", options.max_seen);
}

} // namespace

auto main(int argc, char** argv) -> int
{
    try {
        const auto options = parse_options(argc, argv);
        const auto start = std::chrono::steady_clock::now();
        const auto counts = count_file(options);
        const auto end = std::chrono::steady_clock::now();
        const auto seconds = std::chrono::duration<double>(end - start).count();
        print_counts(options, counts, seconds);
        return 0;
    } catch (const std::exception& error) {
        fmt::print(stderr, "error: {}\n", error.what());
        return 1;
    }
}
