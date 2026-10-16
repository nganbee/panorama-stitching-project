// File: v0_stage34.cpp
// Authors: Nguyen Duy Khanh
// Description: V0 (sequential) Stage 3 BRIEF descriptors and Stage 4
//              brute-force Hamming matching with Lowe ratio test and
//              mutual-nearest-neighbour cross-check.
// External sources: none.
#include "../core_headers/common.h"
#include "../core_headers/v0_stages.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <vector>

namespace {

struct BriefPair {
    int ax, ay;
    int bx, by;
};

int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Deterministically generate the same 256 comparison pairs for every run.
// The fixed seed is part of the V0 contract so descriptors remain comparable
// across V0/V1/V2/V3.
const std::array<BriefPair, 256>& brief_pattern() {
    static const std::array<BriefPair, 256> pattern = [] {
        std::array<BriefPair, 256> p{};
        uint32_t state = params::kBriefSeed;
        auto next_u32 = [&]() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return state;
        };
        const int radius = params::kPatchSize / 2;
        for (auto& pair : p) {
            pair.ax = static_cast<int>(next_u32() % (2 * radius + 1)) - radius;
            pair.ay = static_cast<int>(next_u32() % (2 * radius + 1)) - radius;
            pair.bx = static_cast<int>(next_u32() % (2 * radius + 1)) - radius;
            pair.by = static_cast<int>(next_u32() % (2 * radius + 1)) - radius;
        }
        return p;
    }();
    return pattern;
}

uint64_t hamming_distance(const Descriptor& a, const Descriptor& b) {
    uint64_t x = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t v = a.bits[i] ^ b.bits[i];
        while (v != 0) {
            v &= v - 1;
            ++x;
        }
    }
    return x;
}

int nearest_descriptor(const std::vector<Descriptor>& query,
                       int query_index,
                       const std::vector<Descriptor>& candidates,
                       int& best_distance) {
    int best_index = -1;
    best_distance = 257;
    for (int j = 0; j < static_cast<int>(candidates.size()); ++j) {
        const int d = static_cast<int>(hamming_distance(query[query_index], candidates[j]));
        if (d < best_distance || (d == best_distance && j < best_index)) {
            best_distance = d;
            best_index = j;
        }
    }
    return best_index;
}

}  // namespace

void v0_describe(const Gray& blurred,
                 const std::vector<Keypoint>& kps,
                 std::vector<Descriptor>& desc) {
    desc.assign(kps.size(), Descriptor{});
    if (!valid_gray_image(blurred)) {
        desc.clear();
        return;
    }
    const auto& pattern = brief_pattern();

    for (size_t i = 0; i < kps.size(); ++i) {
        if (!std::isfinite(kps[i].x) || !std::isfinite(kps[i].y) ||
            kps[i].x < 0 || kps[i].x >= blurred.w ||
            kps[i].y < 0 || kps[i].y >= blurred.h) {
            desc.clear();
            return;
        }
        const int cx = static_cast<int>(kps[i].x);
        const int cy = static_cast<int>(kps[i].y);
        for (int bit = 0; bit < 256; ++bit) {
            const auto& pair = pattern[bit];
            const int ax = clampi(cx + pair.ax, 0, blurred.w - 1);
            const int ay = clampi(cy + pair.ay, 0, blurred.h - 1);
            const int bx = clampi(cx + pair.bx, 0, blurred.w - 1);
            const int by = clampi(cy + pair.by, 0, blurred.h - 1);
            const float va = blurred.data[static_cast<size_t>(ay) * blurred.w + ax];
            const float vb = blurred.data[static_cast<size_t>(by) * blurred.w + bx];
            if (va < vb) {
                desc[i].bits[bit / 64] |= uint64_t{1} << (bit % 64);
            }
        }
    }
}

void v0_match(const std::vector<Descriptor>& dA,
              const std::vector<Descriptor>& dB,
              std::vector<Match>& matches) {
    matches.clear();
    if (dA.empty() || dB.empty()) return;

    std::vector<int> best_a_for_b(dB.size(), -1);
    std::vector<int> best_dist_for_b(dB.size(), 257);
    for (int j = 0; j < static_cast<int>(dB.size()); ++j) {
        best_a_for_b[j] = nearest_descriptor(dB, j, dA, best_dist_for_b[j]);
    }

    for (int i = 0; i < static_cast<int>(dA.size()); ++i) {
        int best_distance = 257;
        const int best_index = nearest_descriptor(dA, i, dB, best_distance);
        if (best_index < 0) continue;

        int second_distance = 257;
        for (int j = 0; j < static_cast<int>(dB.size()); ++j) {
            if (j == best_index) continue;
            const int d = static_cast<int>(hamming_distance(dA[i], dB[j]));
            if (d < second_distance) second_distance = d;
        }

        // A zero-distance tie is ambiguous. With one candidate, the ratio
        // test has no second neighbour and is intentionally rejected.
        if (second_distance == 257) continue;
        if (!(static_cast<float>(best_distance) <
              params::kRatio * static_cast<float>(second_distance))) continue;
        if (best_a_for_b[best_index] != i) continue;

        matches.push_back({i, best_index, best_distance});
    }
}
