// Authors: Nguyen Duy Khanh; V2 coverage assisted by OpenAI Codex.
// External sources: none. Compare both CUDA matchers against V0.
#include "../../core_headers/v0_stages.h"
#include "../../core_headers/v1_stages.h"
#include "../../core_headers/v2_stages.h"
#include <iostream>
#include <random>
#include <stdexcept>

void require(bool ok) { if (!ok) throw std::runtime_error("S4 CPU/GPU mismatch"); }
void check_match(const std::vector<Descriptor>& a, const std::vector<Descriptor>& b) {
    std::vector<Match> cpu, gpu;
    v0_match(a, b, cpu);
    v1_match(a, b, gpu);
    require(cpu.size() == gpu.size());
    for (size_t i = 0; i < cpu.size(); ++i)
        require(cpu[i].idxA == gpu[i].idxA && cpu[i].idxB == gpu[i].idxB && cpu[i].dist == gpu[i].dist);
    v2_match(a, b, gpu);
    require(cpu.size() == gpu.size());
    for (size_t i = 0; i < cpu.size(); ++i)
        require(cpu[i].idxA == gpu[i].idxA && cpu[i].idxB == gpu[i].idxB && cpu[i].dist == gpu[i].dist);
}
int main() {
    try {
        std::mt19937 rng(42);
        check_match({}, {});
        check_match({}, {Descriptor{}});
        check_match({Descriptor{}}, {});
        check_match({Descriptor{}}, {Descriptor{}}); // no second neighbour
        check_match({Descriptor{}}, {Descriptor{},Descriptor{}}); // ambiguous tie
        std::vector<Descriptor> a(259), b(131);
        for (auto& d : a) for (auto& word : d.bits) word = (uint64_t(rng()) << 32) | rng();
        for (auto& d : b) for (auto& word : d.bits) word = (uint64_t(rng()) << 32) | rng();
        for (size_t i = 0; i < b.size(); i += 3) b[i] = a[i];
        b[1] = b[0];
        check_match(a, b);
        check_match(b, a);
        // Partial/full tiles and inactive queries must all reach barriers.
        for (int n : {1, 2, 127, 128, 129, 255, 256, 257, 3000}) {
            std::vector<Descriptor> x(n), y(n + 1);
            for (auto& d : x) for (auto& word : d.bits) word = (uint64_t(rng()) << 32) | rng();
            for (int i = 0; i < n; ++i) y[i] = x[i];
            y[n] = x[0]; // equal best distances across tiles
            check_match(x, y);
            check_match(y, x);
        }
        // Nonzero distance ties straddle the tile boundary; reject ambiguity.
        std::vector<Descriptor> ties(129);
        for (auto& d : ties) for (auto& word : d.bits) word = ~uint64_t{0};
        ties[0] = Descriptor{}; ties[0].bits[0] = 1;
        ties[128] = ties[0];
        check_match({Descriptor{}}, ties);
        // Strict ratio boundary: 4 < 0.8*5 is false; 3 < 0.8*5 is true.
        Descriptor four{}, five{}, three{};
        four.bits[0]=15; five.bits[0]=31; three.bits[0]=7;
        check_match({Descriptor{}}, {four,five});
        check_match({Descriptor{}}, {three,five});
        std::vector<Match> expected;
        v2_match({Descriptor{}}, {four,five}, expected); require(expected.empty());
        v2_match({Descriptor{}}, {three,five}, expected); require(expected.size()==1);
        // Both forward queries pass ratio; reverse nearest must retain idxA=0.
        check_match({Descriptor{},Descriptor{}}, {three,five});
        v2_match({Descriptor{},Descriptor{}}, {three,five}, expected);
        require(expected.size()==1 && expected[0].idxA==0);
        std::cout << "PASS: S4 V0/V1/V2 ordered matching, tiles, ratio and cross-check\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
