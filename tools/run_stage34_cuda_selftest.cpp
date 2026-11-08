#include "../src/core_headers/v0_stages.h"
#include "../src/core_headers/v1_stages.h"
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

void require(bool ok) { if (!ok) throw std::runtime_error("S3/S4 CPU/GPU mismatch"); }
void check_match(const std::vector<Descriptor>& a, const std::vector<Descriptor>& b) {
    std::vector<Match> cpu, gpu;
    v0_match(a, b, cpu);
    v1_match(a, b, gpu);
    require(cpu.size() == gpu.size());
    for (size_t i = 0; i < cpu.size(); ++i)
        require(cpu[i].idxA == gpu[i].idxA && cpu[i].idxB == gpu[i].idxB && cpu[i].dist == gpu[i].dist);
}
int main() {
    try {
        std::mt19937 rng(42);
        Gray image;
        image.w = 47; image.h = 39;
        image.data.resize(image.w * image.h);
        for (auto& p : image.data) p = static_cast<float>(rng() % 256);
        // Includes clamped borders and fractional keypoint coordinates.
        std::vector<Keypoint> kps{{0, 0, 0}, {46, 38, 0}, {20.75f, 18.5f, 0}};
        for (int i = 0; i < 257; ++i)
            kps.push_back({static_cast<float>(rng() % 47), static_cast<float>(rng() % 39), 0});
        std::vector<Descriptor> cpu, gpu;
        v0_describe(image, kps, cpu);
        v1_describe(image, kps, gpu);
        require(cpu.size() == gpu.size());
        for (size_t i = 0; i < cpu.size(); ++i)
            for (int j = 0; j < 4; ++j) require(cpu[i].bits[j] == gpu[i].bits[j]);
        check_match(cpu, gpu);
        check_match({}, gpu);
        check_match(cpu, {});
        check_match({Descriptor{}}, {Descriptor{}}); // no second neighbour
        check_match({Descriptor{}}, {Descriptor{}, Descriptor{}}); // zero tie
        std::vector<Descriptor> a(259), b(131);
        for (auto& d : a) for (auto& word : d.bits) word = (uint64_t(rng()) << 32) | rng();
        for (auto& d : b) for (auto& word : d.bits) word = (uint64_t(rng()) << 32) | rng();
        for (size_t i = 0; i < b.size(); i += 3) b[i] = a[i];
        b[1] = b[0];
        check_match(a, b);
        check_match(b, a);
        kps.push_back({std::numeric_limits<float>::quiet_NaN(), 0, 0});
        v1_describe(image, kps, gpu);
        require(gpu.empty());
        v1_describe(Gray{}, {}, gpu);
        require(gpu.empty());
        std::cout << "S3/S4 CUDA parity tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
