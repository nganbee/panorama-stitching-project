// Authors: Nguyen Duy Khanh; V6 coverage assisted by OpenAI Codex.
// External sources: none. Compare both CUDA matchers against V0.
#include "../../core_headers/v0_stages.h"
#include "match_host.h"
// Include the implementation in this test TU to inspect raw top-2 before filtering.
#include "../../stage_implementations/v6_stage4.cu"
#include <bitset>

#include <iostream>
#include <random>
#include <stdexcept>

void require(bool ok) { if (!ok) throw std::runtime_error("S4 CPU/GPU mismatch"); }
void check_nearest(const std::vector<Descriptor>& a,const std::vector<Descriptor>& b) {
    if (a.empty()) return;
    v1_cuda::Buffer<Descriptor> da(a.size()),db(b.size());
    v1_cuda::Buffer<int> best(a.size()),first(a.size()),second(a.size());
    v1_cuda::check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(Descriptor),cudaMemcpyHostToDevice));
    if (!b.empty()) v1_cuda::check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(Descriptor),cudaMemcpyHostToDevice));
    v1_cuda::launch([&] {
        nearest_warp_kernel<<<static_cast<unsigned>((a.size()+3)/4),128>>>(
            da.data,int(a.size()),db.data,int(b.size()),best.data,first.data,second.data);
    });
    std::vector<int> indices(a.size()),d1(a.size()),d2(a.size());
    v1_cuda::check(cudaMemcpy(indices.data(),best.data,a.size()*sizeof(int),cudaMemcpyDeviceToHost));
    v1_cuda::check(cudaMemcpy(d1.data(),first.data,a.size()*sizeof(int),cudaMemcpyDeviceToHost));
    v1_cuda::check(cudaMemcpy(d2.data(),second.data,a.size()*sizeof(int),cudaMemcpyDeviceToHost));
    for (size_t i=0; i<a.size(); ++i) {
        int index=-1,x=257,y=257;
        for (size_t j=0; j<b.size(); ++j) {
            int d=0;
            for (int k=0; k<4; ++k) d+=int(std::bitset<64>(a[i].bits[k]^b[j].bits[k]).count());
            if (d<x) { y=x; x=d; index=int(j); } else if (d<y) y=d;
        }
        require(indices[i]==index && d1[i]==x && d2[i]==y);
    }
}
void check_match(const std::vector<Descriptor>& a, const std::vector<Descriptor>& b) {
    check_nearest(a,b);
    std::vector<Match> cpu, gpu;
    v0_match(a, b, cpu);
    match5(a, b, gpu);
    require(cpu.size() == gpu.size());
    for (size_t i = 0; i < cpu.size(); ++i)
        require(cpu[i].idxA == gpu[i].idxA && cpu[i].idxB == gpu[i].idxB && cpu[i].dist == gpu[i].dist);
    match6(a, b, gpu);
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
        // Partial candidate sets and inactive final warps must preserve top-2.
        for (int n : {1, 2, 3, 4, 5, 31, 32, 33, 127, 128, 129, 255, 256, 257, 3000}) {
            std::vector<Descriptor> x(n), y(n + 1);
            for (auto& d : x) for (auto& word : d.bits) word = (uint64_t(rng()) << 32) | rng();
            for (int i = 0; i < n; ++i) y[i] = x[i];
            y[n] = x[0]; // equal best distances across tiles
            check_match(x, y);
            check_match(y, x);
        }
        // Nonzero distance ties cross lane/stride boundaries; reject ambiguity.
        std::vector<Descriptor> ties(129);
        for (auto& d : ties) for (auto& word : d.bits) word = ~uint64_t{0};
        ties[0] = Descriptor{}; ties[0].bits[0] = 1;
        ties[31] = ties[0]; ties[32] = ties[0]; ties[128] = ties[0];
        check_match({Descriptor{}}, ties);
        for (int count : {7, 31, 32, 33, 63, 64, 65, 95, 96, 97}) {
            std::vector<Descriptor> queries(5), candidates(count);
            for (auto& d : queries) for (auto& word : d.bits) word=(uint64_t(rng())<<32)|rng();
            for (auto& d : candidates) for (auto& word : d.bits) word=(uint64_t(rng())<<32)|rng();
            candidates.back()=queries[0];
            check_match(queries,candidates);
            check_match(candidates,queries);
        }
        // Strict ratio boundary: 4 < 0.8*5 is false; 3 < 0.8*5 is true.
        Descriptor four{}, five{}, three{};
        four.bits[0]=15; five.bits[0]=31; three.bits[0]=7;
        check_match({Descriptor{}}, {four,five});
        check_match({Descriptor{}}, {three,five});
        std::vector<Match> expected;
        match6({Descriptor{}}, {four,five}, expected); require(expected.empty());
        match6({Descriptor{}}, {three,five}, expected); require(expected.size()==1);
        // Both forward queries pass ratio; reverse nearest must retain idxA=0.
        check_match({Descriptor{},Descriptor{}}, {three,five});
        match6({Descriptor{},Descriptor{}}, {three,five}, expected);
        require(expected.size()==1 && expected[0].idxA==0);
        std::cout << "PASS: S4 V0/V5/V6 ordered matching, shared-tile warp top-2, ratio and cross-check\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
