// File: stage34_selftest.cpp
// Authors: Team 8; implementation assisted by OpenAI Codex.
// Description: Descriptor invariants and matching acceptance/rejection cases.
// External sources: none.
#include "v0_stages.h"
#include <cstdio>
#include <cstring>
#include <limits>

namespace {
bool check(const char* name, bool ok) {
    std::printf("[STAGE34] %s %s\n", name, ok ? "PASS" : "FAIL"); return ok;
}
Descriptor bits(int count) {
    Descriptor d{};
    for (int i=0;i<count;++i) d.bits[i/64] |= uint64_t{1}<<(i%64);
    return d;
}
}
int main() {
    bool ok=true;
    Gray g; g.w=64; g.h=64; g.data.assign(4096,100.f);
    std::vector<Keypoint> kps{{16,16,1},{32,32,1}};
    std::vector<Descriptor> d;
    v0_describe(g,kps,d);
    bool zero=d.size()==2;
    for(const auto& x:d) for(auto b:x.bits) zero &= b==0;
    ok &= check("constant_patch_zero_descriptor",zero);
    for(int y=0;y<64;++y) for(int x=0;x<64;++x)
        g.data[y*64+x]=float(((x%32)*17+(y%32)*31)%251);
    kps={{16,16,1},{48,16,1}};
    v0_describe(g,kps,d);
    ok &= check("translated_identical_patches",d.size()==2 && std::memcmp(&d[0],&d[1],sizeof(Descriptor))==0);
    kps[0].x=std::numeric_limits<float>::quiet_NaN(); v0_describe(g,kps,d);
    ok &= check("reject_nonfinite_keypoint",d.empty());
    std::vector<Match> m;
    const std::vector<Descriptor> a{bits(0),bits(256),bits(128)};
    const std::vector<Descriptor> b{bits(128),bits(0),bits(256)};
    v0_match(a,b,m);
    ok &= check("exact_permutation_hamming",m.size()==3 && m[0].idxB==1 && m[1].idxB==2 && m[2].idxB==0 && m[0].dist==0);
    v0_match({bits(0)},{bits(0),bits(0)},m);
    ok &= check("reject_ambiguous_zero_tie",m.empty());
    v0_match({bits(0)},{bits(0)},m);
    ok &= check("reject_missing_second_neighbor",m.empty());
    v0_match({bits(0),bits(0)},{bits(0),bits(256)},m);
    ok &= check("mutual_cross_check",m.size()==1 && m[0].idxA==0);
    v0_match({bits(0)},{bits(9),bits(10)},m);
    ok &= check("ratio_rejects_weak_match",m.empty());
    v0_match({},b,m); ok &= check("empty_inputs",m.empty());
    return ok?0:1;
}
