// File: synthetic_homography_test.cpp
// Authors: Team 8 (V0 validation)
// Description: Deterministic V0 S5 correctness check against a known
//              translation homography with additional outlier matches.
// External sources: none.
#include "../src/core_headers/common.h"
#include "../src/core_headers/v0_stages.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <limits>

namespace {
bool projective_case() {
    const Homography truth={{1.015,0.021,37,-0.014,0.99,-19,0.00008,-0.00003,1}};
    std::vector<Keypoint> a,b; std::vector<Match> matches;
    for(int y=0;y<6;++y) for(int x=0;x<8;++x) {
        const double bx=80+70*x,by=90+65*y;
        const double z=truth.h[6]*bx+truth.h[7]*by+1;
        b.push_back({float(bx),float(by),1});
        a.push_back({float((truth.h[0]*bx+truth.h[1]*by+truth.h[2])/z),
                     float((truth.h[3]*bx+truth.h[4]*by+truth.h[5])/z),1});
        const int i=int(a.size())-1; matches.push_back({i,i,0});
    }
    for(int i=0;i<15;++i) matches.push_back({i,47-i,200});
    Homography estimated{}; int inliers=0;
    bool ok=v0_ransac_homography(a,b,matches,estimated,inliers);
    double max_error=0,mean_error=0;
    for(size_t i=0;i<a.size() && ok;++i) {
        double z=estimated.h[6]*b[i].x+estimated.h[7]*b[i].y+estimated.h[8];
        double dx=(estimated.h[0]*b[i].x+estimated.h[1]*b[i].y+estimated.h[2])/z-a[i].x;
        double dy=(estimated.h[3]*b[i].x+estimated.h[4]*b[i].y+estimated.h[5])/z-a[i].y;
        const double e=std::hypot(dx,dy); max_error=std::max(max_error,e);mean_error+=e/a.size();
    }
    ok=ok && inliers>=48 && max_error<1e-3;
    std::printf("[S5] projective_with_outliers %s inliers=%d mean_error_px=%.3e max_error_px=%.3e\n",
                ok?"PASS":"FAIL",inliers,mean_error,max_error);
    return ok;
}
bool reject_bad_cases() {
    std::vector<Keypoint> a,b; std::vector<Match> m;
    for(int i=0;i<8;++i) {a.push_back({float(i+3),float(2*i+5),1});b.push_back({float(i),float(2*i),1});m.push_back({i,i,0});}
    Homography H{};int n=0;
    bool ok=!v0_ransac_homography(a,b,m,H,n);
    m.assign(4,Match{-1,999,0});ok &= !v0_ransac_homography(a,b,m,H,n);
    m.clear();ok &= !v0_ransac_homography(a,b,m,H,n);
    for(int i=0;i<8;++i) m.push_back({i,i,0});
    for(auto& p:a) p.x=std::numeric_limits<float>::quiet_NaN();
    ok &= !v0_ransac_homography(a,b,m,H,n);
    std::printf("[S5] degenerate_invalid_nonfinite %s\n",ok?"PASS":"FAIL");
    return ok;
}
}

int main() {
    std::vector<Keypoint> kpA;
    std::vector<Keypoint> kpB;
    std::vector<Match> matches;

    constexpr float tx = 37.0f;
    constexpr float ty = -19.0f;
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 8; ++x) {
            const float sx = 80.0f + 70.0f * x;
            const float sy = 90.0f + 65.0f * y;
            kpB.push_back({sx, sy, 1.0f});
            kpA.push_back({sx + tx, sy + ty, 1.0f});
            matches.push_back({static_cast<int>(kpA.size() - 1),
                               static_cast<int>(kpB.size() - 1), 0});
        }
    }

    // Add deterministic false matches. RANSAC should reject these.
    for (int i = 0; i < 12; ++i) {
        const int a = i;
        const int b = static_cast<int>(kpB.size()) - 1 - i;
        matches.push_back({a, b, 200});
    }

    Homography H{};
    int inliers = 0;
    const bool ok = v0_ransac_homography(kpA, kpB, matches, H, inliers);
    const double tx_error = std::fabs(H.h[2] - tx);
    const double ty_error = std::fabs(H.h[5] - ty);
    const bool pass = ok && inliers >= 40 && tx_error < 1e-3 && ty_error < 1e-3;

    std::printf("[SYNTHETIC_HOMOGRAPHY] %s inliers=%d tx=%.6f ty=%.6f "
                "tx_error=%.3e ty_error=%.3e\n",
                pass ? "PASS" : "FAIL", inliers, H.h[2], H.h[5], tx_error, ty_error);
    const bool projective=projective_case();
    const bool bad_cases=reject_bad_cases();
    return pass && projective && bad_cases ? 0 : 1;
}
