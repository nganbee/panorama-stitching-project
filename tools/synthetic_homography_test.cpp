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
    return pass ? 0 : 1;
}
