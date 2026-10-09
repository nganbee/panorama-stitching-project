// File: v0_stage6.cpp
// Description: V0 (sequential) Stage 6 (Warping & Blending).
//              Warps image B onto image A's frame using the estimated Homography,
//              and blends them using a distance-based feathering approach.
// External sources: none
#include "../core_headers/common.h"
#include "../core_headers/v0_stages.h"
#include <cmath>
#include <algorithm>
#include <vector>

void v0_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano) {
    const double* h = H.h;
    
    // 1. Calculate panorama bounding box in A's coordinate system
    int min_x = 0, min_y = 0;
    int max_x = A.w, max_y = A.h;

    auto update_bounds = [&](double bx, double by) {
        double z = h[6]*bx + h[7]*by + h[8];
        double ax = (h[0]*bx + h[1]*by + h[2]) / z;
        double ay = (h[3]*bx + h[4]*by + h[5]) / z;
        if (ax < min_x) min_x = (int)std::floor(ax);
        if (ax > max_x) max_x = (int)std::ceil(ax);
        if (ay < min_y) min_y = (int)std::floor(ay);
        if (ay > max_y) max_y = (int)std::ceil(ay);
    };

    // Project B's 4 corners onto A's coordinate space
    update_bounds(0, 0);
    update_bounds(B.w - 1, 0);
    update_bounds(0, B.h - 1);
    update_bounds(B.w - 1, B.h - 1);

    int pano_w = max_x - min_x;
    int pano_h = max_y - min_y;
    pano.w = pano_w;
    pano.h = pano_h;
    pano.c = 3;
    pano.data.assign((size_t)pano_w * pano_h * 3, 0);

    // 2. Compute Inverse Homography (Adjugate matrix of H)
    // For projective transformation, multiplying by a scalar doesn't change coordinates.
    double H_inv[9];
    H_inv[0] = h[4]*h[8] - h[5]*h[7];
    H_inv[1] = h[2]*h[7] - h[1]*h[8];
    H_inv[2] = h[1]*h[5] - h[2]*h[4];
    
    H_inv[3] = h[5]*h[6] - h[3]*h[8];
    H_inv[4] = h[0]*h[8] - h[2]*h[6];
    H_inv[5] = h[2]*h[3] - h[0]*h[5];
    
    H_inv[6] = h[3]*h[7] - h[4]*h[6];
    H_inv[7] = h[1]*h[6] - h[0]*h[7];
    H_inv[8] = h[0]*h[4] - h[1]*h[3];

    // 3. Inverse Warping and Blending
    for (int py = 0; py < pano_h; ++py) {
        for (int px = 0; px < pano_w; ++px) {
            // Coordinate in A's space
            int ax = px + min_x;
            int ay = py + min_y;
            
            bool in_A = (ax >= 0 && ax < A.w && ay >= 0 && ay < A.h);
            
            // Map coordinate back to B's space
            double z = H_inv[6]*ax + H_inv[7]*ay + H_inv[8];
            double bx = (H_inv[0]*ax + H_inv[1]*ay + H_inv[2]) / z;
            double by = (H_inv[3]*ax + H_inv[4]*ay + H_inv[5]) / z;
            
            bool in_B = (bx >= 0 && bx <= B.w - 1 && by >= 0 && by <= B.h - 1);
            
            uint8_t cA[3] = {0, 0, 0};
            uint8_t cB[3] = {0, 0, 0};
            
            // Get color from A (nearest neighbor since exact integer coordinate)
            if (in_A) {
                size_t idx = ((size_t)ay * A.w + ax) * 3;
                cA[0] = A.data[idx];
                cA[1] = A.data[idx+1];
                cA[2] = A.data[idx+2];
            }
            
            // Get color from B (Bilinear Interpolation for sub-pixel accuracy)
            if (in_B) {
                int bx0 = (int)bx;
                int by0 = (int)by;
                int bx1 = std::min(bx0 + 1, B.w - 1);
                int by1 = std::min(by0 + 1, B.h - 1);
                double dx = bx - bx0;
                double dy = by - by0;
                
                for(int c = 0; c < 3; ++c) {
                    double v00 = B.data[((size_t)by0 * B.w + bx0) * 3 + c];
                    double v10 = B.data[((size_t)by0 * B.w + bx1) * 3 + c];
                    double v01 = B.data[((size_t)by1 * B.w + bx0) * 3 + c];
                    double v11 = B.data[((size_t)by1 * B.w + bx1) * 3 + c];
                    
                    double v0 = v00 * (1 - dx) + v10 * dx;
                    double v1 = v01 * (1 - dx) + v11 * dx;
                    cB[c] = (uint8_t)(v0 * (1 - dy) + v1 * dy);
                }
            }
            
            // Write to panorama and blend if overlapping
            size_t out_idx = ((size_t)py * pano_w + px) * 3;
            
            if (in_A && in_B) {
                // Feather blending: Weight based on distance to nearest border
                double distA = std::min({(double)ax, (double)(A.w - 1 - ax), (double)ay, (double)(A.h - 1 - ay)});
                double distB = std::min({bx, (double)(B.w - 1) - bx, by, (double)(B.h - 1) - by});
                
                double wA = distA / (distA + distB + 1e-5);
                double wB = distB / (distA + distB + 1e-5);
                
                pano.data[out_idx]   = (uint8_t)(cA[0] * wA + cB[0] * wB);
                pano.data[out_idx+1] = (uint8_t)(cA[1] * wA + cB[1] * wB);
                pano.data[out_idx+2] = (uint8_t)(cA[2] * wA + cB[2] * wB);
            } else if (in_A) {
                pano.data[out_idx]   = cA[0];
                pano.data[out_idx+1] = cA[1];
                pano.data[out_idx+2] = cA[2];
            } else if (in_B) {
                pano.data[out_idx]   = cB[0];
                pano.data[out_idx+1] = cB[1];
                pano.data[out_idx+2] = cB[2];
            }
        }
    }
}
