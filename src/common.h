// Data types, common pipeline parameters for all versions

#pragma once
#include <cstdint>
#include <vector>
#include <string>

struct Image {
    int w = 0;
    int h = 0;
    int c = 3;
    std::vector<uint8_t> data; // RGB image data
};

struct Gray {
    int w = 0;
    int h = 0;
    std::vector<float> data; // Grayscale image data
};

struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float score = 0.0f; // Harris score
};

struct Descriptor {
   uint64_t bits[4]; // BRIEF descriptor (256 bits)
};

struct Match {
    int idxA = -1; // Index of the keypoint in the first image
    int idxB = -1; // Index of the keypoint in the second image
    int dist; // Hamming distance between the two descriptors
};

struct Homography {
    double h[9]; // 3x3 homography matrix
};

// Pipeline parameters, used identically by every version
namespace params {
    // Stage 1
    constexpr float kGaussSigma   = 1.4f;
    constexpr int   kGaussRadius  = 4;          // ceil(3*sigma)
    // Stage 2
    constexpr float kHarrisK      = 0.04f;
    constexpr float kHarrisThresh = 0.01f;      // relative to max response
    constexpr int kHarrisWinRadius = 2;         // 5x5 window for Harris corner detection
    constexpr int   kNmsRadius    = 2;          // 5x5 window
    constexpr int   kTopK         = 3000;
    constexpr int   kBorder       = 16;         // >= half BRIEF patch
    // Stage 3
    constexpr int   kPatchSize    = 31;
    constexpr uint32_t kBriefSeed = 12345u;
    // Stage 4
    constexpr float kRatio        = 0.8f;
    // Stage 5
    constexpr int   kRansacIters  = 2000;
    constexpr float kRansacThresh = 3.0f;       // px
    constexpr uint32_t kRansacSeed = 2026u;
}

// Timing information for each stage, used identically by every version
struct StageTimes {
    double gray = 0, blur = 0, harris = 0, nms_topk = 0;
    double describe = 0, match = 0, ransac = 0, warp_blend = 0;
    double h2d = 0, d2h = 0;         // host-device transfer (0 in V0)
    double kernel_total = 0;         // sum of GPU kernel time (V1+)
    double end_to_end = 0;           // excludes image load/save
};

struct PipelineResult {
    Image panorama;
    Homography H{};
    int nKpA = 0, nKpB = 0, nMatches = 0, nInliers = 0;
    StageTimes times;
};
