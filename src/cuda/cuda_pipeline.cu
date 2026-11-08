// Authors: Team 8, with Codex implementation assistance.
// Real CUDA implementations of all six panorama stages.
// External sources: none. Algorithms follow the project V0 contract.
#include "cuda_pipeline.h"
#include "homography_math.cuh"
#include "../core_headers/cuda_utils.cuh"
#include "../core_headers/v0_stages.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using cuda_homography::Point;
constexpr int threads = 256;
constexpr int tile = 16;
constexpr int match_tile = 128;
constexpr double threshold2 = static_cast<double>(params::kRansacThresh) * params::kRansacThresh;
struct BriefPair { int ax, ay, bx, by; };
struct Nearest { int index, best, second; };
struct Score { int count; double error; };
__constant__ float gaussian[2 * params::kGaussRadius + 1];
__constant__ BriefPair brief_pairs[256];

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
int blocks(std::size_t n) { return static_cast<int>((n + threads - 1) / threads); }
__device__ int clamp_index(int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); }
__device__ float at(const float* in, int w, int h, int x, int y) {
    return in[static_cast<std::size_t>(clamp_index(y, h)) * w + clamp_index(x, w)];
}

__global__ void gray_kernel(const uint8_t* rgb, float* gray, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gray[i] = 0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2];
}

template<bool horizontal>
__global__ void blur_global(const float* in, float* out, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float value = 0;
    for (int d = -params::kGaussRadius; d <= params::kGaussRadius; ++d)
        value += gaussian[d + params::kGaussRadius] * at(in, w, h, x + (horizontal ? d : 0), y + (horizontal ? 0 : d));
    out[static_cast<std::size_t>(y) * w + x] = value;
}

// V2/V3: load each halo pixel once per tile, then reuse it for neighbouring outputs.
template<bool horizontal>
__global__ void blur_shared(const float* in, float* out, int w, int h) {
    constexpr int r = params::kGaussRadius;
    constexpr int sw = tile + (horizontal ? 2 * r : 0);
    constexpr int sh = tile + (horizontal ? 0 : 2 * r);
    __shared__ float cache[sh][sw];
    const int tid = threadIdx.y * tile + threadIdx.x;
    const int origin_x = blockIdx.x * tile, origin_y = blockIdx.y * tile;
    for (int i = tid; i < sw * sh; i += tile * tile)
        cache[i / sw][i % sw] = at(in, w, h, origin_x + i % sw - (horizontal ? r : 0),
                                 origin_y + i / sw - (horizontal ? 0 : r));
    __syncthreads();
    const int x = origin_x + threadIdx.x, y = origin_y + threadIdx.y;
    if (x >= w || y >= h) return;
    float value = 0;
    for (int d = -r; d <= r; ++d)
        value += gaussian[d + r] * cache[threadIdx.y + (horizontal ? 0 : d + r)]
                                                   [threadIdx.x + (horizontal ? d + r : 0)];
    out[static_cast<std::size_t>(y) * w + x] = value;
}

__global__ void tensor_kernel(const float* in, float* xx, float* yy, float* xy, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float ix = ((at(in,w,h,x+1,y-1) + 2 * at(in,w,h,x+1,y) + at(in,w,h,x+1,y+1)) -
                      (at(in,w,h,x-1,y-1) + 2 * at(in,w,h,x-1,y) + at(in,w,h,x-1,y+1))) * 0.125f;
    const float iy = ((at(in,w,h,x-1,y+1) + 2 * at(in,w,h,x,y+1) + at(in,w,h,x+1,y+1)) -
                      (at(in,w,h,x-1,y-1) + 2 * at(in,w,h,x,y-1) + at(in,w,h,x+1,y-1))) * 0.125f;
    const std::size_t i = static_cast<std::size_t>(y) * w + x;
    xx[i] = ix * ix; yy[i] = iy * iy; xy[i] = ix * iy;
}

template<bool shared>
__global__ void harris_kernel(const float* xx, const float* yy, const float* xy, float* out, int w, int h) {
    constexpr int r = params::kHarrisWinRadius, side = tile + 2 * r;
    // NVCC removes this static storage for the false specialization.
    __shared__ float cxx[side][side], cyy[side][side], cxy[side][side];
    const int x = blockIdx.x * tile + threadIdx.x, y = blockIdx.y * tile + threadIdx.y;
    if (shared) {
        const int tid = threadIdx.y * tile + threadIdx.x;
        for (int i = tid; i < side * side; i += tile * tile) {
            const int sx = blockIdx.x * tile + i % side - r;
            const int sy = blockIdx.y * tile + i / side - r;
            cxx[i / side][i % side] = at(xx,w,h,sx,sy);
            cyy[i / side][i % side] = at(yy,w,h,sx,sy);
            cxy[i / side][i % side] = at(xy,w,h,sx,sy);
        }
        __syncthreads();
    }
    if (x >= w || y >= h) return;
    float sxx = 0, syy = 0, sxy = 0;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            if (shared) {
                const int row = threadIdx.y + dy + r, col = threadIdx.x + dx + r;
                sxx += cxx[row][col]; syy += cyy[row][col]; sxy += cxy[row][col];
            } else {
                sxx += at(xx,w,h,x+dx,y+dy); syy += at(yy,w,h,x+dx,y+dy); sxy += at(xy,w,h,x+dx,y+dy);
            }
        }
    const float det = sxx * syy - sxy * sxy, trace = sxx + syy;
    out[static_cast<std::size_t>(y) * w + x] = det - params::kHarrisK * trace * trace;
}

__global__ void maximum_kernel(const float* in, int n, float* out) {
    __shared__ float values[threads];
    float value = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        value = fmaxf(value, in[i]);
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = values[0];
}

__global__ void nms_kernel(const float* response, const float* maximum, Keypoint* candidates,
                           int* count, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = params::kBorder, r = params::kNmsRadius;
    if (x < b || y < b || x >= w - b || y >= h - b) return;
    const int i = y * w + x;
    const float value = response[i];
    if (value <= params::kHarrisThresh * maximum[0]) return;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            const int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            const int j = ny * w + nx;
            if (response[j] > value || (response[j] == value && j < i)) return;
        }
    candidates[atomicAdd(count, 1)] = {static_cast<float>(x), static_cast<float>(y), value};
}

__device__ bool brief_bit(const float* in, int w, int h, const Keypoint& kp, int bit) {
    const BriefPair p = brief_pairs[bit];
    const int x = static_cast<int>(kp.x), y = static_cast<int>(kp.y);
    return at(in,w,h,x+p.ax,y+p.ay) < at(in,w,h,x+p.bx,y+p.by);
}
__global__ void describe_scalar(const float* in, int w, int h, const Keypoint* kp, int n, Descriptor* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Descriptor d{};
    for (int bit = 0; bit < 256; ++bit)
        if (brief_bit(in,w,h,kp[i],bit)) d.bits[bit / 64] |= uint64_t{1} << (bit % 64);
    out[i] = d;
}
__global__ void describe_warp(const float* in, int w, int h, const Keypoint* kp, int n, Descriptor* out) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (i >= n) return; // Whole warp exits together.
    for (int word = 0; word < 4; ++word) {
        const unsigned low = __ballot_sync(0xffffffffu, brief_bit(in,w,h,kp[i],word * 64 + lane));
        const unsigned high = __ballot_sync(0xffffffffu, brief_bit(in,w,h,kp[i],word * 64 + 32 + lane));
        if (lane == 0) out[i].bits[word] = static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
    }
}

__device__ int distance(const Descriptor& a, const Descriptor& b) {
    int result = 0;
    for (int word = 0; word < 4; ++word)
        result += __popcll(static_cast<unsigned long long>(a.bits[word] ^ b.bits[word]));
    return result;
}
__device__ void update_nearest(Nearest& result, int candidate, int d) {
    // Candidates are visited in ascending index: ties keep the first index,
    // and another identical best distance becomes the second neighbour.
    if (d < result.best) { result.second = result.best; result.best = d; result.index = candidate; }
    else if (d < result.second) result.second = d;
}

template<bool shared>
__global__ void nearest_kernel(const Descriptor* queries, int nq, const Descriptor* candidates,
                               int nc, Nearest* out) {
    __shared__ Descriptor cache[match_tile];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < nq;
    Descriptor query{};
    if (active) query = queries[i];
    Nearest result{-1, 257, 257};
    if (shared) {
        for (int base = 0; base < nc; base += match_tile) {
            const int j = base + threadIdx.x;
            if (j < nc) cache[threadIdx.x] = candidates[j];
            __syncthreads();
            const int limit = min(match_tile, nc - base);
            if (active)
                for (int k = 0; k < limit; ++k) update_nearest(result, base + k, distance(query, cache[k]));
            __syncthreads();
        }
    } else if (active) {
        for (int j = 0; j < nc; ++j) update_nearest(result, j, distance(query,candidates[j]));
    }
    if (active) out[i] = result;
}

__global__ void crosscheck_kernel(const Nearest* forward, const Nearest* reverse, int n, Match* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Nearest m = forward[i];
    out[i] = {-1, -1, 257};
    if (m.index >= 0 && m.second != 257 && static_cast<float>(m.best) < params::kRatio * m.second &&
        reverse[m.index].index == i) out[i] = {i, m.index, m.best};
}

__global__ void prepare_correspondences_kernel(
    const Keypoint* kpA, int nKpA, const Keypoint* kpB, int nKpB,
    const Match* matches, int nMatches, Point* src, Point* dst, int* nValid) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int count = 0;
    for (int i = 0; i < nMatches; ++i) {
        const Match match = matches[i];
        if (match.idxA < 0 || match.idxA >= nKpA ||
            match.idxB < 0 || match.idxB >= nKpB) continue;
        const Keypoint a = kpA[match.idxA];
        const Keypoint b = kpB[match.idxB];
        if (!cuda_homography::finite(a.x) || !cuda_homography::finite(a.y) ||
            !cuda_homography::finite(b.x) || !cuda_homography::finite(b.y)) continue;
        // Keep the same stable order as V0; H maps points from B into A.
        src[count] = {static_cast<double>(b.x), static_cast<double>(b.y)};
        dst[count] = {static_cast<double>(a.x), static_cast<double>(a.y)};
        ++count;
    }
    *nValid = count;
}

__device__ uint32_t next_ransac_random(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__global__ void hypotheses_kernel(const Point* src, const Point* dst, const int* nValid,
                                  Homography* models, int* valid, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    valid[i] = 0;
    const int count = *nValid;
    if (count < 4) return;

    // Fixed per-hypothesis streams are deterministic and unaffected by thread
    // scheduling, so later S5 variants can reuse the same samples.
    uint32_t state = static_cast<uint32_t>(params::kRansacSeed) ^
        (static_cast<uint32_t>(i + 1) * 0x9e3779b9u);
    if (state == 0u) state = 0x6d2b79f5u;
    int sample[4];
    for (int slot = 0; slot < 4; ++slot) {
        bool duplicate;
        do {
            sample[slot] = static_cast<int>(next_ransac_random(state) %
                                            static_cast<uint32_t>(count));
            duplicate = false;
            for (int j = 0; j < slot; ++j)
                duplicate = duplicate || sample[slot] == sample[j];
        } while (duplicate);
    }
    valid[i] = cuda_homography::estimate(src, dst, sample, 4, models[i]) ? 1 : 0;
}
__global__ void score_serial(const Point* src, const Point* dst, const int* nValid, const Homography* models,
                             const int* valid, Score* scores, int iterations) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= iterations) return;
    Score score{-1, DBL_MAX};
    if (valid[i]) {
        score = {0,0};
        for (int j = 0; j < *nValid; ++j) {
            const double error = cuda_homography::squared_error(models[i],src[j],dst[j]);
            if (error <= threshold2) { ++score.count; score.error += error; }
        }
    }
    scores[i] = score;
}
// V3: one block per hypothesis, coalesced match reads and a tree reduction.
__global__ void score_parallel(const Point* src, const Point* dst, const int* nValid, const Homography* models,
                               const int* valid, Score* scores) {
    __shared__ int counts[threads];
    __shared__ double errors[threads];
    const int i = blockIdx.x;
    const int n = *nValid;
    int count = 0; double sum = 0;
    if (valid[i])
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            const double error = cuda_homography::squared_error(models[i],src[j],dst[j]);
            if (error <= threshold2) { ++count; sum += error; }
        }
    counts[threadIdx.x] = count; errors[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) { counts[threadIdx.x] += counts[threadIdx.x+stride]; errors[threadIdx.x] += errors[threadIdx.x+stride]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[i] = valid[i] ? Score{counts[0],errors[0]} : Score{-1,DBL_MAX};
}

__global__ void select_refine_count_kernel(
    const Point* src, const Point* dst, const int* nValid,
    const Homography* models, const int* model_valid, const Score* scores,
    int* selected_inliers, Homography* output_h, int* output_inliers) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int n = *nValid;
    for (int i = 0; i < 9; ++i) output_h->h[i] = 0.0;
    *output_inliers = 0;
    if (n < 4) return;

    int best = -1;
    for (int i = 0; i < params::kRansacIters; ++i) {
        if (!model_valid[i] || scores[i].count < 4) continue;
        if (best < 0 || scores[i].count > scores[best].count ||
            (scores[i].count == scores[best].count && scores[i].error < scores[best].error))
            best = i;
    }
    if (best < 0) return;

    int selected_count = 0;
    for (int i = 0; i < n; ++i)
        if (cuda_homography::squared_error(models[best], src[i], dst[i]) <= threshold2)
            selected_inliers[selected_count++] = i;
    if (selected_count < 4) return;

    Homography refined{};
    if (!cuda_homography::estimate(src, dst, selected_inliers, selected_count, refined)) return;
    int final_count = 0;
    for (int i = 0; i < n; ++i)
        if (cuda_homography::squared_error(refined, src[i], dst[i]) <= threshold2)
            ++final_count;
    if (final_count < 4) return;

    *output_h = refined;
    *output_inliers = final_count;
}

__device__ uint8_t rounded_byte(double value) {
    value = fmin(255.0, fmax(0.0, value));
    return static_cast<uint8_t>(floor(value + 0.5));
}
__global__ void warp_kernel(const uint8_t* a, int aw, int ah, const uint8_t* b, int bw, int bh,
                            Homography inv, int minx, int miny, int w, int h, uint8_t* out, uint8_t* valid) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    const int ax = px + minx, ay = py + miny;
    const bool ina = ax >= 0 && ay >= 0 && ax < aw && ay < ah;
    const double z = inv.h[6] * ax + inv.h[7] * ay + inv.h[8];
    double bx = 0, by = 0;
    const double z_scale = fabs(inv.h[6] * ax) + fabs(inv.h[7] * ay) + fabs(inv.h[8]);
    bool inb = cuda_homography::finite(z) && z != 0 && fabs(z) > 64 * DBL_EPSILON * z_scale;
    if (inb) {
        bx = (inv.h[0] * ax + inv.h[1] * ay + inv.h[2]) / z;
        by = (inv.h[3] * ax + inv.h[4] * ay + inv.h[5]) / z;
        const double tolerance = 1e-7 + 64 * DBL_EPSILON * max(bw,bh);
        inb = cuda_homography::finite(bx) && cuda_homography::finite(by) &&
              bx >= -tolerance && by >= -tolerance && bx <= bw - 1 + tolerance && by <= bh - 1 + tolerance;
        if (inb) { bx = fmin(bw - 1.0,fmax(0.0,bx)); by = fmin(bh - 1.0,fmax(0.0,by)); }
    }
    const std::size_t pixel = static_cast<std::size_t>(py) * w + px;
    valid[pixel] = ina || inb ? 1 : 0;
    double wa = ina ? 1 : 0, wb = inb ? 1 : 0;
    if (ina && inb) {
        const double da = fmin(fmin(static_cast<double>(ax),aw-1.0-ax), fmin(static_cast<double>(ay),ah-1.0-ay));
        const double db = fmin(fmin(bx,bw-1.0-bx), fmin(by,bh-1.0-by));
        const double sum = da + db;
        wa = sum > 0 ? da / sum : 0.5; wb = sum > 0 ? db / sum : 0.5;
    }
    for (int c = 0; c < 3; ++c) {
        double ca = 0, cb = 0;
        if (ina) ca = a[(static_cast<std::size_t>(ay) * aw + ax) * 3 + c];
        if (inb) {
            const int x0 = static_cast<int>(bx), y0 = static_cast<int>(by);
            const int x1 = min(x0+1,bw-1), y1 = min(y0+1,bh-1);
            const double dx = bx-x0, dy = by-y0;
            const double v00 = b[(static_cast<std::size_t>(y0)*bw+x0)*3+c];
            const double v10 = b[(static_cast<std::size_t>(y0)*bw+x1)*3+c];
            const double v01 = b[(static_cast<std::size_t>(y1)*bw+x0)*3+c];
            const double v11 = b[(static_cast<std::size_t>(y1)*bw+x1)*3+c];
            cb = (v00*(1-dx)+v10*dx)*(1-dy) + (v01*(1-dx)+v11*dx)*dy;
        }
        out[pixel * 3 + c] = rounded_byte(ca*wa + cb*wb);
    }
}

struct DeviceImage {
    DeviceBuffer<uint8_t> rgb;
    DeviceBuffer<float> gray, temp, blurred, xx, yy, xy, response, maxima, maximum;
    DeviceBuffer<Keypoint> candidates, keypoints;
    DeviceBuffer<int> count;
    DeviceBuffer<Descriptor> descriptors;
    std::vector<Keypoint> host_keypoints;
    int w = 0, h = 0;
    void reserve(const Image& img) {
        w = img.w; h = img.h;
        const std::size_t n = static_cast<std::size_t>(w) * h;
        rgb.reserve(n*3); gray.reserve(n); temp.reserve(n); blurred.reserve(n);
        xx.reserve(n); yy.reserve(n); xy.reserve(n); response.reserve(n);
        maxima.reserve(blocks(n)); maximum.reserve(1); candidates.reserve(n); count.reserve(1);
        keypoints.reserve(params::kTopK); descriptors.reserve(params::kTopK);
    }
};
struct Workspace {
    GpuTimer timer;
    DeviceImage a, b;
    DeviceBuffer<Nearest> forward, reverse;
    DeviceBuffer<Match> raw_matches;
    DeviceBuffer<Keypoint> s5_kp_a, s5_kp_b;
    DeviceBuffer<Match> s5_matches;
    DeviceBuffer<Point> src, dst;
    DeviceBuffer<int> n_s5_valid, model_valid, selected_inliers, n_s5_inliers;
    DeviceBuffer<Homography> models, s5_output_h;
    DeviceBuffer<Score> scores;
    DeviceBuffer<uint8_t> panorama, coverage;
};

template<class Function>
void kernels(Workspace& ws, StageTimes& t, double& stage, Function launch) {
    ws.timer.start();
    launch();
    CUDA_CHECK(cudaGetLastError());
    const double ms = ws.timer.stop();
    stage += ms; t.kernel_total += ms;
}
template<class T>
void upload(Workspace& ws, StageTimes& t, T* device, const T* host, std::size_t count) {
    if (count == 0) return;
    ws.timer.start(); CUDA_CHECK(cudaMemcpy(device,host,count*sizeof(T),cudaMemcpyHostToDevice));
    t.h2d += ws.timer.stop();
}
template<class T>
void download(Workspace& ws, StageTimes& t, T* host, const T* device, std::size_t count) {
    if (count == 0) return;
    ws.timer.start(); CUDA_CHECK(cudaMemcpy(host,device,count*sizeof(T),cudaMemcpyDeviceToHost));
    t.d2h += ws.timer.stop();
}

void upload_constants(Workspace& ws, StageTimes& t) {
    // Exact xorshift32 draw order from v0_stage34.cpp, not a platform RNG.
    std::array<BriefPair,256> pattern{};
    uint32_t state = params::kBriefSeed;
    auto draw = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    const int radius = params::kPatchSize / 2;
    for (auto& p : pattern) {
        p.ax = static_cast<int>(draw() % (2*radius+1)) - radius;
        p.ay = static_cast<int>(draw() % (2*radius+1)) - radius;
        p.bx = static_cast<int>(draw() % (2*radius+1)) - radius;
        p.by = static_cast<int>(draw() % (2*radius+1)) - radius;
    }
    float gauss[2*params::kGaussRadius+1], sum = 0;
    for (int i = -params::kGaussRadius; i <= params::kGaussRadius; ++i) {
        gauss[i+params::kGaussRadius] = std::exp(-static_cast<float>(i*i)/(2*params::kGaussSigma*params::kGaussSigma));
        sum += gauss[i+params::kGaussRadius];
    }
    for (float& value : gauss) value /= sum;
    ws.timer.start();
    CUDA_CHECK(cudaMemcpyToSymbol(brief_pairs,pattern.data(),sizeof(pattern)));
    CUDA_CHECK(cudaMemcpyToSymbol(gaussian,gauss,sizeof(gauss)));
    t.h2d += ws.timer.stop();
}

void launch_blur(int version, DeviceImage& f) {
    const dim3 block(tile,tile), grid((f.w+tile-1)/tile,(f.h+tile-1)/tile);
    if (version == 1) {
        blur_global<true><<<grid,block>>>(f.gray.get(),f.temp.get(),f.w,f.h);
        blur_global<false><<<grid,block>>>(f.temp.get(),f.blurred.get(),f.w,f.h);
    } else {
        blur_shared<true><<<grid,block>>>(f.gray.get(),f.temp.get(),f.w,f.h);
        blur_shared<false><<<grid,block>>>(f.temp.get(),f.blurred.get(),f.w,f.h);
    }
}
void launch_harris(int version, DeviceImage& f) {
    const dim3 block(tile,tile), grid((f.w+tile-1)/tile,(f.h+tile-1)/tile);
    tensor_kernel<<<grid,block>>>(f.blurred.get(),f.xx.get(),f.yy.get(),f.xy.get(),f.w,f.h);
    if (version == 1) harris_kernel<false><<<grid,block>>>(f.xx.get(),f.yy.get(),f.xy.get(),f.response.get(),f.w,f.h);
    else harris_kernel<true><<<grid,block>>>(f.xx.get(),f.yy.get(),f.xy.get(),f.response.get(),f.w,f.h);
}
void select_keypoints(int version, Workspace& ws, DeviceImage& f, StageTimes& t) {
    (void)version;
    const int n = f.w*f.h, nb = blocks(n);
    const dim3 block(tile,tile), grid((f.w+tile-1)/tile,(f.h+tile-1)/tile);
    // Device memset is initialization, not a kernel; do not inflate kernel_total.
    CUDA_CHECK(cudaMemset(f.count.get(),0,sizeof(int)));
    kernels(ws,t,t.nms_topk,[&]() {
        maximum_kernel<<<nb,threads>>>(f.response.get(),n,f.maxima.get());
        maximum_kernel<<<1,threads>>>(f.maxima.get(),nb,f.maximum.get());
        nms_kernel<<<grid,block>>>(f.response.get(),f.maximum.get(),f.candidates.get(),f.count.get(),f.w,f.h);
    });
    int count = 0; download(ws,t,&count,f.count.get(),1);
    if (count < 0 || count > n) throw std::runtime_error("invalid CUDA NMS count");
    f.host_keypoints.resize(static_cast<std::size_t>(count));
    download(ws,t,f.host_keypoints.data(),f.candidates.get(),f.host_keypoints.size());
    const auto start = Clock::now();
    auto better = [](const Keypoint& a, const Keypoint& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    };
    if (count > params::kTopK) {
        std::partial_sort(f.host_keypoints.begin(),f.host_keypoints.begin()+params::kTopK,f.host_keypoints.end(),better);
        f.host_keypoints.resize(params::kTopK);
    } else std::sort(f.host_keypoints.begin(),f.host_keypoints.end(),better);
    t.nms_topk += elapsed(start);
    upload(ws,t,f.keypoints.get(),f.host_keypoints.data(),f.host_keypoints.size());
}
void launch_describe(int version, DeviceImage& f) {
    const int n = static_cast<int>(f.host_keypoints.size());
    if (n == 0) return;
    if (version == 1) describe_scalar<<<blocks(n),threads>>>(f.blurred.get(),f.w,f.h,f.keypoints.get(),n,f.descriptors.get());
    else describe_warp<<<blocks(static_cast<std::size_t>(n)*32),threads>>>(f.blurred.get(),f.w,f.h,f.keypoints.get(),n,f.descriptors.get());
}
void extract(int version, Workspace& ws, const Image& img, DeviceImage& f, StageTimes& t) {
    f.reserve(img); upload(ws,t,f.rgb.get(),img.data.data(),img.data.size());
    const int n = img.w*img.h;
    kernels(ws,t,t.gray,[&]() { gray_kernel<<<blocks(n),threads>>>(f.rgb.get(),f.gray.get(),n); });
    kernels(ws,t,t.blur,[&]() { launch_blur(version,f); });
    kernels(ws,t,t.harris,[&]() { launch_harris(version,f); });
    select_keypoints(version,ws,f,t);
    if (!f.host_keypoints.empty()) kernels(ws,t,t.describe,[&]() { launch_describe(version,f); });
}

std::vector<Match> match_features(int version, Workspace& ws, int na, int nb, StageTimes& t) {
    if (na == 0 || nb == 0) return {};
    ws.forward.reserve(na); ws.reverse.reserve(nb); ws.raw_matches.reserve(na);
    kernels(ws,t,t.match,[&]() {
        if (version == 1) {
            nearest_kernel<false><<<(na+match_tile-1)/match_tile,match_tile>>>(ws.a.descriptors.get(),na,ws.b.descriptors.get(),nb,ws.forward.get());
            nearest_kernel<false><<<(nb+match_tile-1)/match_tile,match_tile>>>(ws.b.descriptors.get(),nb,ws.a.descriptors.get(),na,ws.reverse.get());
        } else {
            nearest_kernel<true><<<(na+match_tile-1)/match_tile,match_tile>>>(ws.a.descriptors.get(),na,ws.b.descriptors.get(),nb,ws.forward.get());
            nearest_kernel<true><<<(nb+match_tile-1)/match_tile,match_tile>>>(ws.b.descriptors.get(),nb,ws.a.descriptors.get(),na,ws.reverse.get());
        }
        crosscheck_kernel<<<blocks(na),threads>>>(ws.forward.get(),ws.reverse.get(),na,ws.raw_matches.get());
    });
    std::vector<Match> raw(static_cast<std::size_t>(na));
    download(ws,t,raw.data(),ws.raw_matches.get(),raw.size());
    const auto start = Clock::now();
    std::vector<Match> result;
    for (const auto& m : raw) if (m.idxA >= 0) result.push_back(m);
    t.match += elapsed(start);
    return result;
}

bool ransac( int version, Workspace& ws, const std::vector<Keypoint>& ka,
             const std::vector<Keypoint>& kb, const std::vector<Match>& matches,
             Homography& out, int& ninliers, StageTimes& t) {
    ninliers = 0;
    out = Homography{};
    if (matches.size() < 4 || ka.empty() || kb.empty() ||
        matches.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        ka.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        kb.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) return false;
    const int n_matches = static_cast<int>(matches.size());
    const int iterations = params::kRansacIters;
    ws.s5_kp_a.reserve(ka.size()); ws.s5_kp_b.reserve(kb.size());
    ws.s5_matches.reserve(matches.size());
    ws.src.reserve(matches.size()); ws.dst.reserve(matches.size());
    ws.n_s5_valid.reserve(1); ws.selected_inliers.reserve(matches.size());
    ws.models.reserve(iterations); ws.model_valid.reserve(iterations); ws.scores.reserve(iterations);
    ws.s5_output_h.reserve(1); ws.n_s5_inliers.reserve(1);
    upload(ws,t,ws.s5_kp_a.get(),ka.data(),ka.size());
    upload(ws,t,ws.s5_kp_b.get(),kb.data(),kb.size());
    upload(ws,t,ws.s5_matches.get(),matches.data(),matches.size());
    kernels(ws,t,t.ransac,[&]() {
        prepare_correspondences_kernel<<<1,1>>>(
            ws.s5_kp_a.get(),static_cast<int>(ka.size()),ws.s5_kp_b.get(),static_cast<int>(kb.size()),
            ws.s5_matches.get(),n_matches,ws.src.get(),ws.dst.get(),ws.n_s5_valid.get());
        CUDA_CHECK(cudaGetLastError());
        hypotheses_kernel<<<blocks(iterations),threads>>>(
            ws.src.get(),ws.dst.get(),ws.n_s5_valid.get(),ws.models.get(),ws.model_valid.get(),iterations);
        CUDA_CHECK(cudaGetLastError());
        if (version == 3)
            score_parallel<<<iterations,threads>>>(ws.src.get(),ws.dst.get(),ws.n_s5_valid.get(),
                                                   ws.models.get(),ws.model_valid.get(),ws.scores.get());
        else
            score_serial<<<blocks(iterations),threads>>>(ws.src.get(),ws.dst.get(),ws.n_s5_valid.get(),
                                                         ws.models.get(),ws.model_valid.get(),ws.scores.get(),iterations);
        CUDA_CHECK(cudaGetLastError());
        select_refine_count_kernel<<<1,1>>>(
            ws.src.get(),ws.dst.get(),ws.n_s5_valid.get(),ws.models.get(),ws.model_valid.get(),ws.scores.get(),
            ws.selected_inliers.get(),ws.s5_output_h.get(),ws.n_s5_inliers.get());
    });
    download(ws,t,&out,ws.s5_output_h.get(),1);
    download(ws,t,&ninliers,ws.n_s5_inliers.get(),1);
    if (ninliers < 4) { out = Homography{}; ninliers = 0; return false; }
    return true;
}

struct Canvas { int minx, miny, w, h; Homography inv; };
bool valid_image(const Image& image) {
    return valid_rgb_image(image) && image.w <= 32768 && image.h <= 32768 &&
           static_cast<std::size_t>(image.w)*image.h <= 64u*1024u*1024u;
}
bool canvas(const Image& a, const Image& b, const Homography& input, Canvas& c) {
    Homography h = input;
    double scale = 0;
    for (double value : h.h) { if (!std::isfinite(value)) return false; scale = std::max(scale,std::fabs(value)); }
    if (!(scale > 0)) return false;
    for (double& value : h.h) value /= scale;
    c.inv.h[0] = h.h[4]*h.h[8]-h.h[5]*h.h[7];
    c.inv.h[1] = h.h[2]*h.h[7]-h.h[1]*h.h[8];
    c.inv.h[2] = h.h[1]*h.h[5]-h.h[2]*h.h[4];
    c.inv.h[3] = h.h[5]*h.h[6]-h.h[3]*h.h[8];
    c.inv.h[4] = h.h[0]*h.h[8]-h.h[2]*h.h[6];
    c.inv.h[5] = h.h[2]*h.h[3]-h.h[0]*h.h[5];
    c.inv.h[6] = h.h[3]*h.h[7]-h.h[4]*h.h[6];
    c.inv.h[7] = h.h[1]*h.h[6]-h.h[0]*h.h[7];
    c.inv.h[8] = h.h[0]*h.h[4]-h.h[1]*h.h[3];
    const double t0 = h.h[0]*c.inv.h[0], t1 = h.h[1]*c.inv.h[3], t2 = h.h[2]*c.inv.h[6];
    const double determinant = t0+t1+t2;
    if (!std::isfinite(determinant) || std::fabs(determinant) <= 64*DBL_EPSILON*(std::fabs(t0)+std::fabs(t1)+std::fabs(t2))) return false;
    double minx = 0, miny = 0, maxx = a.w-1.0, maxy = a.h-1.0;
    int sign = 0;
    const double corners[4][2]{{0,0},{b.w-1.0,0},{0,b.h-1.0},{b.w-1.0,b.h-1.0}};
    for (const auto& p : corners) {
        const double z = h.h[6]*p[0]+h.h[7]*p[1]+h.h[8];
        const double magnitude = std::fabs(h.h[6]*p[0])+std::fabs(h.h[7]*p[1])+std::fabs(h.h[8]);
        if (!std::isfinite(z) || std::fabs(z) <= 64*DBL_EPSILON*magnitude) return false;
        const int next_sign = z > 0 ? 1 : -1;
        if (sign != 0 && sign != next_sign) return false;
        sign = next_sign;
        double x = (h.h[0]*p[0]+h.h[1]*p[1]+h.h[2])/z;
        double y = (h.h[3]*p[0]+h.h[4]*p[1]+h.h[5])/z;
        if (!std::isfinite(x) || !std::isfinite(y)) return false;
        const double rx = std::round(x), ry = std::round(y);
        if (std::fabs(x-rx) <= 1e-8+64*DBL_EPSILON*std::fabs(x)) x = rx;
        if (std::fabs(y-ry) <= 1e-8+64*DBL_EPSILON*std::fabs(y)) y = ry;
        minx = std::min(minx,x); miny = std::min(miny,y); maxx = std::max(maxx,x); maxy = std::max(maxy,y);
    }
    minx = std::floor(minx); miny = std::floor(miny); maxx = std::ceil(maxx); maxy = std::ceil(maxy);
    const double width = maxx-minx+1, height = maxy-miny+1;
    if (minx < std::numeric_limits<int>::min() || miny < std::numeric_limits<int>::min() ||
        maxx > std::numeric_limits<int>::max() || maxy > std::numeric_limits<int>::max() ||
        width < 1 || height < 1 || width > 32768 || height > 32768 || width*height > 64.0*1024*1024) return false;
    c.minx = static_cast<int>(minx); c.miny = static_cast<int>(miny); c.w = static_cast<int>(width); c.h = static_cast<int>(height);
    return true;
}

// Exact V0 mask-based largest-rectangle crop, deliberately kept on the host.
void crop_valid(Image& img, const std::vector<uint8_t>& valid) {
    int bx = 0, by = 0, bw = 0, bh = 0;
    std::vector<int> heights(static_cast<std::size_t>(img.w),0), stack;
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) heights[x] = valid[static_cast<std::size_t>(y)*img.w+x] ? heights[x]+1 : 0;
        stack.clear();
        for (int x = 0; x <= img.w; ++x) {
            const int current = x < img.w ? heights[x] : 0;
            while (!stack.empty() && current < heights[stack.back()]) {
                const int bar = stack.back(); stack.pop_back();
                const int left = stack.empty() ? 0 : stack.back()+1;
                const int width = x-left, height = heights[bar];
                const long long area = static_cast<long long>(width)*height, best = static_cast<long long>(bw)*bh;
                if (area > best || (area == best && width > bw)) { bx=left; by=y-height+1; bw=width; bh=height; }
            }
            if (x < img.w) stack.push_back(x);
        }
    }
    if (bw == 0 || bh == 0) { img = Image{}; return; }
    if (bw == img.w && bh == img.h) return;
    std::vector<uint8_t> cropped(static_cast<std::size_t>(bw)*bh*3);
    for (int y = 0; y < bh; ++y)
        std::copy_n(img.data.begin()+(static_cast<std::size_t>(by+y)*img.w+bx)*3,
                    static_cast<std::size_t>(bw)*3,cropped.begin()+static_cast<std::size_t>(y)*bw*3);
    img.w=bw; img.h=bh; img.data=std::move(cropped);
}
bool warp(Workspace& ws, const Image& a, const Image& b, const Homography& h, Image& out, StageTimes& t) {
    auto start = Clock::now(); Canvas c{};
    const bool ok = canvas(a,b,h,c);
    t.warp_blend += elapsed(start);
    if (!ok) return false;
    const std::size_t n = static_cast<std::size_t>(c.w)*c.h;
    ws.panorama.reserve(n*3); ws.coverage.reserve(n);
    kernels(ws,t,t.warp_blend,[&]() {
        warp_kernel<<<dim3((c.w+tile-1)/tile,(c.h+tile-1)/tile),dim3(tile,tile)>>>(
            ws.a.rgb.get(),a.w,a.h,ws.b.rgb.get(),b.w,b.h,c.inv,c.minx,c.miny,c.w,c.h,ws.panorama.get(),ws.coverage.get());
    });
    out.w=c.w; out.h=c.h; out.c=3; out.data.resize(n*3);
    std::vector<uint8_t> valid(n);
    download(ws,t,out.data.data(),ws.panorama.get(),out.data.size()); download(ws,t,valid.data(),ws.coverage.get(),n);
    start = Clock::now(); crop_valid(out,valid); t.warp_blend += elapsed(start);
    return out.w > 0 && out.h > 0;
}

bool pipeline(int version, Workspace& ws, const Image& a, const Image& b, PipelineResult& out) {
    StageTimes& t = out.times;
    if (!valid_image(a) || !valid_image(b)) return false;
    upload_constants(ws,t);
    extract(version,ws,a,ws.a,t); extract(version,ws,b,ws.b,t);
    out.nKpA = static_cast<int>(ws.a.host_keypoints.size()); out.nKpB = static_cast<int>(ws.b.host_keypoints.size());
    if (out.collect_debug) { out.keypointsA=ws.a.host_keypoints; out.keypointsB=ws.b.host_keypoints; }
    const auto matches = match_features(version,ws,out.nKpA,out.nKpB,t);
    out.nMatches = static_cast<int>(matches.size());
    if (out.collect_debug) out.matches=matches;
    if (!ransac(version,ws,ws.a.host_keypoints,ws.b.host_keypoints,matches,out.H,out.nInliers,t)) return false;
    return warp(ws,a,b,out.H,out.panorama,t);
}

bool run(int version, const Image& a, const Image& b, PipelineResult& out) {
    const bool debug = out.collect_debug; out=PipelineResult{}; out.collect_debug=debug;
    const auto total = Clock::now(); bool ok = false;
    try {
        // V3 keeps device allocations and events across warm-up/benchmark runs.
        // Device changes are handled explicitly so cached pointers stay valid.
        if (version == 3) {
            static thread_local std::unique_ptr<Workspace> cached;
            static thread_local int cached_device = -1;
            int device = 0; CUDA_CHECK(cudaGetDevice(&device));
            if (cached && cached_device != device) {
                CUDA_CHECK(cudaSetDevice(cached_device)); cached.reset();
                CUDA_CHECK(cudaSetDevice(device));
            }
            if (!cached) { cached=std::make_unique<Workspace>(); cached_device=device; }
            ok=pipeline(version,*cached,a,b,out);
        } else {
            Workspace ws; ok=pipeline(version,ws,a,b,out);
        }
    } catch (const std::exception& e) { std::cerr << "V" << version << " CUDA pipeline failed: " << e.what() << '\n'; }
    if (!ok) out.panorama=Image{};
    out.times.end_to_end=elapsed(total);
    return ok;
}
} // namespace

bool run_v1(const Image& a, const Image& b, PipelineResult& out) { return run(1,a,b,out); }
bool run_v2(const Image& a, const Image& b, PipelineResult& out) { return run(2,a,b,out); }
bool run_v3(const Image& a, const Image& b, PipelineResult& out) { return run(3,a,b,out); }

bool cuda_selftest(int version) {
    if (version < 1 || version > 3) return false;
    try {
        Workspace ws; StageTimes t; upload_constants(ws,t);
        Image image; image.w=97; image.h=83; image.c=3; image.data.resize(static_cast<std::size_t>(image.w)*image.h*3);
        uint32_t state=741u;
        for (auto& value : image.data) { state=1664525u*state+1013904223u; value=static_cast<uint8_t>(state>>24); }
        ws.a.reserve(image); upload(ws,t,ws.a.rgb.get(),image.data.data(),image.data.size());
        Gray gray,blurred,response;
        v0_rgb_to_gray(image,gray); v0_gaussian_blur(gray,blurred); v0_harris_response(blurred,response);
        kernels(ws,t,t.gray,[&]() { gray_kernel<<<blocks(gray.data.size()),threads>>>(ws.a.rgb.get(),ws.a.gray.get(),static_cast<int>(gray.data.size())); });
        auto check_floats = [&](const char* name, const Gray& ref, const float* device, double relative, double absolute) {
            std::vector<float> actual(ref.data.size()); download(ws,t,actual.data(),device,actual.size());
            for (std::size_t i=0;i<actual.size();++i)
                if (!std::isfinite(actual[i]) || std::fabs(actual[i]-ref.data[i]) > absolute+relative*std::fabs(ref.data[i]))
                    throw std::runtime_error(std::string("CUDA selftest ")+name+" differs at pixel "+std::to_string(i));
        };
        check_floats("gray",gray,ws.a.gray.get(),1e-6,1e-4);
        kernels(ws,t,t.blur,[&]() { launch_blur(version,ws.a); });
        check_floats("blur",blurred,ws.a.blurred.get(),2e-6,2e-4);
        // Isolate Harris arithmetic from preceding-stage numeric differences.
        upload(ws,t,ws.a.blurred.get(),blurred.data.data(),blurred.data.size());
        kernels(ws,t,t.harris,[&]() { launch_harris(version,ws.a); });
        check_floats("harris",response,ws.a.response.get(),2e-5,0.05);
        // NMS/BRIEF must be bit-exact when their inputs are identical.
        upload(ws,t,ws.a.response.get(),response.data.data(),response.data.size());
        std::vector<Keypoint> reference_kp; v0_nms_topk(response,reference_kp);
        if (reference_kp.size()<4) throw std::runtime_error("CUDA selftest fixture has too few keypoints");
        select_keypoints(version,ws,ws.a,t);
        if (reference_kp.size()!=ws.a.host_keypoints.size()) throw std::runtime_error("CUDA NMS count differs");
        for (std::size_t i=0;i<reference_kp.size();++i) {
            const auto& a=reference_kp[i]; const auto& b=ws.a.host_keypoints[i];
            if (a.x!=b.x || a.y!=b.y || a.score!=b.score) throw std::runtime_error("CUDA top-K differs");
        }
        std::vector<Descriptor> reference_desc; v0_describe(blurred,reference_kp,reference_desc);
        kernels(ws,t,t.describe,[&]() { launch_describe(version,ws.a); });
        std::vector<Descriptor> actual_desc(reference_desc.size());
        download(ws,t,actual_desc.data(),ws.a.descriptors.get(),actual_desc.size());
        if (actual_desc.size() && std::memcmp(actual_desc.data(),reference_desc.data(),actual_desc.size()*sizeof(Descriptor)))
            throw std::runtime_error("CUDA BRIEF bits differ");
        // Deliberate duplicate descriptors exercise ratio rejection and tie order.
        std::vector<Descriptor> other=reference_desc;
        if (other.size()>3) other[1]=other[0];
        ws.b.descriptors.reserve(other.size()); upload(ws,t,ws.b.descriptors.get(),other.data(),other.size());
        const auto actual_matches=match_features(version,ws,static_cast<int>(reference_desc.size()),static_cast<int>(other.size()),t);
        std::vector<Match> reference_matches; v0_match(reference_desc,other,reference_matches);
        if (actual_matches.size()!=reference_matches.size()) throw std::runtime_error("CUDA match count differs");
        for (std::size_t i=0;i<actual_matches.size();++i)
            if (actual_matches[i].idxA!=reference_matches[i].idxA || actual_matches[i].idxB!=reference_matches[i].idxB ||
                actual_matches[i].dist!=reference_matches[i].dist) throw std::runtime_error("CUDA matcher differs");
        std::vector<Keypoint> ka,kb; std::vector<Match> correspondences;
        for (int y=0;y<6;++y) for (int x=0;x<8;++x) {
            const int i=static_cast<int>(kb.size()); const float sx=10.0f+x*9, sy=8.0f+y*10;
            kb.push_back({sx,sy,1}); ka.push_back({sx+7,sy+3,1}); correspondences.push_back({i,i,0});
        }
        // Eight wrong matches, alongside forty exact inliers.
        for (int i=40;i<48;++i) { ka[i].x+=20; ka[i].y-=17; }
        Homography estimated{}; int ninliers=0;
        if (!ransac(version,ws,ka,kb,correspondences,estimated,ninliers,t) || ninliers!=40 ||
            std::fabs(estimated.h[2]-7)>1e-6 || std::fabs(estimated.h[5]-3)>1e-6)
            throw std::runtime_error("CUDA RANSAC known translation failed");
        Image small; small.w=10; small.h=10; small.c=3; small.data.assign(300,123);
        ws.a.reserve(small); ws.b.reserve(small);
        upload(ws,t,ws.a.rgb.get(),small.data.data(),small.data.size()); upload(ws,t,ws.b.rgb.get(),small.data.data(),small.data.size());
        const std::array<Homography,3> transformations{{
            Homography{{1,0,0,0,1,0,0,0,1}}, Homography{{1,0,4,0,1,0,0,0,1}}, Homography{{1,0,0,0,1,4,0,0,1}}
        }};
        for (const auto& h: transformations) {
            Image actual,reference; v0_warp_blend(small,small,h,reference);
            if (!warp(ws,small,small,h,actual,t) || actual.w!=reference.w || actual.h!=reference.h || actual.data!=reference.data)
                throw std::runtime_error("CUDA warp/feather/bounds differs");
        }
        // Perspective/bilinear test with nonuniform colour and invalid mask crop.
        for (std::size_t i=0;i<small.data.size();++i) small.data[i]=static_cast<uint8_t>((i*37)%256);
        upload(ws,t,ws.a.rgb.get(),small.data.data(),small.data.size()); upload(ws,t,ws.b.rgb.get(),small.data.data(),small.data.size());
        Homography perspective{{1,0.06,2.25,0.02,1,-0.25,0.001,0.002,1}};
        Image actual,reference; v0_warp_blend(small,small,perspective,reference);
        if (!warp(ws,small,small,perspective,actual,t) || actual.w!=reference.w || actual.h!=reference.h || actual.data!=reference.data)
            throw std::runtime_error("CUDA perspective bilinear/crop differs");
        std::cout << "V" << version << " CUDA stage selftest PASS\n";
        return true;
    } catch (const std::exception& e) { std::cerr << "V" << version << " CUDA selftest failed: " << e.what() << '\n'; return false; }
}
