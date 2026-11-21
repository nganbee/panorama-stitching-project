// Authors: Team 8; implementation assisted by OpenAI Codex.
// Include only V3 internals; a separate test object observes the original V1.
#include "../../core_headers/v0_stages.h"
#include "../../core_headers/v1_stages.h"
#include "../../core_headers/v3_stages.h"
#define PANORAMA_STAGE6_NO_ENTRYPOINTS
#include "../../stage_implementations/v3_stage6.cu"
#undef PANORAMA_STAGE6_NO_ENTRYPOINTS
#include <algorithm>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>

void stage6_reference_raw(const Image&, const Image&, const Homography&,
                          Image&, std::vector<uint8_t>&, int&, int&);

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
Image pattern(int w, int h, unsigned seed) {
    Image image;
    image.w = w; image.h = h;
    image.data.resize(static_cast<size_t>(w) * h * 3);
    std::mt19937 rng(seed);
    for (auto& value : image.data) value = static_cast<uint8_t>(rng() & 255u);
    return image;
}
Homography translation(double x = 0, double y = 0) {
    return Homography{{1,0,x,0,1,y,0,0,1}};
}

void compare_raw(const Image& a, const Image& b, const Homography& h) {
    Image reference;
    std::vector<uint8_t> reference_mask;
    int origin_x=0,origin_y=0;
    stage6_reference_raw(a,b,h,reference,reference_mask,origin_x,origin_y);
    v1_cuda::Buffer<CanvasConfig> device_config(1);
    v1_cuda::launch([&] { prepare_canvas_kernel<<<1,1>>>(a.w,a.h,b.w,b.h,h,device_config.data); });
    CanvasConfig config;
    v1_cuda::copy(&config,device_config.data,sizeof(config),cudaMemcpyDeviceToHost);
    require(config.ok, "Positive raw-canvas case rejected");
    require(config.width==reference.w && config.height==reference.h &&
            config.origin_x==origin_x && config.origin_y==origin_y,"V1/V3 canvas geometry differs");
    const size_t pixels = static_cast<size_t>(config.width) * config.height;
    v1_cuda::Buffer<uint8_t> da(a.data.size()), db(b.data.size());
    v1_cuda::Buffer<uint8_t> texture_canvas(pixels * 3), texture_mask(pixels);
    v1_cuda::copy(da.data, a.data.data(), a.data.size(), cudaMemcpyHostToDevice);
    v1_cuda::copy(db.data, b.data.data(), b.data.size(), cudaMemcpyHostToDevice);
    TextureSource ta(da.data, a.data.size()), tb(db.data, b.data.size());
    const dim3 block(16,16), grid((config.width+15)/16, (config.height+15)/16);
    v1_cuda::launch([&] { warp_blend_texture_kernel<<<grid,block>>>(
        da.data,a.w,a.h,db.data,b.w,b.h,config.inverse,config.origin_x,config.origin_y,
        config.width,config.height,texture_canvas.data,texture_mask.data,ta.view(),tb.view()); });
    std::vector<uint8_t> actual(pixels * 3);
    v1_cuda::copy(actual.data(),texture_canvas.data,actual.size(),cudaMemcpyDeviceToHost);
    require(reference.data == actual, "Uncropped canvas differs from V1");
    actual.resize(pixels);
    v1_cuda::copy(actual.data(),texture_mask.data,pixels,cudaMemcpyDeviceToHost);
    require(reference_mask == actual, "Valid mask differs from V1");
}

void compare(const Image& a, const Image& b, const Homography& h, const char* name) {
    Image cpu, baseline, texture, repeated;
    v0_warp_blend(a,b,h,cpu);
    v1_warp_blend(a,b,h,baseline);
    v3_warp_blend(a,b,h,texture);
    v3_warp_blend(a,b,h,repeated);
    require(valid_rgb_image(cpu) && valid_rgb_image(baseline) && valid_rgb_image(texture),
            "Positive panorama case failed; parity is NOT verified");
    require(cpu.w == texture.w && cpu.h == texture.h && cpu.data.size() == texture.data.size(),
            "V0/V3 crop dimensions differ");
    double mae = 0;
    for (size_t i=0; i<cpu.data.size(); ++i) {
        const int error = std::abs(int(cpu.data[i])-int(texture.data[i]));
        require(error <= 1, "V0/V3 RGB error exceeds 1");
        mae += error;
    }
    require(baseline.w == texture.w && baseline.h == texture.h && baseline.data == texture.data,
            "V1/V3 crop or RGB bytes differ");
    require(texture.w == repeated.w && texture.h == repeated.h && texture.data == repeated.data,
            "V3 is nondeterministic");
    compare_raw(a,b,h);
    std::printf("PASS: %s; canvas/mask/crop exact vs V1; V0 RGB MAE=%.9f\n",
                name, mae/cpu.data.size());
}

void reject(const Image& a, const Image& b, const Homography& h, const char* name) {
    Image baseline, texture;
    v1_warp_blend(a,b,h,baseline); v3_warp_blend(a,b,h,texture);
    require(baseline.data.empty() && texture.data.empty(), "Invalid S6 input accepted");
    std::printf("PASS: expected rejection (%s); no panorama correctness claim\n", name);
}
} // namespace

int main() {
    try {
        const Image a = pattern(37,29,1), b = pattern(31,23,2);
        compare(a,a,translation(),"identity random RGB");
        compare(a,b,translation(9,4),"integer translation");
        compare(a,b,translation(-7,-3),"negative translation");
        compare(a,b,translation(2.5,3.25),"fractional translation / manual bilinear");
        const Homography projective{{1,0.05,4,0.02,1,-2,0.0005,0.0002,1}};
        compare(a,b,projective,"perspective / feather / partial blocks");
        compare(pattern(1,1,3),pattern(1,1,4),translation(),"single pixel");
        compare(pattern(1,513,3),pattern(1,513,4),translation(),"single column");
        compare(pattern(513,1,3),pattern(513,1,4),translation(),"single row");
        const Image large = pattern(257,263,5);
        compare(large,large,translation(63.5,-7.25),"multiple crop reduction groups");
        compare(a,b,translation(90,0),"disjoint valid regions / crop tie ordering");
        Homography singular = translation(); singular.h[0] = singular.h[4] = 0;
        reject(a,b,singular,"singular H");
        Homography nonfinite = translation(); nonfinite.h[2] = std::numeric_limits<double>::infinity();
        reject(a,b,nonfinite,"nonfinite H");
        reject(Image{},b,translation(),"empty image");
        Image malformed = a; malformed.data.pop_back();
        reject(malformed,b,translation(),"malformed RGB buffer");
        compare(a,b,translation(),"repeat after failure / texture cleanup");
        std::puts("PASS: V3 S6 texture verification");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"V3 S6 FAIL: %s\n",error.what());
        return 1;
    }
}
