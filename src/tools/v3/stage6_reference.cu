// Test-only access to the original V1 raw canvas, mask and crop rectangle.
// Rename only the exported wrapper so the normal V1 object can also be linked.
#include "../../core_headers/v1_stages.h"
#include <array>
#define v1_warp_blend stage6_reference_unused_wrapper
#include "../../stage_implementations/v1_stage6.cu"
#undef v1_warp_blend

std::array<int, 4> stage6_reference_raw(const Image& a, const Image& b, const Homography& h,
                                        Image& canvas, std::vector<uint8_t>& mask,
                                        int& origin_x, int& origin_y) {
    v1_cuda::Buffer<CanvasConfig> device_config(1);
    v1_cuda::launch([&] { prepare_canvas_kernel<<<1,1>>>(
        a.w,a.h,b.w,b.h,h,device_config.data); });
    CanvasConfig config;
    v1_cuda::copy(&config,device_config.data,sizeof(config),cudaMemcpyDeviceToHost);
    if (!config.ok) throw std::runtime_error("V1 reference rejected positive canvas case");
    origin_x=config.origin_x; origin_y=config.origin_y;
    canvas.w=config.width; canvas.h=config.height; canvas.c=3;
    const size_t pixels=static_cast<size_t>(canvas.w)*canvas.h;
    canvas.data.resize(pixels*3); mask.resize(pixels);
    v1_cuda::Buffer<uint8_t> da(a.data.size()),db(b.data.size()),output(pixels*3),valid(pixels);
    v1_cuda::copy(da.data,a.data.data(),a.data.size(),cudaMemcpyHostToDevice);
    v1_cuda::copy(db.data,b.data.data(),b.data.size(),cudaMemcpyHostToDevice);
    const dim3 block(16,16),grid((canvas.w+15)/16,(canvas.h+15)/16);
    v1_cuda::launch([&] { warp_blend_kernel<<<grid,block>>>(
        da.data,a.w,a.h,db.data,b.w,b.h,config.inverse,config.origin_x,config.origin_y,
        canvas.w,canvas.h,output.data,valid.data); });
    v1_cuda::copy(canvas.data.data(),output.data,canvas.data.size(),cudaMemcpyDeviceToHost);
    v1_cuda::copy(mask.data(),valid.data,pixels,cudaMemcpyDeviceToHost);

    v1_cuda::Buffer<int> heights(pixels), stacks(pixels);
    v1_cuda::Buffer<GpuRect> rows(canvas.h), rect(1);
    constexpr int threads = 128;
    v1_cuda::launch([&] { build_crop_histograms_kernel<<<(canvas.w + threads - 1) / threads, threads>>>(
        valid.data, canvas.w, canvas.h, heights.data); });
    v1_cuda::launch([&] { find_crop_rows_kernel<<<(canvas.h + threads - 1) / threads, threads>>>(
        heights.data, canvas.w, canvas.h, stacks.data, rows.data); });
    v1_cuda::launch([&] { reduce_crop_rows_kernel<<<1, kCropReduceThreads>>>(
        rows.data, canvas.h, rect.data); });
    GpuRect crop{};
    v1_cuda::copy(&crop, rect.data, sizeof(crop), cudaMemcpyDeviceToHost);
    return {crop.x, crop.y, crop.w, crop.h};
}
