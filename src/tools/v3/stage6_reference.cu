// Compare the crop scanners on the same mask produced by the shared warp.
#include "../../core_headers/v3_stages.h"
#define PANORAMA_STAGE6_NO_ENTRYPOINTS
#include "../../stage_implementations/v1_stage6.cu"
#undef PANORAMA_STAGE6_NO_ENTRYPOINTS

bool stage6_crop_rects_match(const Image& a, const Image& b, const Homography& h) {
    v1_cuda::Buffer<CanvasConfig> device_config(1);
    v1_cuda::launch([&] { prepare_canvas_kernel<<<1,1>>>(
        a.w,a.h,b.w,b.h,h,device_config.data); });
    CanvasConfig config;
    v1_cuda::copy(&config,device_config.data,sizeof(config),cudaMemcpyDeviceToHost);
    if (!config.ok) throw std::runtime_error("V1 reference rejected positive canvas case");
    const int width = config.width, height = config.height;
    const size_t pixels = static_cast<size_t>(width) * height;
    v1_cuda::Buffer<uint8_t> da(a.data.size()),db(b.data.size()),output(pixels*3),valid(pixels);
    v1_cuda::copy(da.data,a.data.data(),a.data.size(),cudaMemcpyHostToDevice);
    v1_cuda::copy(db.data,b.data.data(),b.data.size(),cudaMemcpyHostToDevice);
    const dim3 block(16,16),grid((width+15)/16,(height+15)/16);
    v1_cuda::launch([&] { warp_blend_kernel<<<grid,block>>>(
        da.data,a.w,a.h,db.data,b.w,b.h,config.inverse,config.origin_x,config.origin_y,
        width,height,output.data,valid.data); });

    v1_cuda::Buffer<int> heights(pixels), stacks(pixels);
    v1_cuda::Buffer<GpuRect> rows(height), rect(1);
    const GpuRect baseline = find_crop(valid.data, width, height, heights.data, stacks.data,
                                      rows.data, rect.data, launch_baseline_crop_rows);
    const GpuRect optimized = find_crop(valid.data, width, height, heights.data, stacks.data,
                                       rows.data, rect.data, stage6_detail::launch_compressed_crop_rows);
    return baseline.x == optimized.x && baseline.y == optimized.y &&
           baseline.w == optimized.w && baseline.h == optimized.h;
}
