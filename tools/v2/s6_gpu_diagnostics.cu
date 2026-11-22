// Authors: Team 8; V2 verification assisted by OpenAI Codex.
// External sources: none. Replays original S6 kernels outside benchmark timing.
// Link this TU INSTEAD OF v1_stage6.cu in the verification binary only.
#include "s6_diagnostics.h"
#include "../../src/stage_implementations/v1_stage6.cu"

V2S6Diagnostics v2_gpu_s6_diagnostics(const Image& A, const Image& B, const Homography& H) {
    V2S6Diagnostics out;
    if (!valid_rgb_image(A) || !valid_rgb_image(B)) return out;
    v1_cuda::Buffer<CanvasConfig> config_device(1);
    v1_cuda::launch([&] { prepare_canvas_kernel<<<1, 1>>>(A.w, A.h, B.w, B.h, H, config_device.data); });
    CanvasConfig config{};
    v1_cuda::copy(&config, config_device.data, sizeof(config), cudaMemcpyDeviceToHost);
    if (!config.ok) return out;
    out.width = config.width; out.height = config.height;
    out.origin_x = config.origin_x; out.origin_y = config.origin_y;
    const size_t pixels = size_t(out.width) * out.height;
    v1_cuda::Buffer<uint8_t> a(A.data.size()), b(B.data.size()), rgb(pixels * 3), mask(pixels);
    v1_cuda::Buffer<int> heights(pixels), stacks(pixels);
    v1_cuda::Buffer<GpuRect> rows(out.height), rect(1);
    v1_cuda::copy(a.data, A.data.data(), A.data.size(), cudaMemcpyHostToDevice);
    v1_cuda::copy(b.data, B.data.data(), B.data.size(), cudaMemcpyHostToDevice);
    const dim3 block(16, 16), grid((out.width + 15) / 16, (out.height + 15) / 16);
    v1_cuda::launch([&] { warp_blend_kernel<<<grid, block>>>(a.data, A.w, A.h, b.data, B.w, B.h,
        config.inverse, out.origin_x, out.origin_y, out.width, out.height, rgb.data, mask.data); });
    v1_cuda::launch([&] { build_crop_histograms_kernel<<<(out.width + 127) / 128, 128>>>(
        mask.data, out.width, out.height, heights.data); });
    v1_cuda::launch([&] { find_crop_rows_kernel<<<(out.height + 127) / 128, 128>>>(
        heights.data, out.width, out.height, stacks.data, rows.data); });
    v1_cuda::launch([&] { reduce_crop_rows_kernel<<<1, kCropReduceThreads>>>(rows.data, out.height, rect.data); });
    GpuRect crop{};
    v1_cuda::copy(&crop, rect.data, sizeof(crop), cudaMemcpyDeviceToHost);
    out.mask.resize(pixels);
    v1_cuda::copy(out.mask.data(), mask.data, pixels, cudaMemcpyDeviceToHost);
    out.crop_x = crop.x; out.crop_y = crop.y; out.crop_w = crop.w; out.crop_h = crop.h;
    out.ok = crop.w > 0 && crop.h > 0;
    return out;
}
