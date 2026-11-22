// Authors: Nguyen Duy Khanh; verification assisted by OpenAI Codex.
// External sources: none. Untimed real-image verification against PLAN.md.
#include "../../src/core_headers/versions.h"
#include "../../src/core_headers/v0_stages.h"
#include "../../src/core_headers/v1_stages.h"
#include "../../src/core_headers/v2_stages.h"
#include "../../src/core_headers/report_io.h"
#include "../../src/libraries/stb_image.h"
#include "verify_helpers.h"
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
std::map<std::string, bool> checks;
std::map<std::string, double> metrics;
void check(const std::string& label, bool ok) {
    checks[label] = ok;
    std::cout << (ok ? "PASS " : "FAIL ") << label << '\n';
}
Image load(const char* path) {
    Image image; int channels;
    auto* data = stbi_load(path, &image.w, &image.h, &channels, 3);
    if (!data) throw std::runtime_error(std::string("Cannot load image: ") + path);
    image.data.assign(data, data + size_t(image.w)*image.h*3);
    stbi_image_free(data); return image;
}
void gray_check(const std::string& name, const Gray& a, const Gray& b, double atol, double rtol) {
    bool ok = a.w==b.w && a.h==b.h && a.data.size()==b.data.size();
    double maximum = 0;
    if (ok) for (size_t i=0; i<a.data.size(); ++i) {
        const double error = std::fabs(double(a.data[i])-b.data[i]);
        maximum = std::max(maximum,error);
        ok &= std::isfinite(error) && error <= atol + rtol*std::fabs(a.data[i]);
    }
    metrics[name+"_max_abs_error"] = maximum;
    check(name,ok);
}
struct Features { std::vector<Keypoint> kp; std::vector<Descriptor> desc; };
void extract(const Image& image, const std::string& side, Features& c, Features& g) {
    Gray cg,gg,cb,gb,cr,gr;
    v0_rgb_to_gray(image,cg); v1_rgb_to_gray(image,gg);
    gray_check(side+"_gray",cg,gg,1e-4,1e-6);
    // Isolate each stage on the same input, then check propagation end-to-end.
    v0_gaussian_blur(cg,cb); v1_gaussian_blur(cg,gb);
    gray_check(side+"_blur_same_input",cb,gb,2e-4,2e-6);
    v1_gaussian_blur(gg,gb);
    v0_harris_response(cb,cr); v1_harris_response(cb,gr);
    gray_check(side+"_harris_same_input",cr,gr,5e-2,2e-5);
    v0_nms_topk(cr,c.kp); v1_nms_topk(cr,g.kp);
    check(side+"_keypoints_same_input",v2_equal_keypoints(c.kp,g.kp));
    v1_harris_response(gb,gr); v1_nms_topk(gr,g.kp);
    check(side+"_keypoints_propagated",v2_equal_keypoints(c.kp,g.kp));
    v0_describe(cb,c.kp,c.desc); v1_describe(gb,g.kp,g.desc);
    std::vector<Descriptor> same_input;
    v0_describe(gb,g.kp,same_input);
    check(side+"_descriptor_same_input",v2_equal_descriptors(same_input,g.desc));
    check(side+"_descriptor_propagated",v2_equal_descriptors(c.desc,g.desc));
}
void compare(const Image& a, const Image& b, const PipelineResult& ref,
             const PipelineResult& target, const std::string& label, bool exact) {
    check(label+"_keypoints",v2_equal_keypoints(ref.keypointsA,target.keypointsA) &&
                                   v2_equal_keypoints(ref.keypointsB,target.keypointsB));
    check(label+"_ordered_matches",v2_equal_matches(ref.matches,target.matches));
    check(label+"_inliers",ref.nInliers==target.nInliers);
    const double error = v2_projection_error(ref.H,target.H,b.w,b.h,ref.keypointsB);
    metrics[label+"_projection_error_px"] = error;
    check(label+"_projection",error <= (exact ? 0 : 1e-3));
    const bool shape = ref.panorama.w==target.panorama.w && ref.panorama.h==target.panorama.h &&
                       ref.panorama.data.size()==target.panorama.data.size();
    check(label+"_panorama_shape",shape);
    double maximum = 0, sum = 0;
    if (shape) for (size_t i=0; i<ref.panorama.data.size(); ++i) {
        const int d = std::abs(int(ref.panorama.data[i])-int(target.panorama.data[i]));
        maximum = std::max(maximum,double(d)); sum += d;
    }
    metrics[label+"_pixel_max_error"] = shape ? maximum : -1;
    metrics[label+"_pixel_mae"] = shape && !ref.panorama.data.empty() ? sum/ref.panorama.data.size() : -1;
    check(label+"_panorama_pixels",shape && maximum <= (exact ? 0 : 1));
    // V1 and V2 share the original S6 implementation. Inspect actual kernels'
    // masks and crop outside timing instead of guessing validity from black RGB.
    const auto reference = exact ? v2_gpu_s6_diagnostics(a,b,ref.H) : v2_cpu_s6_diagnostics(a,b,ref.H);
    const auto gpu = v2_gpu_s6_diagnostics(a,b,target.H);
    metrics[label+"_canvas_width"] = gpu.width;
    metrics[label+"_canvas_height"] = gpu.height;
    metrics[label+"_crop_x"] = gpu.crop_x; metrics[label+"_crop_y"] = gpu.crop_y;
    metrics[label+"_crop_width"] = gpu.crop_w; metrics[label+"_crop_height"] = gpu.crop_h;
    double mask_errors = -1;
    if (v2_equal_canvas(reference,gpu)) {
        mask_errors = 0;
        for (size_t i=0; i<reference.mask.size(); ++i) mask_errors += reference.mask[i]!=gpu.mask[i];
    }
    metrics[label+"_mask_mismatch_pixels"] = mask_errors;
    check(label+"_canvas",v2_equal_canvas(reference,gpu));
    check(label+"_valid_mask",v2_equal_canvas(reference,gpu) && reference.mask==gpu.mask);
    check(label+"_crop",v2_equal_canvas(reference,gpu) && v2_equal_crop(reference,gpu) &&
          gpu.crop_w==target.panorama.w && gpu.crop_h==target.panorama.h &&
          reference.crop_w==ref.panorama.w && reference.crop_h==ref.panorama.h);
}
void write(const char* path, const std::string& status, const std::string& error) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write verification JSON");
    out << std::setprecision(17) << "{\"status\":" << json_string(status)
        << ",\"error\":" << json_string(error) << ",\"checks\":{";
    bool first=true;
    for (const auto& item:checks) {
        if (!first) out << ','; first=false;
        out << json_string(item.first) << ':' << (item.second ? "true" : "false");
    }
    out << "},\"metrics\":{"; first=true;
    for (const auto& item:metrics) {
        if (!first) out << ','; first=false;
        out << json_string(item.first) << ':';
        if (std::isfinite(item.second)) out << item.second; else out << "null";
    }
    out << "}}\n";
}
}
int main(int argc, char** argv) {
    if (argc!=4) { std::cerr << "Usage: verify_v2_pair left right report.json\n"; return 2; }
    std::string status="error", error;
    try {
        const Image a=load(argv[1]), b=load(argv[2]);
        Features ca,ga,cb,gb;
        extract(a,"A",ca,ga); extract(b,"B",cb,gb);
        std::vector<Match> m0,m1,m2;
        v0_match(ga.desc,gb.desc,m0); v1_match(ga.desc,gb.desc,m1); v2_match(ga.desc,gb.desc,m2);
        check("S4_same_input_V0_V1",v2_equal_matches(m0,m1));
        check("S4_same_input_V0_V2",v2_equal_matches(m0,m2));
        check("S4_same_input_V1_V2",v2_equal_matches(m1,m2));
        PipelineResult results[3]; bool success[3];
        for (int v=0; v<3; ++v) {
            results[v].collect_debug=true;
            success[v]=get_version(v)(a,b,results[v]);
            checks["pipeline_v"+std::to_string(v)+"_succeeded"]=success[v];
        }
        if (!success[0] || !success[1] || !success[2]) {
            status = (success[0]==success[1] && success[1]==success[2]) ? "consistent_pipeline_failure" : "pipeline_status_mismatch";
            error = "Panorama not verified; inspect matches, homography and stderr. Equal failure is not correctness PASS.";
        } else {
            compare(a,b,results[0],results[2],"V0_V2",false);
            compare(a,b,results[1],results[2],"V1_V2",true);
            bool ok=true; for (const auto& item:checks) ok &= item.second;
            status=ok ? "pass" : "correctness_failed";
        }
    } catch (const std::exception& e) { error=e.what(); status="error"; }
    try { write(argv[3],status,error); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
    std::cout << "Verification: " << status << '\n';
    return status=="pass" ? 0 : 1;
}
