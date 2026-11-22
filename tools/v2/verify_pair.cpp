// Authors: Nguyen Duy Khanh; verification assisted by OpenAI Codex.
// External sources: none. Untimed S4 and full-pipeline result verification.
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
struct Features { std::vector<Keypoint> kp; std::vector<Descriptor> desc; };
Features extract(const Image& image) {
    // Reuse V1 extraction to supply identical inputs to the three matchers.
    // V2 does not modify or re-test the internal S1-S3 implementations here.
    Gray gray,blur,response;
    Features features;
    v1_rgb_to_gray(image,gray); v1_gaussian_blur(gray,blur);
    v1_harris_response(blur,response); v1_nms_topk(response,features.kp);
    v1_describe(blur,features.kp,features.desc);
    return features;
}
void compare(const Image& b, const PipelineResult& ref,
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
        const Features ga=extract(a), gb=extract(b);
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
            compare(b,results[0],results[2],"V0_V2",false);
            compare(b,results[1],results[2],"V1_V2",true);
            bool ok=true; for (const auto& item:checks) ok &= item.second;
            status=ok ? "pass" : "correctness_failed";
        }
    } catch (const std::exception& e) { error=e.what(); status="error"; }
    try { write(argv[3],status,error); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
    std::cout << "Verification: " << status << '\n';
    return status=="pass" ? 0 : 1;
}
