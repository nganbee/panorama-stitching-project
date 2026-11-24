// Authors: Nguyen Duy Khanh; benchmark assisted by OpenAI Codex.
// External sources: none. Uses production pipelines; all I/O outside timing.
#include "../../core_headers/report_io.h"
#include "../../libraries/stb_image.h"
#include "../../libraries/stb_image_write.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <cuda_runtime.h>

namespace {
struct Field { const char* name; double StageTimes::* member; };
const Field fields[] = {
    {"gray",&StageTimes::gray},{"blur",&StageTimes::blur},{"harris",&StageTimes::harris},
    {"nms_topk",&StageTimes::nms_topk},{"describe",&StageTimes::describe},{"match",&StageTimes::match},
    {"ransac",&StageTimes::ransac},{"warp_blend",&StageTimes::warp_blend},{"h2d",&StageTimes::h2d},
    {"d2h",&StageTimes::d2h},{"kernel_total",&StageTimes::kernel_total},{"end_to_end",&StageTimes::end_to_end}};
Image load(const char* path) {
    Image image; int c;
    auto* p=stbi_load(path,&image.w,&image.h,&c,3);
    if (!p) throw std::runtime_error(std::string("Cannot load ")+path);
    image.data.assign(p,p+size_t(image.w)*image.h*3); stbi_image_free(p); return image;
}
}
int main(int argc,char** argv) {
    if (argc!=7) { std::cerr << "Usage: benchmark_v2_pair left right output.png version runs report.json\n"; return 2; }
    std::string error;
    try {
        const int version=std::stoi(argv[4]), runs=std::stoi(argv[5]);
        const auto pipeline=get_version(version);
        if (!pipeline || runs<5) throw std::runtime_error("Invalid version or runs < 5");
        int device; cudaDeviceProp properties{};
        if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&properties,device)!=cudaSuccess)
            throw std::runtime_error("CUDA device unavailable");
        std::cout << "GPU: " << properties.name << " version=" << version << " runs=" << runs << '\n';
        const Image a=load(argv[1]), b=load(argv[2]);
        PipelineResult warmup;
        if (!pipeline(a,b,warmup)) throw std::runtime_error("Warm-up pipeline failed");
        warmup=PipelineResult{};
        std::vector<StageTimes> samples;
        StageTimes mean{}; PipelineResult last;
        for (int i=0; i<runs; ++i) {
            PipelineResult result;
            if (!pipeline(a,b,result)) throw std::runtime_error("Timed pipeline failed at run "+std::to_string(i));
            samples.push_back(result.times);
            for (auto field:fields) mean.*(field.member) += result.times.*(field.member)/runs;
            last=std::move(result);
        }
        if (!stbi_write_png(argv[3],last.panorama.w,last.panorama.h,3,last.panorama.data.data(),last.panorama.w*3))
            throw std::runtime_error("Cannot write panorama");
        std::ostringstream base;
        write_metrics_json(base,version,runs,a,b,last,mean,argv[3]);
        std::string json=base.str(); json.erase(json.find_last_of('}'));
        std::ofstream out(argv[6]);
        if (!out) throw std::runtime_error("Cannot write benchmark report");
        out << json << ",\"gpu_name\":" << json_string(properties.name) << ",\"samples_ms\":[" << std::setprecision(17);
        for (size_t i=0; i<samples.size(); ++i) {
            if (i) out << ','; out << '{'; bool first=true;
            for (auto field:fields) {
                if (!first) out << ','; first=false;
                out << json_string(field.name) << ':' << samples[i].*(field.member);
            }
            out << '}';
        }
        out << "]}\n";
        if (!out) throw std::runtime_error("Failed writing benchmark report");
        return 0;
    } catch (const std::exception& e) { error=e.what(); }
    std::cerr << error << '\n';
    std::ofstream out(argv[6]);
    if (out) out << "{\"status\":\"failed\",\"error\":" << json_string(error) << "}\n";
    return 1;
}
