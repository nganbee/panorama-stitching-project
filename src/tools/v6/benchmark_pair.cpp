// Authors: Nguyen Duy Khanh; assisted by OpenAI Codex.
// Benchmark production pipelines; image I/O and verification are outside timing.
#include "../../core_headers/report_io.h"
#include "../../libraries/stb_image.h"
#include "../../libraries/stb_image_write.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <array>

namespace {
struct Field { const char* name; double StageTimes::* member; };
const Field fields[] = {
    {"gray",&StageTimes::gray},{"blur",&StageTimes::blur},{"harris",&StageTimes::harris},
    {"nms_topk",&StageTimes::nms_topk},{"describe",&StageTimes::describe},{"match",&StageTimes::match},
    {"ransac",&StageTimes::ransac},{"warp_blend",&StageTimes::warp_blend},{"h2d",&StageTimes::h2d},
    {"d2h",&StageTimes::d2h},{"kernel_total",&StageTimes::kernel_total},{"end_to_end",&StageTimes::end_to_end}};
Image load(const char* path) {
    Image image; int channels;
    auto* data=stbi_load(path,&image.w,&image.h,&channels,3);
    if (!data) throw std::runtime_error(std::string("Cannot load ")+path);
    image.data.assign(data,data+size_t(image.w)*image.h*3);
    stbi_image_free(data); return image;
}
struct Measurement {
    int version;
    PipelineFn pipeline;
    std::vector<StageTimes> samples;
    std::vector<int> positions;
    PipelineResult last;
    explicit Measurement(int v):version(v),pipeline(get_version(v)) {
        if (!pipeline) throw std::runtime_error("Unsupported version");
    }
    void warmup(const Image& a,const Image& b) {
        PipelineResult result;
        if (!pipeline(a,b,result)) throw std::runtime_error("Warm-up failed for V"+std::to_string(version));
    }
    void measure(const Image& a,const Image& b,int position) {
        PipelineResult result;
        if (!pipeline(a,b,result)) throw std::runtime_error("Timed pipeline failed for V"+std::to_string(version));
        samples.push_back(result.times); positions.push_back(position); last=std::move(result);
    }
    void write(const Image& a,const Image& b,const std::string& png,const std::string& report,
               const std::string& gpu,bool paired) const {
        if (!stbi_write_png(png.c_str(),last.panorama.w,last.panorama.h,3,
                            last.panorama.data.data(),last.panorama.w*3))
            throw std::runtime_error("Cannot write panorama");
        StageTimes mean{};
        for (const auto& sample:samples) {
            for (auto field:fields) mean.*(field.member) += sample.*(field.member)/samples.size();
            mean.h2d_bytes += sample.h2d_bytes/samples.size();
            mean.d2h_bytes += sample.d2h_bytes/samples.size();
        }
        std::ostringstream base;
        write_metrics_json(base,version,int(samples.size()),a,b,last,mean,png);
        std::string json=base.str(); json.erase(json.find_last_of('}'));
        std::ofstream out(report);
        if (!out) throw std::runtime_error("Cannot write report");
        out << json << ",\"gpu_name\":" << json_string(gpu)
            << ",\"measurement_mode\":" << json_string(paired ? "same_process_interleaved" : "single_version")
            << ",\"samples_ms\":[" << std::setprecision(17);
        for (size_t i=0;i<samples.size();++i) {
            if (i) out << ',';
            out << '{'; bool first=true;
            for (auto field:fields) {
                if (!first) out << ','; first=false;
                out << json_string(field.name) << ':' << samples[i].*(field.member);
            }
            out << '}';
        }
        out << "],\"sample_positions\":[";
        for (size_t i=0;i<positions.size();++i) { if (i) out << ','; out << positions[i]; }
        out << "]}\n";
        if (!out) throw std::runtime_error("Failed writing report");
    }
};
void failed(const std::string& path,const std::string& error) {
    std::ofstream out(path);
    if (out) out << "{\"status\":\"failed\",\"error\":" << json_string(error) << "}\n";
}
}
int main(int argc,char** argv) {
    const bool paired = argc==8 && std::string(argv[1])=="--compare";
    if (!paired && argc!=7) {
        std::cerr << "Usage: benchmark_v6_pair left right output.png version runs report.json\n"
                  << "   or: benchmark_v6_pair --compare left right result_dir case runs first_version\n";
        return 2;
    }
    std::vector<std::string> reports;
    try {
        const int runs=std::stoi(argv[paired ? 6 : 5]);
        if (runs<5)
            throw std::runtime_error("Need at least 5 timed samples");
        int device; cudaDeviceProp properties{};
        if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&properties,device)!=cudaSuccess)
            throw std::runtime_error("CUDA device unavailable");
        // Both versions use the same host images and the same selected device/context.
        const Image a=load(argv[paired ? 2 : 1]), b=load(argv[paired ? 3 : 2]);
        if (!paired) {
            reports.push_back(argv[6]);
            Measurement measured(std::stoi(argv[4]));
            std::cout << "GPU: " << properties.name << " version=" << measured.version << " runs=" << runs << '\n';
            measured.warmup(a,b);
            for (int i=0;i<runs;++i) measured.measure(a,b,i);
            measured.write(a,b,argv[3],argv[6],properties.name,false);
        } else {
            const std::string directory=argv[4], name=argv[5];
            reports={directory+"/"+name+"_v5.json",directory+"/"+name+"_v6.json"};
            const int first=std::stoi(argv[7]);
            if (first!=5 && first!=6) throw std::runtime_error("First version must be 5 or 6");
            std::array<Measurement,2> measured{Measurement(5),Measurement(6)};
            const int start=first==5 ? 0 : 1;
            std::cout << "GPU: " << properties.name << " versions=5,6 runs_per_version=" << runs
                      << " mode=same_process_interleaved\n";
            measured[start].warmup(a,b); measured[1-start].warmup(a,b);
            std::vector<int> order;
            for (int round=0;round<runs;++round) {
                const int lead=(start+round)%2;
                for (int turn=0;turn<2;++turn) {
                    const int which=(lead+turn)%2;
                    measured[which].measure(a,b,int(order.size()));
                    order.push_back(measured[which].version);
                }
            }
            for (int i=0;i<2;++i) {
                std::cout << "GPU: " << properties.name << " version=" << measured[i].version
                          << " samples=" << measured[i].samples.size() << '\n';
                measured[i].write(a,b,directory+"/panoramas/"+name+"_v"+std::to_string(measured[i].version)+".png",
                                  reports[i],properties.name,true);
            }
            std::ofstream manifest(directory+"/"+name+"_paired.json");
            manifest << "{\"status\":\"ok\",\"measurement_mode\":\"same_process_interleaved\","
                     << "\"runs_per_version\":" << runs << ",\"warmup_runs_per_version\":1,\"order\":[";
            for (size_t i=0;i<order.size();++i) { if (i) manifest << ','; manifest << order[i]; }
            manifest << "]}\n";
            if (!manifest) throw std::runtime_error("Cannot write paired measurement manifest");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        if (!paired && reports.empty()) reports.push_back(argv[6]);
        if (paired && reports.empty()) {
            const std::string prefix=std::string(argv[4])+"/"+argv[5];
            reports={prefix+"_v5.json",prefix+"_v6.json"};
        }
        for (const auto& path:reports) failed(path,e.what());
        if (paired) failed(std::string(argv[4])+"/"+argv[5]+"_paired.json",e.what());
        return 1;
    }
}
