// File: synthetic_pipeline_test.cpp
// Authors: Team 8; implementation assisted by OpenAI Codex.
// Description: Actual image pairs through all six stages, compared with known H.
// External sources: stb_image_write by Sean Barrett, https://github.com/nothings/stb.
#include "versions.h"
#include "report_io.h"
#include "stb_image_write.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

namespace {
bool make_dirs(const std::string& path) {
    for(size_t i=1;i<=path.size();++i) {
        if(i<path.size() && path[i]!='/' && path[i]!='\\') continue;
        const std::string part=path.substr(0,i);
        if(part.empty() || part.back()==':') continue;
#ifdef _WIN32
        _mkdir(part.c_str());
#else
        mkdir(part.c_str(),0755);
#endif
    }
    struct stat info{};
    return stat(path.c_str(),&info)==0 && (info.st_mode&S_IFDIR);
}
uint32_t hash_pixel(int x,int y) {
    uint32_t v=uint32_t(x)*0x9e3779b9u ^ uint32_t(y)*0x85ebca6bu ^ 14120u;
    v^=v>>16; v*=0x7feb352du; v^=v>>15; v*=0x846ca68bu; v^=v>>16;
    return v;
}
double world_color(int x,int y,int c) {
    // Smooth a deterministic world texture before sampling either image.
    double sum=0;
    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx)
        sum+=double((hash_pixel(x+dx,y+dy)>>(8*c))&255u);
    return sum/9.0;
}
uint8_t sample(double x,double y,int c) {
    int x0=int(std::floor(x)),y0=int(std::floor(y));
    double dx=x-x0,dy=y-y0;
    const double v=(world_color(x0,y0,c)*(1-dx)+world_color(x0+1,y0,c)*dx)*(1-dy)
                  +(world_color(x0,y0+1,c)*(1-dx)+world_color(x0+1,y0+1,c)*dx)*dy;
    return uint8_t(std::lround(v));
}
Image image_from_h(const Homography& H) {
    Image a; a.w=320; a.h=240; a.c=3; a.data.resize(size_t(a.w)*a.h*3);
    for(int y=0;y<a.h;++y) for(int x=0;x<a.w;++x) {
        double z=H.h[6]*x+H.h[7]*y+H.h[8];
        double wx=(H.h[0]*x+H.h[1]*y+H.h[2])/z;
        double wy=(H.h[3]*x+H.h[4]*y+H.h[5])/z;
        for(int c=0;c<3;++c) a.data[(size_t(y)*a.w+x)*3+c]=sample(wx,wy,c);
    }
    return a;
}
void matrix(std::ostream& o,const Homography& H) {
    o<<'['; for(int i=0;i<9;++i) {if(i)o<<',';o<<H.h[i];} o<<']';
}
bool test(const char* name,const Homography& truth,PipelineFn pipeline,
          const std::string& directory,std::ostream& o) {
    const Homography identity={{1,0,0,0,1,0,0,0,1}};
    const Image A=image_from_h(identity),B=image_from_h(truth);
    PipelineResult result;
    const bool ran=pipeline(A,B,result);
    double mean=0,max_error=0;
    if(ran) {
        for(int y:{0,B.h-1}) for(int x:{0,B.w-1}) {
            double tz=truth.h[6]*x+truth.h[7]*y+truth.h[8];
            double rz=result.H.h[6]*x+result.H.h[7]*y+result.H.h[8];
            double dx=(truth.h[0]*x+truth.h[1]*y+truth.h[2])/tz
                     -(result.H.h[0]*x+result.H.h[1]*y+result.H.h[2])/rz;
            double dy=(truth.h[3]*x+truth.h[4]*y+truth.h[5])/tz
                     -(result.H.h[3]*x+result.H.h[4]*y+result.H.h[5])/rz;
            double error=std::hypot(dx,dy);
            if(!std::isfinite(error)) error=1e30;
            mean+=error/4; max_error=std::max(max_error,error);
        }
    } else mean=max_error=1e30;
    bool passed=ran && result.nInliers>=20 && max_error<2.0 && valid_rgb_image(result.panorama);
    std::string left,right,panorama;
    if(!directory.empty()) {
        left=directory+'/'+name+"_left.png"; right=directory+'/'+name+"_right.png";
        panorama=directory+'/'+name+"_panorama.png";
        passed &= stbi_write_png(left.c_str(),A.w,A.h,3,A.data.data(),A.w*3)!=0;
        passed &= stbi_write_png(right.c_str(),B.w,B.h,3,B.data.data(),B.w*3)!=0;
        if(ran) passed &= stbi_write_png(panorama.c_str(),result.panorama.w,result.panorama.h,3,
                                      result.panorama.data.data(),result.panorama.w*3)!=0;
    }
    o<<"{\"name\":"<<json_string(name)<<",\"passed\":"<<(passed?"true":"false")
     <<",\"mean_corner_error_px\":"<<mean<<",\"max_corner_error_px\":"<<max_error
     <<",\"matches\":"<<result.nMatches<<",\"inliers\":"<<result.nInliers
     <<",\"H_ground_truth\":";matrix(o,truth);o<<",\"H_estimated\":";matrix(o,result.H);
    o<<",\"left\":"<<json_string(left)<<",\"right\":"<<json_string(right)
     <<",\"panorama\":"<<json_string(panorama)<<'}';
    return passed;
}
}
int main(int argc,char** argv) {
    std::string directory;
    int version=0;
    for(int i=1;i<argc;++i) {
        if(std::string(argv[i])=="--version" && i+1<argc) {
            char* end=nullptr; const long v=std::strtol(argv[++i],&end,10);
            if(*end || v<0 || v>3) return 2;
            version=int(v);
        } else if(directory.empty()) directory=argv[i];
        else return 2;
    }
    PipelineFn pipeline=get_version(version);
    if(!pipeline || (!directory.empty() && !make_dirs(directory))) return 2;
    const Homography translation={{1,0,176,0,1,0,0,0,1}};
    const Homography projective={{1,0.008,172,-0.006,1,2,0.00001,-0.000008,1}};
    std::cout<<std::setprecision(17)<<"{\"schema_version\":1,\"version\":"<<version<<",\"tests\":[";
    bool ok=test("translation",translation,pipeline,directory,std::cout);
    std::cout<<',';ok=test("mild_projective",projective,pipeline,directory,std::cout)&&ok;
    Image invalid; PipelineResult r;
    ok=(!pipeline(invalid,invalid,r))&&ok;
    std::cout<<"],\"all_passed\":"<<(ok?"true":"false")
             <<",\"invalid_input_rejected\":"<<(!valid_rgb_image(r.panorama)?"true":"false")
             <<",\"status\":\""<<(ok?"ok":"failed")<<"\"}\n";
    return ok?0:1;
}
