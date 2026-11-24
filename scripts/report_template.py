#!/usr/bin/env python3
"""Regenerate editable report templates (clears saved outputs: run deliberately).
Authors: Group 08 (names/IDs pending), AI-assisted; no external code adapted.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source})


md("""# CSC14120 — Group 08 — Midterm: Proposal và V0
Notebook build trực tiếp source đã nộp, chạy kiểm thử và hiển thị các kết quả CPU thật. Tên/MSSV, ảnh chụp và đăng ký địa điểm, link dataset, nhận xét riêng và khai báo AI chờ nhóm xác nhận. Ảnh tổng hợp được ghi nhãn rõ, không thay cho ảnh địa điểm.

**Colab CPU:** upload/giải nén toàn bộ `Group08_Midterm.zip` vào `/content`, rồi Runtime → Run all. Cần `src/`, `tools/`, `scripts/`, `docs/`. Khi manifest có link ZIP công khai, notebook tự tải dataset còn thiếu; link Drive cần gdown, chỉ cài nếu runtime chưa có. Với Drive riêng tư, mount trước rồi đặt `PROJECT_SOURCE` hoặc `DATASET_ARCHIVE` trỏ tới source/ZIP đã mount. Trên máy local, mở notebook tại thư mục repo có Python và compiler C++. Dataset thật dùng `dataset_manifest.json` theo file mẫu; copy `docs/team_info.example.json` thành `team_info.json` để điền thông tin nhóm. Sau Run all, dùng nút export/strict package ở cuối; bản notebook source sao chép lúc setup không được coi là output hiện tại.

Không viết lại C++ trong cell, không điền timing/speedup bằng tay; mọi số đo trong bảng và chart đến từ output của executable. Chưa có dataset thì cell báo phần còn thiếu và tiếp tục kiểm thử synthetic. Midterm yêu cầu kế hoạch GPU ở mục 3.7, chưa yêu cầu chạy CUDA hay số liệu GPU.""")
code('''from pathlib import Path
import os, sys, json, shutil, subprocess, platform, hashlib, tempfile
from IPython.display import display, Markdown
from PIL import Image
import pandas as pd

candidates = [Path(os.environ["PROJECT_SOURCE"])] if os.environ.get("PROJECT_SOURCE") else []
candidates += [Path.cwd(), Path("/content"), Path("/content/panorama-stitching-project")]
if Path("/content").is_dir():
    candidates += [p.parent.parent for p in Path("/content").glob("*/scripts/build.py")]
SOURCE = next((p.resolve() for p in candidates if (p/"src").is_dir() and (p/"scripts/build.py").is_file()), None)
if SOURCE is None:
    raise RuntimeError("Giải nén gói nguồn vào /content hoặc đặt PROJECT_SOURCE; cần src/, tools/, scripts/, docs/.")
os.environ.setdefault("PROJECT_SOURCE",str(SOURCE))  # Rerun tiếp tục dùng source gốc, không lấy workspace cũ.
ROOT = SOURCE
try:
    import google.colab
    IS_COLAB=Path("/content").is_dir()
except ImportError: IS_COLAB=False
if IS_COLAB:
    ROOT = Path(tempfile.mkdtemp(prefix="Group08_work_",dir="/content"))
    if SOURCE != ROOT.resolve():
        for folder in ("src", "tools", "scripts", "docs"):
            if (SOURCE/folder).is_dir(): shutil.copytree(SOURCE/folder, ROOT/folder, dirs_exist_ok=True)
        for name in ("CMakeLists.txt", "README.md", "Group08_Midterm.ipynb", "dataset_manifest.example.json", "dataset_manifest.json", "team_info.json"):
            if (SOURCE/name).is_file(): shutil.copy2(SOURCE/name, ROOT/name)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT/"scripts"))
for module in ("provenance","resolve_dataset","export_colab","prepare_dataset","benchmark","report_utils","package_submission"):
    sys.modules.pop(module,None)  # Rerun phải đọc helper của source vừa copy.
OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
if IS_COLAB:
    from export_colab import sync_source_notebook
    # Bản source không được mang output cũ sang lần chạy mới.
    report_source=ROOT/"Group08_Midterm.ipynb"
    source_document=json.loads(report_source.read_text(encoding="utf-8"))
    for cell in source_document["cells"]:
        if cell.get("cell_type")=="code": cell["execution_count"]=None; cell["outputs"]=[]
    source_document.setdefault("metadata",{})["execution_status"]="source_copy"
    report_source.write_text(json.dumps(source_document,ensure_ascii=False,indent=1),encoding="utf-8")
    try: sync_source_notebook(report_source)
    except Exception as error:
        print("Colab không hỗ trợ export source nội bộ:",str(error))
        print("Source gốc vẫn được build. Nếu đã sửa cell, tải/upload notebook mới cùng source trước Run all.")
    if (SOURCE/"dataset_manifest.json").is_file():
        from resolve_dataset import copy_referenced_images
        print("Ảnh có sẵn được copy theo manifest:",copy_referenced_images(SOURCE/"dataset_manifest.json",ROOT))
def run(args):
    command = [str(a) for a in args]
    result = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
    print("$", " ".join(command))
    if result.stdout: print(result.stdout)
    if result.stderr: print(result.stderr)
    if result.returncode: raise RuntimeError(f"Command failed ({result.returncode})")
    return result.stdout
def binary(name):
    return ROOT/"build"/(name + (".exe" if os.name == "nt" else ""))
from provenance import source_sha256, implementation_sha256
print(json.dumps({"source_root":str(SOURCE), "platform":platform.platform(), "processor":platform.processor(),
                  "python":sys.version, "source_sha256":source_sha256(ROOT),
                  "implementation_sha256":implementation_sha256(ROOT),
                  "report_inputs_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                    for name in ("team_info.json","dataset_manifest.json") if (ROOT/name).is_file()}}, indent=2))
run([sys.executable, "scripts/build.py", "--compiler", "g++"])
run(["g++", "--version"])''')
md("""## 3.1 Thông tin nhóm và kế hoạch
Nickname từ DOCX được giữ nguyên; không suy đoán họ tên/MSSV. Nhóm điền `team_info.json`, xác nhận tác giả source và điều chỉnh phân công nếu cần. Bảng kế hoạch dưới bao gồm từng stage và V0 đến V3.""")
code('''team_file = ROOT/"team_info.json"
TEAM = json.loads((team_file if team_file.is_file() else ROOT/"docs/team_info.example.json").read_text(encoding="utf-8"))
display(pd.DataFrame([{"Tên gọi":m["nickname"], "Họ tên":m.get("name") or "NHÓM TỰ ĐIỀN",
                       "MSSV":m.get("student_id") or "NHÓM TỰ ĐIỀN", "Phụ trách":m["responsibility"]} for m in TEAM["members"]]))
display(pd.DataFrame([
    {"Người":"Thịnh", "Stage":"S1-S2", "V0/Midterm":"methods, timing, header", "V1":"thread/pixel, Harris/NMS", "V2":"shared-memory filter", "V3":"filter/memory reuse", "Final":"Introduction, slides, đọc chéo S3-S4"},
    {"Người":"Khánh", "Stage":"S3-S4", "V0/Midterm":"methods, timing, header", "V1":"thread/keypoint, query matching", "V2":"descriptor tile, popcount", "V3":"warp/shuffle reduction", "Final":"References, AI, failure case, đọc chéo S1-S2"},
    {"Người":"Thụ", "Stage":"S5", "V0/Midterm":"synthetic, error, timing/Amdahl/GPU plan", "V1":"hypothesis/match scoring", "V2":"block reduction, cache", "V3":"warp consensus reduction", "Final":"Nsight, performance charts, đọc chéo S6"},
    {"Người":"Ngân", "Stage":"S6/integration", "V0/Midterm":"dataset, report/build/package", "V1":"thread/output pixel", "V2":"warp/blend cache", "V3":"data reuse/transfer", "Final":"gallery, README, demo/video, đọc chéo S5"}
]))
print("Source authors đã được nhóm xác nhận:", TEAM.get("source_authors_confirmed",False))''')
md("""## 3.2 Hiểu bài toán
Hai ảnh RGB của cùng một cảnh có vùng chồng lấp được ghép thành panorama. S1 chuyển grayscale và làm mịn để giảm nhiễu; S2 tìm Harris corner và loại điểm trùng; S3 mô tả patch bằng BRIEF; S4 so Hamming, ratio test và mutual cross-check; S5 chọn consensus RANSAC và refine homography; S6 inverse-warp, bilinear và feather vùng giao nhau.

```text
A → S1 → S2 → S3 ─┐
                 ├→ S4 → S5 (B→A) → S6 → panorama
B → S1 → S2 → S3 ─┘
```

Homography luôn ánh xạ **B → A**. S6 dùng nghịch đảo H để lấy mẫu B tại từng output pixel; mask phân biệt nền trống với pixel hợp lệ rồi crop rectangle hợp lệ. Ảnh ít texture, pattern lặp, xoay/scale lớn hoặc parallax có thể thất bại. BRIEF hiện không chuẩn hóa orientation/scale; homography không mô tả tốt mọi cảnh có nhiều độ sâu. Nhóm cần minh họa giới hạn bằng cặp challenging thật.""")
md("""## 3.3 Phương pháp lựa chọn
Tham số chính lấy từ binary qua `--dump-params`, cùng nguồn `common.h` cho mọi version.

| Stage | Phương pháp | Chất lượng và suitability GPU |
|---|---|---|
| S1 | RGB→gray, Gaussian separable | Giảm nhiễu; từng pixel độc lập, filter thuận lợi shared-memory. |
| S2 | Harris, NMS, deterministic top-K | Giữ góc mạnh; gradient/response/NMS song song pixel, top-K cần sort/reduction ổn định. |
| S3 | BRIEF nhị phân, pattern seed cố định | Descriptor gọn/tái lập; thread/keypoint; không bất biến quay/scale. |
| S4 | Hamming brute-force, Lowe ratio, cross-check | Loại match mơ hồ; all-pairs dễ tile/popcount; tie và thứ nhì phải xử lý đúng. |
| S5 | RANSAC, normalized linear homography solve, inlier refinement | Chống outlier, normalization giúp ổn định; score hypothesis/match dễ song song. |
| S6 | Inverse warp, bilinear, feather, valid mask/crop | Pixel output độc lập; trọng số biên dương, canvas pixel-inclusive tránh mất hàng/cột. |

Các version giữ nguyên phương pháp/ngưỡng khi benchmark để so sánh công bằng. Chất lượng được kiểm tra bằng sai số corner reprojection và panorama; dataset thật chờ nhóm cung cấp.""")
code('''PARAMETERS = json.loads(subprocess.check_output([str(binary("stitch")), "--dump-params"], text=True, cwd=ROOT))
display(PARAMETERS)
display(pd.DataFrame([{"Tham số":k,"Giá trị":v} for k,v in PARAMETERS.items() if k!="available_versions"]))''')
md("""## 3.4 Sequential version V0
V0 dùng C++ đơn luồng trong `src/stage_implementations/v0_stage*.cpp` và `src/integration/v0_pipeline.cpp`, theo `common.h`, `v0_stages.h`, `versions.h`. Không có OpenMP, `std::thread` hay SIMD tự viết. Compiler vẫn có thể tối ưu tự động.

Setup biên dịch source đã nộp, không chứa bản C++ khác trong cell. CLI loại image I/O và debug-artifact generation khỏi timing; S1-S3 cộng cả A/B, giữ riêng stage/end-to-end và feature/match/inlier counts. Self-test chạy binary vừa build, exit code lỗi làm notebook dừng.""")
code('''for name in ("stage12_selftest", "stage34_selftest", "stage6_selftest", "synthetic_homography_test", "cuda_host_math_selftest"):
    run([binary(name)])
print("V0 regressions: PASS (process trả exit code thành công).")''')
md("""## 3.5 V0 results và profiling
### Correctness toàn pipeline
Chương trình tạo cặp ảnh tổng hợp với transform/seed đã biết. Pipeline tự tạo keypoint/descriptor/match, không bơm match đúng vào pipeline. S5 self-test riêng phía trên bổ sung outlier để kiểm tra consensus. Sai số H được tính bằng khoảng cách corner ảnh B khi chiếu bằng H_estimated và H_ground_truth, báo mean/max pixel; không so thẳng phần tử H vốn có thể khác hệ số scale.""")
code('''SYNTHETIC = json.loads(run([binary("synthetic_pipeline_test"), OUT/"synthetic"]))
if not SYNTHETIC.get("all_passed"): raise RuntimeError("Synthetic correctness failed")
display(pd.DataFrame([{k:t.get(k) for k in ("name","passed","mean_corner_error_px","max_corner_error_px","matches","inliers")} for t in SYNTHETIC["tests"]]))
for test in SYNTHETIC["tests"]:
    display(Markdown("**Synthetic: "+test["name"]+"**"))
    display(pd.DataFrame({"H_ground_truth":test["H_ground_truth"],"H_estimated":test["H_estimated"]}))
    with Image.open(test["panorama"]) as source:
        preview=source.copy(); preview.thumbnail((1000,450)); display(preview)''')
md("""### Timing, hotspot, Amdahl và panorama theo site
`prepare_dataset.py` tạo small/medium/large bằng chung hệ số scale cho A/B, giữ aspect ratio và ghi actual dimensions. Ảnh fit trong target box, không kéo giãn nếu aspect khác. `benchmark.py` ghi JSON/CSV/panorama, stdout/stderr cho từng pair/resolution/version. Các bảng/chart/hotspot/Amdahl đọc trực tiếp JSON đo được.

Mặc định Amdahl chọn S1-S4 và S6; S5 có kế hoạch CUDA nhưng cần profiling địa điểm thật để xét overhead so với lợi ích. Hotspot từ ảnh tổng hợp chỉ chứng minh công cụ chạy, không dự đoán performance của địa điểm. Mỗi CLI có warm-up và số lần đo được hiển thị trong output.""")
code('''from prepare_dataset import prepare, load_manifest, requirements
from benchmark import benchmark
from report_utils import show_benchmark
RUNS=5; ACCELERATE=["S1","S2","S3","S4","S6"]
DATASET_FILE=ROOT/"dataset_manifest.json"; HAS_DATASET=DATASET_FILE.is_file()
if HAS_DATASET:
    from resolve_dataset import resolve_dataset
    print(json.dumps(resolve_dataset(DATASET_FILE,archive_path=os.environ.get("DATASET_ARCHIVE"),
                                     allow_install=IS_COLAB),ensure_ascii=False,indent=2))
    MANIFEST=load_manifest(DATASET_FILE)
    prepared_file=prepare(DATASET_FILE,OUT/"prepared")
    benchmark(prepared_file,binary("stitch"),OUT/"benchmark",runs=RUNS,accelerate=ACCELERATE)
    show_benchmark(OUT/"benchmark/benchmark.json")
else:
    print("PENDING NHÓM: dataset_manifest.json và ảnh thật; timing/gallery địa điểm chưa có.")''')
code('''# Ba size, đủ sáu stage, chỉ là synthetic scaling smoke khi chưa có dataset nhóm.
if not HAS_DATASET:
    test=SYNTHETIC["tests"][0]
    manifest={"schema_version":1,"synthetic":True,"dataset_url":"",
              "sites":[{"site_id":"synthetic_demo","display_name":"Synthetic smoke","registration":"Không phải site"}],
              "pairs":[{"pair_id":"synthetic_demo","site_id":"synthetic_demo","kind":"standard",
                        "left":str(Path(test["left"]).resolve()),"right":str(Path(test["right"]).resolve())}]}
    manifest_file=OUT/"synthetic/dataset_manifest.synthetic.json"
    manifest_file.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    smoke_prepared=prepare(manifest_file,OUT/"synthetic_prepared")
    benchmark(smoke_prepared,binary("stitch"),OUT/"synthetic_benchmark",runs=RUNS,accelerate=ACCELERATE)
    show_benchmark(OUT/"synthetic_benchmark/benchmark.json")
else: print("Correctness synthetic đã chạy; timing dùng dataset thật ở cell trên.")''')
md("""## 3.6 Dataset
PDF giữa kỳ yêu cầu ít nhất ba địa điểm đã đăng ký, một cặp standard mỗi nơi và ít nhất một challenging toàn dataset. DOCX có một dòng tóm tắt rộng hơn; bảng phân công DOCX khớp PDF. Có thể thêm challenging cho mỗi site nếu giảng viên yêu cầu. Dataset archive phải có ảnh gốc và prepared pairs, tải qua link output dưới.

Manifest kiểm tra schema, pair ID duy nhất, site tồn tại và input images có thật. `raw_left/raw_right` là đường dẫn ảnh gốc trong dataset archive. Notebook tự lấy ZIP công khai khi ảnh thiếu, kiểm tra đường dẫn ZIP rồi chỉ copy các file ảnh manifest tham chiếu; hỗ trợ thư mục ảnh tương đối bất kỳ. `DATASET_ARCHIVE` dùng ZIP local/Drive riêng tư đã mount. EXIF-transpose trước resize; output RGB và actual dimensions ghi vào prepared manifest. Ảnh synthetic không thay thế địa điểm đăng ký.""")
code('''if HAS_DATASET:
    print("Link dataset:",MANIFEST.get("dataset_url") or "NHÓM TỰ ĐIỀN")
    display(pd.DataFrame(MANIFEST["sites"])); display(pd.DataFrame(MANIFEST["pairs"]))
    print("Dataset còn chờ:",requirements(MANIFEST))
    print("Nhận xét chất lượng:",TEAM.get("dataset_quality_notes") or "NHÓM TỰ ĐIỀN")
    print("Failure case:",TEAM.get("failure_case_analysis") or "NHÓM TỰ ĐIỀN từ cặp challenging")
else:
    print("PENDING NHÓM: site names/registration, ảnh gốc, standard/challenging, link dataset, nhận xét.")
    display(json.loads((ROOT/"dataset_manifest.example.json").read_text(encoding="utf-8")))''')
md("""## 3.7 GPU plan
Profiling 3.5 xếp hạng stage theo resolution. Tập Amdahl mặc định S1-S4/S6 cần được nhóm xác nhận trên dataset thật; thêm S5 nếu lợi ích vượt overhead. Midterm chỉ yêu cầu nêu stage dự định tăng tốc, mapping V1, ý tưởng tối ưu và rủi ro; chưa cần biên dịch/chạy CUDA hoặc có số liệu GPU. Source CUDA tùy chọn phục vụ phần Final, không phải điều kiện để hoàn thành báo cáo giữa kỳ.

| Stage | Mapping V1 đề xuất | Optimization V2/V3 | Rủi ro |
|---|---|---|---|
| S1 | thread/pixel, gray/blur từng pass | constant filter/shared halo, buffer reuse | borders, bandwidth, occupancy |
| S2 | thread/pixel gradient/response/NMS; max reduction/compaction/top-K | tiled/integral accumulation, stable top-K | ordering/tie, transfer/serial cost |
| S3 | thread/keypoint, pattern chung | image cache/constant pattern, descriptor layout | random access/bounds/bit determinism |
| S4 | thread/query hoặc warp/query | popcount, descriptor tile, shuffle reduction | best/second tie, ratio/cross-check |
| S5 | hypothesis/warp/block, match/thread để score | block reduction, cache coordinates | degeneracy, precision, seed, CPU refinement |
| S6 | thread/output pixel | bilinear cache/texture, data reuse | inclusive canvas, mask/crop/feather, host postprocess |

Nguồn hiện cài V2 shared-memory halo cho blur/Harris, warp BRIEF và tiled matching; V3 thêm block RANSAC scoring/reduction và cache scratch/event. CPU vẫn thực hiện deterministic top-K, match compaction, seeded sampling/model selection/refinement và canvas/crop. Các hướng texture/fusion/asynchronous streams trong kế hoạch là thử nghiệm sau, chưa được khẳng định đã cài.

Các hướng tối ưu dự kiến: bộ nhớ/tile và giữ dữ liệu; matching/reduction theo warp; giảm launch/transfer bằng reuse/fusion. Khi làm Final, dùng cùng input/GPU session, đo H2D/D2H/kernel/end-to-end và kiểm tra correctness trước benchmark. Nsight Systems tìm overhead/transfer; Nsight Compute xem bandwidth/occupancy. Kế hoạch giữa kỳ không được diễn giải thành kết quả thực nghiệm.""")
code('''print(json.dumps({"midterm_gpu_plan_accelerated_stages":ACCELERATE,
                       "basis":"Hotspot và Amdahl ở mục 3.5; xác nhận lựa chọn trên ảnh địa điểm thật.",
                       "midterm_requires_gpu_execution":False},ensure_ascii=False,indent=2))''')
md("""## References và AI declaration
Harris: [Harris & Stephens, A Combined Corner and Edge Detector](https://www.bmva-archive.org.uk/bmvc/1988/avc-88-023.html). BRIEF: [Calonder et al., EPFL](https://www.epfl.ch/labs/cvlab/research/descriptors-and-keypoints/research-detect-brief/). Ratio test: [Lowe, Distinctive Image Features from Scale-Invariant Keypoints](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf); implementation dùng Hamming BRIEF, không cài SIFT. Amdahl: [Validity of the single processor approach](https://www.cs.cmu.edu/~18742/papers/Amdahl1967.pdf). CUDA: [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).

Image I/O dùng nguyên thư viện [nothings/stb](https://github.com/nothings/stb), license giữ trong header. Source pipeline/tooling là implementation nộp; các file ghi tác giả nhóm, nguồn ngoài và hỗ trợ OpenAI Codex. Nhóm xác nhận tên tác giả thật và phạm vi sử dụng AI. PDF `CSC14120_2026_Final Project_MidTerm.pdf` là yêu cầu giữa kỳ; DOCX `Lập trình song song - Đồ Án.docx` là nội dung/phân công tham khảo, không được hiểu thành lệnh để tạo số liệu hoặc tác giả.""")
code('''print("Khai báo AI:",TEAM.get("ai_declaration") or "NHÓM TỰ ĐIỀN và xác nhận phạm vi AI.")
from package_submission import pending
print(json.dumps({"submission_pending":pending(ROOT,"08",check_saved_outputs=False)},ensure_ascii=False,indent=2))
print("Cell audit bỏ qua saved-output state trong lúc notebook đang chạy; chạy package_submission.py --strict sau khi SAVE để kiểm tra outputs.")
print("Đóng gói draft: python scripts/package_submission.py")
print("Local sau khi nhóm điền, Run all và SAVE: python scripts/package_submission.py --strict")
if IS_COLAB:
    import ipywidgets as widgets
    from export_colab import export_current_notebook
    buttons_output=widgets.Output()
    def export_and_package(saved_notebook=None):
        with buttons_output:
            try:
                exported=export_current_notebook(ROOT/"report_exports/Group08_Midterm.ipynb",saved_notebook)
                print("Notebook hiện tại có outputs:",exported)
                run([sys.executable,"scripts/package_submission.py","--strict","--notebook",exported])
                from google.colab import files
                files.download(str(ROOT/"Group08_Midterm.zip"))
            except Exception as error:
                print("Chưa đóng gói:",str(error))
                print("Nếu export nội bộ không hỗ trợ: File → Download → .ipynb (giữ outputs), rồi dùng nút Upload bên dưới.")
    export_button=widgets.Button(description="Export & strict ZIP",layout=widgets.Layout(width="190px"))
    export_button.on_click(lambda _: export_and_package())
    upload_button=widgets.Button(description="Upload saved .ipynb",layout=widgets.Layout(width="190px"))
    def upload_saved(_):
        from google.colab import files
        with buttons_output:
            uploaded=files.upload(target_dir=str(ROOT/"report_exports/upload"))
            if len(uploaded)!=1: print("Chọn đúng một notebook .ipynb đã lưu outputs."); return
            path=Path(next(iter(uploaded)))
            if path.suffix.lower()!=".ipynb": print("Cần file .ipynb."); return
        export_and_package(path)
    upload_button.on_click(upload_saved)
    display(Markdown("**Sau khi Run all hoàn tất**, bấm Export & strict ZIP. Nếu Colab không hỗ trợ API nội bộ, tải notebook có outputs rồi Upload saved .ipynb. Strict chỉ đóng gói khi dataset/thông tin nhóm đã đủ."))
    display(widgets.HBox([export_button,upload_button]),buttons_output)''')


def write_notebook(path, items):
    for index, cell in enumerate(items):
        cell["id"] = f"group08-{index:03}"
    notebook = {"cells": items, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_notebook(ROOT/"Group08_Midterm.ipynb", cells)
    print("Report regenerated. Run scripts/execute_notebook.py to save real outputs.")
