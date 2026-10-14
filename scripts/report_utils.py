"""Report displays from measured outputs. Authors: Group 08 (names/IDs pending), AI-assisted.
External sources: IPython/Pandas/Matplotlib/Pillow APIs; no implementation copied.
"""
import json
import hashlib
from pathlib import Path

from IPython.display import Markdown, display
from PIL import Image

STAGES = ["S1", "S2", "S3", "S4", "S5", "S6"]


def show_benchmark(path, gallery=True):
    import matplotlib.pyplot as plt
    import pandas as pd
    path = Path(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"benchmark_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "source_sha256": result.get("source_sha256"),
                      "implementation_sha256": result.get("implementation_sha256")}, indent=2))
    kind = "ẢNH TỔNG HỢP — kiểm thử kỹ thuật, không phải dataset địa điểm" if result["synthetic"] else "DATASET NHÓM"
    display(Markdown(f"**{kind}.** Máy đo: `{result['platform']}` / `{result['processor']}`. {result['timing_convention']}"))
    rows = result["rows"]
    successful = [r for r in rows if r["status"] == "ok"]
    if not successful:
        display(Markdown("Chưa có lần chạy thành công; kiểm tra lỗi bên dưới."))
        display(pd.DataFrame(rows))
        return
    columns = ["pair_id", "resolution", "version", "left_width", "left_height", "right_width", "right_height",
               "runs"] + [f"{s}_ms" for s in STAGES] + ["end_to_end_ms", "matches", "inliers", "speedup_vs_v0"]
    frame = pd.DataFrame(successful)
    display(frame.reindex(columns=columns).round(4))
    # Each bar is an observed pair/size/version; no averaging across unrelated scenes.
    timings = frame[[f"{s}_ms" for s in STAGES]].copy()
    timings.columns = STAGES
    multiple_pairs = len({r["pair_id"] for r in successful}) > 1
    timings.index = [(f"{r['pair_id']}\n" if multiple_pairs else "")
                     + f"{r['resolution']} / V{r['version']}\n"
                     + f"A {r['left_width']}×{r['left_height']} | B {r['right_width']}×{r['right_height']}"
                     for r in successful]
    ax = timings.plot.bar(stacked=True, figsize=(max(8, len(successful)*1.25), 5.5), rot=0 if len(successful)<=3 else 25)
    ax.set_ylabel("Thời gian trung bình (ms)")
    chart_kind = "Ảnh tổng hợp — kiểm thử kỹ thuật" if result["synthetic"] else "Dataset nhóm"
    ax.set_title(f"Per-stage timing\n{chart_kind}", pad=14)
    ax.legend(ncol=6, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()
    analyses = []
    for row in successful:
        amdahl = row.get("amdahl", {})
        limit = amdahl.get("maximum_speedup")
        analyses.append({"pair": row["pair_id"], "resolution": row["resolution"], "version": row["version"],
                         "hot_stages_descending": " > ".join(row["hot_stages"]),
                         "accelerated_stages": ",".join(result["accelerated_stages"]),
                         "p": amdahl.get("p"), "Amdahl_limit": "∞" if amdahl.get("limit_is_infinite") else limit,
                         "status": amdahl.get("status"), "reason": amdahl.get("reason", "")})
    display(pd.DataFrame(analyses))
    display(Markdown("`p = Σ(thời gian các stage được chọn) / end_to_end`; giới hạn lý tưởng `1/(1-p)`. "
                     "Đây là trần lý thuyết của baseline CPU, chưa tính overhead truyền dữ liệu hay tốc độ GPU thực tế. "
                     "Khi chọn đủ các stage, phần điều phối CPU rất nhỏ nên trần có thể rất lớn; không diễn giải nó như speedup đạt được."))
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        display(Markdown("**Failure cases đo được**"))
        display(pd.DataFrame(failed)[["pair_id", "resolution", "version", "error"]])
    if gallery:
        # One panorama per site, preferring the medium standard pair and V0.
        sites = sorted({r["site_id"] for r in successful})
        for site in sites:
            candidates = [r for r in successful if r["site_id"] == site]
            candidates.sort(key=lambda r: (r["version"] != 0, r["kind"] != "standard", r["resolution"] != "medium"))
            row = candidates[0]
            display(Markdown(f"**{site}** — `{row['pair_id']}`, {row['resolution']}, V{row['version']}"))
            with Image.open(row["panorama_path"]) as source:
                preview = source.copy()
                preview.thumbnail((1100, 650))
                display(preview)
