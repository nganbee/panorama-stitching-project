# Panorama stitching — Group 08

Sequential C++ V0 plus optional CUDA V1–V3 use one source tree and one CLI. The Midterm notebook builds submitted sources, runs CPU correctness checks, collects six-stage timings at three sizes, shows panoramas and calculates hotspots/Amdahl from actual JSON output.

The group must supply verified names/IDs/authors, registered-site photographs, dataset link, quality/failure analysis and AI declaration. CUDA correctness, timing, speedup and Nsight results require a real GPU. See [completion checklist](docs/completion_checklist.md). Synthetic measurements are labelled and do not count as the real-site dataset.

## CPU setup and build

A C++17 compiler and Python are required. Report/data tooling uses Pillow, pandas, matplotlib and IPython. Local notebook execution additionally uses ipykernel and jupyter_client; Colab supplies these dependencies.

```bash
python scripts/build.py --compiler g++
python scripts/execute_notebook.py Group08_Midterm.ipynb
```

Build creates `build/stitch` (Windows: `build/stitch.exe`) and the regression/debug tools. It records compiler version and exact build commands in `build/build_info.json`. CMake is also available:

```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake --config Release
ctest --test-dir build-cmake --output-on-failure -C Release
```

To run the notebook from this GitHub branch in Colab, open `Group08_Midterm.ipynb`, add a code cell with this command, and run it once before Runtime → Run all:

```python
!git clone --depth 1 --branch colab-ready https://github.com/nganbee/panorama-stitching-project.git /content/panorama-stitching-project
```

For a private repository, authenticate in Colab before cloning, or use the complete source ZIP/Drive folder and set `PROJECT_SOURCE`. Before **Run all**, create `dataset_manifest.json` from `dataset_manifest.example.json` and `team_info.json` from `docs/team_info.example.json`, then fill the real site registrations, public dataset ZIP link, member names/IDs, and team declarations. These team-specific files are ignored by Git. Setup copies source into a fresh workspace and compiles it. Missing photographs are retrieved from the public dataset ZIP link. Direct HTTP(S) ZIP links and public Google Drive ZIP file links are supported; a Drive download installs `gdown>=5.2,<6` only if it is absent. For private Drive data, mount it first and set `DATASET_ARCHIVE` to the mounted ZIP. The current revision has local execution coverage; a fresh Colab session still needs to be checked with the team's actual shared dataset. See [report workflow](docs/report_workflow.md).

Copy the team-specific config templates in Colab, then edit them with the actual values before running the notebook:

```python
!cp /content/panorama-stitching-project/dataset_manifest.example.json /content/panorama-stitching-project/dataset_manifest.json
!cp /content/panorama-stitching-project/docs/team_info.example.json /content/panorama-stitching-project/team_info.json
```

## Run and inspect

```bash
build/stitch data/left.jpg data/right.jpg out/panorama.png --version 0 --runs 5 --json out/metrics.json
build/stitch --dump-params
build/stitch data/left.jpg data/right.jpg out/panorama.png --version 0 --debug-prefix out/debug --json out/metrics.json
```

On Windows append `.exe`. Timing uses one warm-up and at least five measured runs, excludes image load/save and debug generation, and sums S1–S3 over both images. Homography maps B to A. S6 uses inverse bilinear warp, feather blending with normalized weights and a zero-distance border fallback, a validity mask and crop. JSON contains all six stages/end-to-end, feature counts, H and dimensions; failure is reported rather than inventing an identity result. Debug output contains H, keypoint overlays and match lines (inliers green, rejected matches red).

## Correctness checks

```bash
build/stage12_selftest
build/stage34_selftest
build/stage6_selftest
build/synthetic_homography_test
build/synthetic_pipeline_test out/synthetic
python tools/test_reporting.py
python scripts/test_colab_helpers.py
```

Stage tests cover preprocessing/features, deterministic BRIEF/matching edge cases, homography with outliers/degenerate inputs, and canvas/border/mask blending. The full-pipeline synthetic test creates image pairs for known translation and mild perspective transforms, detects/matches features itself and reports mean/max corner reprojection error with ground truth. It also checks invalid inputs.

## Dataset and benchmark

Copy `dataset_manifest.example.json` to `dataset_manifest.json`, fill site registration/link and actual images. Copy `docs/team_info.example.json` to `team_info.json` and fill verified team fields. [Dataset contract](docs/dataset.md) describes all keys and resize rules.

```bash
python scripts/prepare_dataset.py dataset_manifest.json --validate-only --strict
python scripts/prepare_dataset.py dataset_manifest.json --output out/prepared
python scripts/benchmark.py out/prepared/manifest.prepared.json --executable build/stitch --runs 5
```

Each pair uses the same uniform scale for both images into small/medium/large bounding boxes; aspect ratio is preserved and actual dimensions are recorded. Benchmark writes `out/benchmark/benchmark.json`, CSV, panoramas and logs. The notebook derives tables, charts, hotspot rankings and Amdahl limits from these outputs. Native sources, Python report tooling, notebook cell sources, prepared inputs and benchmark outputs are fingerprinted to reject stale results during strict packaging. Outside the notebook, resolve missing public photographs with `python scripts/resolve_dataset.py dataset_manifest.json --allow-install`, or add `--archive /path/to/local/dataset.zip` for an existing ZIP.

## Optional CUDA and Final validation

```bash
python scripts/build.py --cuda
build/synthetic_pipeline_test out/synthetic_v1 --version 1
build/synthetic_pipeline_test out/synthetic_v2 --version 2
build/synthetic_pipeline_test out/synthetic_v3 --version 3
python scripts/benchmark.py out/prepared/manifest.prepared.json --executable build/stitch --runs 5 --versions 0 1 2 3 --output out/gpu_benchmark
```

CUDA requires nvcc, its compatible host compiler and an NVIDIA device. CPU builds reject CUDA versions; requested GPU execution never silently falls back to V0. Read [GPU validation](docs/gpu_validation.md), inspect the source mapping, run correctness first and retain hardware/driver/runtime metadata and Nsight profiles with GPU measurements. The current CPU report makes no GPU performance claim.

## Package the Midterm

```bash
python scripts/package_submission.py
python scripts/package_submission.py --strict
```

The first command makes a source/report draft and prints remaining team items. Strict mode requires saved outputs matching source/config, authentic team metadata, dataset coverage and measured standard pairs at all sizes. The zip contains the report, src/tools/scripts/docs, CMake/README and manifest/team config. It excludes build products, raw photographs and generated files; share the original/prepared dataset separately through the notebook link.

Locally, save the notebook with all outputs after Run all. In Colab, wait for Run all to finish, then use the final cell's **Export & strict ZIP** button. It retrieves the active notebook including current outputs, passes its explicit path through `--notebook` and downloads the ZIP only if strict validation passes. Colab's frontend export request is internal and can change; **Upload saved .ipynb** accepts a notebook downloaded with outputs when that request is unavailable. A notebook copied during setup is only a source copy and is never treated as the current Colab output. This export fallback can also be used explicitly:

```bash
python scripts/package_submission.py --strict --notebook /path/to/saved/Group08_Midterm.ipynb
```

Image I/O uses [stb](https://github.com/nothings/stb); upstream license remains in the headers. Algorithm references and AI assistance are described in the report. Each modified source file lists authors and external sources; the group must confirm actual contributor names.
