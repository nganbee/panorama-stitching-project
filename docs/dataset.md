# Dataset manifest and measurement contract

Copy `dataset_manifest.example.json` to `dataset_manifest.json` in the project root. Paths are relative to the manifest file (absolute paths are also accepted for local use or mounted Drive files). Use forward slashes and avoid absolute paths or `..` when distributing the dataset archive. Arbitrary relative photo folders such as `photos/site1/` and `originals/site1/` are supported; setup copies referenced existing files with that layout rather than assuming the folder names `data/` and `raw/`.

Required keys: `schema_version: 1`, `dataset_url`, `sites`, `pairs`. Every site has a unique `site_id`, human-readable `display_name`, and `registration` recording the actual approved site. Every pair has a filename-safe unique `pair_id`, existing `site_id`, `kind` of `standard` or `challenging`, and existing `left`/`right` prepared-source photographs. `raw_left`/`raw_right` identify the original photos inside the linked dataset archive; `notes` explains challenging conditions and preparation.

The authoritative midterm PDF requires at least three sites, one standard pair for every registered site and at least one challenging pair overall. `--strict` checks coverage/link/metadata; it cannot independently verify course registration or access permissions for a shared dataset URL. The group must verify the link can be downloaded by the instructor.

`dataset_url` must point to a public downloadable ZIP containing the originals and the prepared-source photographs at the relative paths in the manifest, together with the prepared dataset/manifest required for sharing. A direct HTTP(S) ZIP URL or a Google Drive **file** link shared as **Anyone with the link** is supported. Folder links, Google Docs links, login pages and private share links are not public ZIP downloads. The ZIP may wrap these paths in one enclosing folder. Its manifest cannot overwrite the source package's configuration; only missing image files referenced by the submitted manifest are copied from staging. Traversal paths, links, duplicate entries and oversized archives are rejected. Optional `dataset_archive_sha256` verifies the downloaded ZIP against the team's configured hash.

The report invokes `resolve_dataset.py` before opening real photos. If the photos are already available it downloads nothing; otherwise it downloads the public ZIP and resolves all image/raw paths. A Google Drive transfer uses [gdown's download API](https://github.com/wkentaro/gdown/blob/main/gdown/download.py), with cookies disabled; Colab installs `gdown>=5.2,<6` only if needed. Direct URLs use Python's standard library. For a private Drive ZIP, mount Drive before Run all and set `DATASET_ARCHIVE` to its existing absolute path. Existing mounted absolute image paths can also be used, but a portable manifest/public ZIP is preferable for reproducibility. No resolver can grant access to an inaccessible link.

```bash
python scripts/resolve_dataset.py dataset_manifest.json --allow-install
python scripts/resolve_dataset.py dataset_manifest.json --archive /path/to/dataset.zip
```

```bash
python scripts/prepare_dataset.py dataset_manifest.json --validate-only --strict
python scripts/prepare_dataset.py dataset_manifest.json --output out/prepared
python scripts/benchmark.py out/prepared/manifest.prepared.json --executable build/stitch --runs 5
```

On Windows use `build/stitch.exe`. Prepared output has one pair per small/medium/large size. The bounding boxes are 1280×720, 1920×1080 and 3840×2160; images with another aspect ratio fit inside the box. Both input images receive the same uniform scale `min(target_w/max_pair_w, target_h/max_pair_h)` after EXIF orientation; dimensions are rounded to whole pixels and recorded, and RGB PNGs are written. Using unrelated stretch factors would change correspondence geometry and is avoided.

Results contain pair/site/kind/version, actual image dimensions, warm-up/timed-run counts, all six stage times, end-to-end time, transfers/kernel time, feature counts, observed failure messages, hotspot ranking and Amdahl fraction/limit. The result may record a failed difficult pair; failure is not overwritten by an identity homography or fake output. Keep that case for explanation. At least one successful panorama per site is needed for the report.

For version comparison, use `--versions 0 1 2 3` only after CUDA build and correctness verification. All versions use the same prepared images and runs in one invocation. `--accelerate S1 S2 S3 S4 S6` controls the Amdahl plan; change it based on real profiling. End-to-end is the Amdahl denominator. Amdahl is an ideal bound, not an achieved GPU result.

The dataset archive linked from the notebook must include originals, manifest-referenced source photographs and the prepared pair files/manifest. The source submission archive deliberately excludes photographs, `out/`, `build/` and binaries; report image/chart outputs are stored inside the notebook. Store the dataset archive separately and set `dataset_url` to its downloadable link. Verify Run all on a fresh Colab session with only the source submission available so that this retrieval path is exercised.
