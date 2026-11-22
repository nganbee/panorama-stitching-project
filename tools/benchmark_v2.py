"""Authors: Nguyen Duy Khanh; assisted by OpenAI Codex.
External sources: none. Run V0/V1/V2 in one GPU session; never invent results.
"""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    out = Path('out/v2_benchmark')
    reports = Path('reports/v2_benchmark')
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, command in {
        'gpu': ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
        'cuda': ['nvcc', '--version'],
        'compiler': ['g++', '--version'],
        'commit': ['git', 'rev-parse', 'HEAD'],
        'worktree': ['git', 'status', '--short'],
    }.items():
        result = subprocess.run(command, capture_output=True, text=True)
        metadata[name] = result.stdout + result.stderr
    (reports / 'environment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(metadata['gpu'])
    rows = []
    for site in ['location1', 'location2', 'location3', 'challenge']:
        for size in ['small_720p', 'medium_1080p', 'large_4k']:
            case = f'{site}_{size}'
            pair = Path('dataset') / site / size
            results = {}
            row = {'case': case, 'status': 'failed'}
            for version in [0, 1, 2]:
                label = f'{case}_v{version}'
                image = out / f'{label}.png'
                report = reports / f'{label}.json'
                # Remove stale reports from a previous execution before running.
                if report.exists():
                    report.unlink()
                command = ['build/stitch_v2', str(pair / '1.jpg'), str(pair / '2.jpg'),
                           str(image), '--version', str(version), '--runs', '5', '--json', str(report)]
                result = subprocess.run(command, capture_output=True, text=True)
                (reports / f'{label}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
                if result.returncode != 0 or not report.exists():
                    row[f'v{version}_error'] = result.stderr.strip() or 'No metrics report'
                    continue
                data = json.loads(report.read_text(encoding='utf-8'))
                if data.get('status') != 'ok':
                    row[f'v{version}_error'] = data.get('error', 'Pipeline failed')
                    continue
                results[version] = (data, image)
                for key, value in data['timings_ms'].items():
                    row[f'v{version}_{key}_ms'] = value
                for stage, value in data['stage_timings_ms'].items():
                    row[f'v{version}_{stage}_ms'] = value
            if len(results) == 3:
                correctness = True
                target, target_image = results[2]
                for version in [0, 1]:
                    ref, ref_image = results[version]
                    features_ok = all(ref['features'][key] == target['features'][key]
                                      for key in ['keypoints_A', 'keypoints_B', 'matches', 'inliers'])
                    h_ok = bool(np.allclose(ref['homography_B_to_A'], target['homography_B_to_A'],
                                           rtol=0, atol=1e-5 if version == 0 else 0))
                    a = np.asarray(Image.open(ref_image).convert('RGB'), dtype=np.int16)
                    b = np.asarray(Image.open(target_image).convert('RGB'), dtype=np.int16)
                    same_shape = a.shape == b.shape
                    error = int(np.abs(a - b).max()) if same_shape else None
                    pixels_ok = same_shape and error <= (1 if version == 0 else 0)
                    row[f'v2_vs_v{version}_max_pixel_error'] = error
                    row[f'v2_vs_v{version}_features_ok'] = features_ok
                    row[f'v2_vs_v{version}_homography_ok'] = h_ok
                    row[f'v2_vs_v{version}_pixels_ok'] = pixels_ok
                    correctness &= features_ok and h_ok and pixels_ok
                row['status'] = 'pass' if correctness else 'correctness_failed'
                # Invalid output must not be presented as a successful speedup.
                if correctness:
                    for version in [0, 1]:
                        for metric in ['match', 'end_to_end']:
                            row[f'v{version}_over_v2_{metric}_speedup'] = (
                                row[f'v{version}_{metric}_ms'] / row[f'v2_{metric}_ms'])
                    row['v1_S4_kernel_share'] = row['v1_match_ms'] / row['v1_kernel_total_ms']
            rows.append(row)
            print(case, row['status'], flush=True)
            (reports / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (reports / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failures = [row['case'] for row in rows if row['status'] != 'pass']
    print('Failed cases:', failures)
    if failures:
        raise SystemExit('Benchmark incomplete or correctness failed; inspect saved logs and reports.')


if __name__ == '__main__':
    main()
