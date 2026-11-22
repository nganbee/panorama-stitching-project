"""Authors: Nguyen Duy Khanh; assisted by OpenAI Codex.
External sources: none. Baseline -> independent profiling -> optimized results.
Verification is untimed. Raw samples come from production pipelines.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

SITES = ('location1', 'location2', 'location3', 'challenge')
SIZES = ('small_720p', 'medium_1080p', 'large_4k')


def cases():
    for site in SITES:
        for size in SIZES:
            pair = Path('dataset') / site / size
            yield f'{site}_{size}', pair / '1.jpg', pair / '2.jpg'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment():
    result = {}
    for key, command in {
        'gpu': ['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'],
        'cuda': ['nvcc', '--version'], 'compiler': ['g++', '--version'],
        'commit': ['git', 'rev-parse', 'HEAD'], 'worktree': ['git', 'status', '--short'],
    }.items():
        run = subprocess.run(command, capture_output=True, text=True)
        result[key] = {'returncode': run.returncode, 'text': run.stdout + run.stderr}
        if key in ('gpu', 'cuda', 'compiler') and run.returncode:
            raise RuntimeError(f'Cannot record {key}: {result[key]["text"]}')
    result['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    inputs = [p for _, left, right in cases() for p in (left, right)]
    result['dataset_sha256'] = fingerprint(inputs)
    result['source_sha256'] = fingerprint([*Path('src').rglob('*.*'), *Path('tools').rglob('*.cpp'),
                                            *Path('tools').rglob('*.cu'), *Path('tools').rglob('*.h'),
                                            *Path('tools').rglob('*.py'), Path('tools/v2/build.sh')])
    result['binaries_sha256'] = fingerprint([Path('build/benchmark_v2_pair'), Path('build/verify_v2_pair')])
    return result


def load_report(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def invoke(command, log, report):
    if report.exists():
        report.unlink()
    run = subprocess.run([str(x) for x in command], capture_output=True, text=True)
    log.write_text(run.stdout + run.stderr, encoding='utf-8')
    if not report.is_file():
        write_json(report, {'status': 'process_error', 'returncode': run.returncode,
                            'error': run.stderr.strip() or 'Process produced no report'})
    data = load_report(report)
    if run.returncode and data.get('status') in ('ok', 'pass'):
        data['status'] = 'process_error'
        data['error'] = f'Process returned {run.returncode} despite success report'
        write_json(report, data)
    return data


def timing_fields(row, version, data):
    if not data or data.get('status') != 'ok':
        row[f'v{version}_error'] = (data or {}).get('error', 'Not measured')
        return False
    for key, value in data['timings_ms'].items():
        row[f'v{version}_{key}_ms'] = value
    stages = data['stage_timings_ms']
    row.update({f'v{version}_{key}_ms': value for key, value in stages.items()})
    row.update({f'v{version}_{key}': value for key, value in data.get('features', {}).items()})
    panorama = data.get('panorama', {})
    row[f'v{version}_panorama_width'] = panorama.get('width')
    row[f'v{version}_panorama_height'] = panorama.get('height')
    samples = data.get('samples_ms', [])
    if len(samples) < 5:
        raise ValueError('Benchmark requires at least five raw timed samples')
    samples = [dict(sample, S1=sample['gray']+sample['blur'],
                    S2=sample['harris']+sample['nms_topk'], S3=sample['describe'],
                    S4=sample['match'], S5=sample['ransac'], S6=sample['warp_blend']) for sample in samples]
    for key in {**data['timings_ms'], **stages}:
        values = [float(sample[key]) for sample in samples]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f'Invalid raw timing for V{version} {key}')
        mean = statistics.mean(values)
        reported = data['timings_ms'].get(key, stages.get(key))
        if not math.isclose(mean, reported, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f'Raw timing mean disagrees with report for {key}')
        row[f'v{version}_{key}_min_ms'] = min(values)
        row[f'v{version}_{key}_max_ms'] = max(values)
        row[f'v{version}_{key}_std_ms'] = statistics.stdev(values)
        row[f'v{version}_{key}_cv'] = statistics.stdev(values) / mean if mean else 0
    if version:
        total = data['timings_ms']['kernel_total']
        if total <= 0 or not math.isclose(sum(stages.values()), total, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError('Stage kernel sum disagrees with kernel_total')
        if any(data['timings_ms'][key] <= 0 for key in ('h2d', 'd2h', 'end_to_end')):
            raise ValueError('GPU timing categories not populated')
    return True


def summarize(directory, phase):
    rows = []
    for case, _, _ in cases():
        row = {'case': case, 'status': 'baseline_only' if phase == 'baseline' else 'incomplete'}
        valid = {}
        for version in (0, 1, 2):
            data = load_report(directory / f'{case}_v{version}.json')
            valid[version] = timing_fields(row, version, data) if data else False
        if valid[1]:
            row['v1_S4_kernel_share'] = row['v1_match_ms'] / row['v1_kernel_total_ms']
        verification = load_report(directory / f'{case}_verification.json')
        if verification:
            row['verification_status'] = verification['status']
            row.update(verification.get('metrics', {}))
            row['failed_checks'] = '; '.join(k for k,v in verification.get('checks', {}).items() if not v)
            if verification['status'] == 'pass' and all(valid.values()):
                row['status'] = 'pass'
                for version in (0, 1):
                    for metric in ('match', 'end_to_end'):
                        denominator = row[f'v2_{metric}_ms']
                        row[f'v{version}_over_v2_{metric}_speedup'] = (
                            row[f'v{version}_{metric}_ms'] / denominator if denominator > 0 else None)
            else:
                row['status'] = verification['status'] if verification['status'] != 'pass' else 'benchmark_failed'
        elif phase == 'optimized':
            row['status'] = 'verification_missing'
        if phase == 'baseline' and not (valid[0] and valid[1]):
            row['status'] = 'baseline_pipeline_failure'
        rows.append(row)
    write_json(directory / 'summary.json', rows)
    with (directory / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('baseline', 'optimized'), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir
    current = environment()
    if args.phase == 'baseline':
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'environment.json', current)
    else:
        saved = load_report(directory / 'environment.json')
        stable = ('gpu', 'cuda', 'compiler', 'commit', 'boot_id', 'dataset_sha256',
                  'source_sha256', 'binaries_sha256')
        if not saved or any(saved.get(key) != current.get(key) for key in stable):
            raise RuntimeError('GPU/session/source/input/build changed; start a fresh baseline run')
    images = directory / 'panoramas'
    images.mkdir(exist_ok=True)
    print(current['gpu']['text'], flush=True)
    for case, left, right in cases():
        print(args.phase, case, flush=True)
        if args.phase == 'optimized':
            invoke(['build/verify_v2_pair', left, right, directory / f'{case}_verification.json'],
                   directory / f'{case}_verification.log', directory / f'{case}_verification.json')
        for version in ((0, 1) if args.phase == 'baseline' else (2,)):
            invoke(['build/benchmark_v2_pair', left, right, images / f'{case}_v{version}.png',
                    version, 5, directory / f'{case}_v{version}.json'],
                   directory / f'{case}_v{version}.log', directory / f'{case}_v{version}.json')
        summarize(directory, args.phase)
    rows = summarize(directory, args.phase)
    print('Saved:', directory / 'summary.csv', flush=True)
    print('Statuses:', {r['case']: r['status'] for r in rows}, flush=True)
    if args.phase == 'optimized' and any(r['status'] != 'pass' for r in rows):
        raise SystemExit('Some cases have no verified panorama. Inspect saved failure analysis; do not report them as PASS.')


if __name__ == '__main__':
    main()
