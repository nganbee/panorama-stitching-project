"""Authors: Nguyen Duy Khanh; assisted by OpenAI Codex.
External sources: none. Baseline -> independent profiling -> optimized results.
Verification is untimed. Raw samples come from production pipelines.
"""
import argparse
import csv
import json
import math
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
    result['session'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
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
        mean = sum(values) / len(values)
        reported = data['timings_ms'].get(key, stages.get(key))
        if not math.isclose(mean, reported, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f'Raw timing mean disagrees with report for {key}')
        row[f'v{version}_{key}_min_ms'] = min(values)
    if version:
        total = data['timings_ms']['kernel_total']
        if total <= 0 or not math.isclose(sum(stages.values()), total, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError('Stage kernel sum disagrees with kernel_total')
        if any(data['timings_ms'][key] <= 0 for key in ('h2d', 'd2h', 'end_to_end')):
            raise ValueError('GPU timing categories not populated')
    return True


def validate_pair(directory, case):
    manifest = load_report(directory / f'{case}_paired.json') or {}
    runs = manifest.get('runs_per_version', 0)
    order = manifest.get('order', [])
    if (manifest.get('status') != 'ok' or
            manifest.get('measurement_mode') != 'same_process_interleaved' or
            not isinstance(runs, int) or runs < 5 or len(order) != 2*runs or
            manifest.get('warmup_runs_per_version', 0) < 1):
        return False, 'Missing or invalid same-process measurement manifest'
    start = order[:2]
    if start not in ([5,6], [6,5]):
        return False, 'Invalid paired order'
    for iteration in range(runs):
        expected = start if iteration % 2 == 0 else start[::-1]
        if order[2*iteration:2*iteration+2] != expected:
            return False, 'Paired order does not alternate within this case'
    gpu_names=[]
    for version in (5,6):
        data=load_report(directory / f'{case}_v{version}.json') or {}
        expected_positions=[i for i,v in enumerate(order) if v == version]
        if (data.get('status') != 'ok' or data.get('runs') != runs or
                data.get('warmup_runs',0) < 1 or len(data.get('samples_ms',[])) != runs or
                data.get('measurement_mode') != 'same_process_interleaved' or
                data.get('sample_positions') != expected_positions):
            return False, f'V{version} samples do not match paired measurement order'
        gpu_names.append(data.get('gpu_name'))
    if not gpu_names[0] or gpu_names[0] != gpu_names[1]:
        return False, 'GPU labels disagree between paired reports'
    return True, ''


def invoke_pair(left, right, directory, case, first):
    # A single child process loads the images once and runs both GPU versions.
    for version in (5,6):
        report=directory / f'{case}_v{version}.json'
        if report.exists(): report.unlink()
    manifest=directory / f'{case}_paired.json'
    result=invoke(['build/benchmark_v6_pair','--compare',left,right,directory,case,5,first],
                  directory / f'{case}_paired.log',manifest)
    for version in (5,6):
        report=directory / f'{case}_v{version}.json'
        if result.get('status') != 'ok' or not report.is_file():
            write_json(report, {'status':'process_error', 'error':result.get('error','Paired process produced no report')})


def summarize(directory, phase):
    rows = []
    for case, _, _ in cases():
        row = {'case': case, 'status': 'baseline_only' if phase == 'baseline' else 'incomplete'}
        valid = {}
        for version in (0, 5, 6):
            data = load_report(directory / f'{case}_v{version}.json')
            valid[version] = timing_fields(row, version, data) if data else False
        if valid[5]:
            row['v5_S4_kernel_share'] = row['v5_match_ms'] / row['v5_kernel_total_ms']
        verification = load_report(directory / f'{case}_verification.json')
        if verification:
            row['verification_status'] = verification['status']
            row.update(verification.get('metrics', {}))
            row['failed_checks'] = '; '.join(k for k,v in verification.get('checks', {}).items() if not v)
            paired_ok, paired_error = validate_pair(directory, case) if phase == 'optimized' else (True, '')
            if paired_error: row['measurement_error'] = paired_error
            if verification['status'] == 'pass' and all(valid.values()) and paired_ok:
                row['status'] = 'pass'
                for version in (0, 5):
                    for metric in ('match', 'end_to_end'):
                        denominator = row[f'v6_{metric}_ms']
                        row[f'v{version}_over_v6_{metric}_speedup'] = (
                            row[f'v{version}_{metric}_ms'] / denominator if denominator > 0 else None)
            else:
                row['status'] = verification['status'] if verification['status'] != 'pass' else 'benchmark_failed'
        elif phase == 'optimized':
            row['status'] = 'verification_missing'
        if phase == 'baseline' and not (valid[0] and valid[5]):
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
        if not saved:
            raise RuntimeError('Run the baseline phase first in the same Colab GPU session')
        if saved['gpu']['text'] != current['gpu']['text'] or saved['session'] != current['session']:
            raise RuntimeError('GPU identity changed; rerun the notebook in one session')
        # Keep the baseline used to choose the optimization intact.
        directory = directory / 'comparison'
        directory.mkdir(exist_ok=False)
        write_json(directory / 'environment.json', current)
    images = directory / 'panoramas'
    images.mkdir(exist_ok=True)
    print(current['gpu']['text'], flush=True)
    for case_index, (case, left, right) in enumerate(cases()):
        print(args.phase, case, flush=True)
        if args.phase == 'optimized':
            invoke(['build/verify_v6_pair', left, right, directory / f'{case}_verification.json'],
                   directory / f'{case}_verification.log', directory / f'{case}_verification.json')
        # V0 is measured separately in the same session. GPU versions share
        # one process, with five timed samples each; flip the starting version by case.
        for version in ((0,5) if args.phase == 'baseline' else (0,)):
            invoke(['build/benchmark_v6_pair', left, right, images / f'{case}_v{version}.png',
                    version, 5, directory / f'{case}_v{version}.json'],
                   directory / f'{case}_v{version}.log', directory / f'{case}_v{version}.json')
        if args.phase == 'optimized':
            invoke_pair(left,right,directory,case,5 if case_index % 2 == 0 else 6)
        summarize(directory, args.phase)
    rows = summarize(directory, args.phase)
    print('Saved:', directory / 'summary.csv', flush=True)
    print('Statuses:', {r['case']: r['status'] for r in rows}, flush=True)
    if args.phase == 'optimized' and any(r['status'] != 'pass' for r in rows):
        raise SystemExit('Some cases have no verified panorama. Inspect saved failure analysis; do not report them as PASS.')


if __name__ == '__main__':
    main()
