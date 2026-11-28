"""Authors: Nguyen Duy Khanh; assisted by OpenAI Codex.
External sources: NVIDIA Nsight Systems and Nsight Compute CLI documentation.
Profiler timings are diagnostic evidence, never benchmark samples.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from benchmark import cases, environment, load_report, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--version', type=int, choices=(5, 6), default=5)
    parser.add_argument('--case', choices=[case for case, _, _ in cases()],
                        help='Profile this exact case instead of selecting the largest V5 S4 time')
    args = parser.parse_args()
    directory = args.run_dir / 'profiling'
    if args.case:
        directory = directory / args.case
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'unavailable', 'version': args.version, 'attempts': [],
                'note': 'Not benchmark timing. Inspect actual profiler output before concluding bottleneck.'}
    manifest['environment'] = environment()
    candidates = []
    for case, left, right in cases():
        if args.case and case != args.case:
            continue
        data = load_report(args.run_dir / f'{case}_v5.json')
        if data and data.get('status') == 'ok':
            candidates.append((data['timings_ms']['match'], case, left, right))
    if not candidates:
        manifest['status'] = 'baseline_unavailable'
    else:
        _, case, left, right = max(candidates)
        manifest['case'] = case
        prefix = directory / f'{case}_v{args.version}'
        expected = 'nearest_tiled_kernel' if args.version == 5 else 'nearest_warp_kernel'
        target = ['build/benchmark_v6_pair', str(left), str(right), str(prefix)+'.png',
                  str(args.version), '5', str(prefix)+'_diagnostic_timing.json']

        def invoke(command, name):
            print('$', ' '.join(command), flush=True)
            result = subprocess.run(command, capture_output=True, text=True)
            text = result.stdout + result.stderr
            (directory / name).write_text(text, encoding='utf-8')
            manifest['attempts'].append({'command': command, 'returncode': result.returncode,
                                         'log': str(directory / name)})
            print(text, flush=True)
            return result.returncode, text

        if shutil.which('nsys'):
            code, _ = invoke(['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                              '--force-overwrite=true', '-o', str(prefix), *target], f'v{args.version}_nsys.log')
            report = Path(str(prefix)+'.nsys-rep')
            if code == 0 and report.is_file() and report.stat().st_size:
                code, text = invoke(['nsys', 'stats', '--report', 'cuda_gpu_kern_sum,cuda_gpu_mem_time_sum',
                                     '--format', 'csv', str(report)], f'v{args.version}_nsys_stats.csv')
                if code == 0 and expected in text:
                    manifest.update(status='collected', tool='nsys', report=str(report),
                                    summary=str(directory / f'v{args.version}_nsys_stats.csv'))
        if manifest['status'] != 'collected' and shutil.which('ncu'):
            code, _ = invoke(['ncu', '--set', 'basic', '--kernel-name', 'regex:.*nearest.*',
                              '--launch-count', '2', '--force-overwrite', '--export', str(prefix), *target],
                             f'v{args.version}_ncu.log')
            report = Path(str(prefix)+'.ncu-rep')
            if code == 0 and report.is_file() and report.stat().st_size:
                code, text = invoke(['ncu', '--import', str(report), '--page', 'raw', '--csv'],
                                     f'v{args.version}_ncu_stats.csv')
                if code == 0 and expected in text:
                    manifest.update(status='collected', tool='ncu', report=str(report),
                                    summary=str(directory / f'v{args.version}_ncu_stats.csv'))
    write_json(directory / f'profile_v{args.version}.json', manifest)
    print('Profiling status:', manifest['status'], flush=True)
    if manifest['status'] != 'collected':
        print('PROFILING NOT VERIFIED: tool unavailable, permission denied, or capture failed. '
              'Keep logs; collect real evidence on a supported GPU environment. Timing alone does not replace it.')


if __name__ == '__main__':
    main()
