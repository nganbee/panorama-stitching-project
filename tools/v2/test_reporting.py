"""Authors: Team 8; assisted by OpenAI Codex. External sources: none.
Host-only tests of reporting: no GPU timing or correctness claims.
Run: python -m unittest discover -s tools/v2 -p test_benchmark.py
"""
import tempfile
import unittest
import sys
import json
from pathlib import Path
from unittest.mock import patch

import benchmark as bench


def report():
    sample = dict(gray=1., blur=1., harris=1., nms_topk=1., describe=1., match=1.,
                  ransac=1., warp_blend=1., kernel_total=8., h2d=1., d2h=1., end_to_end=12.)
    return dict(status='ok', timings_ms=sample,
                stage_timings_ms=dict(S1=2., S2=2., S3=1., S4=1., S5=1., S6=1.),
                samples_ms=[dict(sample) for _ in range(5)])


class ReportingTests(unittest.TestCase):
    def test_notebook_result_cells_handle_valid_failed_and_mixed_cases(self):
        try:
            import pandas as pd
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest('Presentation test requires pandas, numpy and Pillow')
        class Dummy:
            def __getattr__(self, name): return self
            def __call__(self, *args, **kwargs): return self
        artifact = Path(__file__).resolve().parents[2]/'test_v2.ipynb'
        notebook = json.loads(artifact.read_text(encoding='utf-8'))
        markers = ('verification_columns =', 'stage_rows = []', 'dispersion_rows = []',
                   'resolution_rows = []', "display(Markdown(f\"**{row['case']}",
                   'faster = valid[', 'checklist = pd.DataFrame')
        selected = [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'
                    and any(marker in ''.join(c['source']) for marker in markers)]
        self.assertEqual(len(selected), 7)
        for statuses in (['pass','pass'], ['correctness_failed','correctness_failed'], ['pass','error']):
            with tempfile.TemporaryDirectory() as root, patch.object(bench, 'cases', return_value=[
                    ('location1_small_720p',Path('a'),Path('b')), ('challenge_large_4k',Path('c'),Path('d'))]), \
                 patch.object(pd.DataFrame, 'plot', new=Dummy()):
                directory=Path(root)
                for (case,_,_), status in zip(bench.cases(), statuses):
                    for version in (0,1,2):
                        bench.write_json(directory/f'{case}_v{version}.json', report())
                    bench.write_json(directory/f'{case}_verification.json',
                                     dict(status=status, checks={'ordered_matches':status=='pass'},metrics={}))
                bench.summarize(directory, 'optimized')
                summary=pd.read_csv(directory/'summary.csv')
                scope = dict(pd=pd, np=np, json=json, Path=Path, PILImage=Image, plt=Dummy(),
                             display=lambda *a,**kw:None, Markdown=lambda x:x,
                             summary=summary, RUN_DIR=directory,
                             sizes={'small_720p':(1280,720),'large_4k':(3840,2160)}, dataset_rows=[{},{}],
                             sanitizer_results={}, profile_v1={'status':'unavailable'},profile_v2={'status':'unavailable'})
                for source in selected:
                    exec(compile(source,'notebook presentation fixture','exec'),scope)
                self.assertTrue((directory/'stage_comparison.csv').is_file())
                self.assertTrue((directory/'dispersion.csv').is_file())

    def test_baseline_and_optimized_phases_preserve_all_cases(self):
        pair_cases = [('bad', Path('bad_left'), Path('right')), ('good', Path('good_left'), Path('right'))]
        env = {'gpu': {'text': 'test-only GPU metadata'}, 'commit': 'test-only commit'}
        def fake_invoke(command, log, path):
            if command[0] == 'build/verify_v2_pair':
                ok = str(command[1]) == 'good_left'
                value = dict(status='pass' if ok else 'correctness_failed',
                             checks={'ordered_matches': ok}, metrics={})
            else:
                value = report()
            bench.write_json(path, value)
            return value
        with tempfile.TemporaryDirectory() as root, patch.object(bench, 'cases', return_value=pair_cases), \
             patch.object(bench, 'environment', return_value=env), patch.object(bench, 'invoke', side_effect=fake_invoke):
            directory = Path(root)/'run'
            with patch.object(sys, 'argv', ['benchmark.py','--phase','baseline','--run-dir',str(directory)]):
                bench.main()
            # Untracked reports change worktree status; that must not invalidate
            # a comparison whose source, inputs and stable metadata are unchanged.
            changed = dict(env, worktree='reports created')
            with patch.object(bench, 'environment', return_value=changed), \
                 patch.object(sys, 'argv', ['benchmark.py','--phase','optimized','--run-dir',str(directory)]):
                with self.assertRaises(SystemExit):
                    bench.main()
            rows = bench.load_report(directory/'summary.json')
            self.assertEqual([r['status'] for r in rows], ['correctness_failed','pass'])
            self.assertNotIn('v1_over_v2_match_speedup', rows[0])
            self.assertIn('v1_over_v2_match_speedup', rows[1])
            with patch.object(bench, 'environment', return_value=dict(env, commit='different')), \
                 patch.object(sys, 'argv', ['benchmark.py','--phase','optimized','--run-dir',str(directory)]):
                with self.assertRaises(RuntimeError):
                    bench.main()

    def test_raw_samples_must_agree_with_mean(self):
        data = report()
        data['samples_ms'][0]['match'] = 10.
        with self.assertRaises(ValueError):
            bench.timing_fields({}, 2, data)

    def test_refuse_missing_or_nonfinite_samples(self):
        for mutation in ('missing', 'nan'):
            data = report()
            if mutation == 'missing': data['samples_ms'].pop()
            else: data['samples_ms'][0]['match'] = float('nan')
            with self.assertRaises(ValueError):
                bench.timing_fields({}, 2, data)

    def test_stage_dispersion_and_minimum(self):
        row = {}
        bench.timing_fields(row, 2, report())
        self.assertEqual(row['v2_S1_min_ms'], 2.)
        self.assertEqual(row['v2_match_std_ms'], 0.)

    def test_failure_is_not_speedup_or_correctness_pass(self):
        with tempfile.TemporaryDirectory() as root, patch.object(bench, 'cases', return_value=[('case', Path('a'), Path('b'))]):
            directory = Path(root)
            for version in (0,1,2):
                bench.write_json(directory / f'case_v{version}.json', report())
            for status in ('correctness_failed', 'consistent_pipeline_failure', 'error'):
                bench.write_json(directory / 'case_verification.json', dict(status=status, checks={'ordered_matches': False}))
                row = bench.summarize(directory, 'optimized')[0]
                self.assertEqual(row['status'], status)
                self.assertNotIn('v1_over_v2_match_speedup', row)
            bench.write_json(directory / 'case_verification.json', dict(status='pass', checks={'ordered_matches': True}))
            row = bench.summarize(directory, 'optimized')[0]
            self.assertEqual(row['status'], 'pass')
            self.assertEqual(row['v1_over_v2_match_speedup'], 1.)


if __name__ == '__main__':
    unittest.main()
