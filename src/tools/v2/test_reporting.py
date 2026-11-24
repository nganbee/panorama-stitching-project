"""Authors: Team 8; assisted by OpenAI Codex. External sources: none.
Development tests only; not part of the submitted experiment workflow.
Run: python -m unittest discover -s src/tools/v2 -p test_reporting.py
"""
import tempfile
import unittest
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
    def test_sample_mean_must_match(self):
        data=report(); data['samples_ms'][0]['match']=10.
        with self.assertRaises(ValueError): bench.timing_fields({},2,data)

    def test_reject_insufficient_or_nonfinite_samples(self):
        for invalid in ('missing','nan'):
            data=report()
            if invalid=='missing': data['samples_ms'].pop()
            else: data['samples_ms'][0]['match']=float('nan')
            with self.assertRaises(ValueError): bench.timing_fields({},2,data)

    def test_minimum_saved_for_stages(self):
        row={}; bench.timing_fields(row,2,report())
        self.assertEqual(row['v2_S1_min_ms'],2.)
        self.assertEqual(row['v2_match_min_ms'],1.)
        self.assertNotIn('v2_match_cv',row)

    def test_failed_cases_have_no_successful_speedup(self):
        with tempfile.TemporaryDirectory() as root, patch.object(bench,'cases',return_value=[('case',Path('a'),Path('b'))]):
            directory=Path(root)
            for version in (0,1,2): bench.write_json(directory/f'case_v{version}.json',report())
            for status in ('correctness_failed','consistent_pipeline_failure','error','pass'):
                bench.write_json(directory/'case_verification.json',dict(status=status,checks={'matches':status=='pass'}))
                row=bench.summarize(directory,'optimized')[0]
                self.assertEqual(row['status'],status)
                self.assertEqual('v1_over_v2_match_speedup' in row,status=='pass')


if __name__=='__main__': unittest.main()
